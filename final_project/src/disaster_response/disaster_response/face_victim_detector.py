import rclpy
from rclpy.node import Node
from message_filters import ApproximateTimeSynchronizer, Subscriber
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from tf2_ros import TransformException, Buffer, TransformListener
from cv_bridge import CvBridge
import cv2
import numpy as np
import struct
import math
import os
import json

HAAR      = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
MODEL_DIR = '/root/codes/models'
SFACE_MODEL  = os.path.join(MODEL_DIR, 'face_recognition_sface_2021dec.onnx')
ENROLL_FILE  = os.path.join(MODEL_DIR, 'enrolled_faces.json')
FACE_SIZE    = (112, 112)
RECOGNITION_THRESHOLD = 0.65

_CASCADE = None

def _get_cascade():
    global _CASCADE
    if _CASCADE is None:
        _CASCADE = cv2.CascadeClassifier(HAAR)
    return _CASCADE


def _q2R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


class FaceVictimDetector(Node):

    def __init__(self):
        super().__init__('face_victim_detector')

        self.declare_parameter('cooldown',      5.0)
        self.declare_parameter('min_face_size', 60)

        self.br = CvBridge()
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._last_time  = 0.0
        self._unknown_count = 0
        self._frame_count   = 0

        _get_cascade()

        if not os.path.exists(SFACE_MODEL):
            raise RuntimeError(f'Missing SFace model: {SFACE_MODEL}')
        self._recogniser = cv2.FaceRecognizerSF.create(SFACE_MODEL, '')

        self._enrolled = {}
        self._load_enrolled()

        self.pub_pose  = self.create_publisher(PoseStamped, '/victim_found_pose',  10)
        self.pub_image = self.create_publisher(Image,       '/victim_found_image', 10)
        self.pub_id    = self.create_publisher(String,      '/victim_found_id',    10)
        self.pub_debug = self.create_publisher(Image,       '/face_detector/debug_image', 10)

        self.sub_rgb   = Subscriber(self, Image,       '/camera/color/image_raw')
        self.sub_depth = Subscriber(self, PointCloud2, '/camera/depth/points')
        self.ts = ApproximateTimeSynchronizer([self.sub_rgb, self.sub_depth], 10, 0.15)
        self.ts.registerCallback(self._camera_cb)

        enrolled_names = list(self._enrolled.keys()) or ['none — run enroll_faces first']
        self.get_logger().info(f'FaceVictimDetector ready | enrolled: {enrolled_names}')
        self.get_logger().info('Debug image → /face_detector/debug_image')

    def _load_enrolled(self):
        if not os.path.exists(ENROLL_FILE):
            self.get_logger().warn('No enrolled faces — all detections will be UNKNOWN')
            return
        with open(ENROLL_FILE) as f:
            data = json.load(f)
        for name, feat_list in data.items():
            # Reshape to (1, N) to match the shape returned by recogniser.feature()
            self._enrolled[name] = np.array(feat_list, dtype=np.float32).reshape(1, -1)
        self.get_logger().info(
            f'Loaded {len(self._enrolled)} enrolled face(s): {list(self._enrolled.keys())}')

    def _identify(self, feature: np.ndarray):
        """Returns (name, score). feature must be shape (1, N)."""
        best_name, best_score = None, -1.0
        for name, enrolled_feat in self._enrolled.items():
            score = self._recogniser.match(feature, enrolled_feat,
                                           cv2.FaceRecognizerSF_FR_COSINE)
            if score > best_score:
                best_score, best_name = score, name
        if best_score >= RECOGNITION_THRESHOLD:
            return best_name, best_score
        self._unknown_count += 1
        return f'UNKNOWN_{self._unknown_count:03d}', best_score

    def _camera_cb(self, rgb_msg: Image, depth_msg: PointCloud2):
        now = self.get_clock().now().nanoseconds * 1e-9
        in_cooldown = (now - self._last_time) < self.get_parameter('cooldown').value
        min_size = self.get_parameter('min_face_size').value

        bgr = self.br.imgmsg_to_cv2(rgb_msg, 'bgr8')
        debug = bgr.copy()

        gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = _get_cascade().detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size))

        self._frame_count += 1
        log_this_frame = (self._frame_count % 30 == 0)  # ~3s at 10fps

        if len(faces) == 0:
            cv2.putText(debug, 'No face detected', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
            if log_this_frame:
                self.get_logger().info('Camera: no faces in frame')
        else:
            if log_this_frame:
                self.get_logger().info(f'Camera: {len(faces)} face(s) detected')

        for (x, y, fw, fh) in faces:
            face_crop = cv2.resize(bgr[y:y+fh, x:x+fw], FACE_SIZE)
            feature   = self._recogniser.feature(face_crop)  # shape (1, 128)
            person_id, score = self._identify(feature)
            known = not person_id.startswith('UNKNOWN')
            color = (0, 200, 0) if known else (0, 140, 255)

            cv2.rectangle(debug, (x, y), (x + fw, y + fh), color, 3)
            label = f'{person_id} ({score:.2f})'
            cv2.putText(debug, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            if log_this_frame:
                self.get_logger().info(
                    f'  → face at ({x},{y}) size {fw}x{fh} — '
                    f'{"MATCH" if known else "unknown"}: {person_id} score={score:.3f}')

            # Only publish victim detection if not in cooldown
            if not in_cooldown:
                cx, cy = x + fw // 2, y + fh // 2
                point_offset = cy * depth_msg.row_step + cx * depth_msg.point_step
                if point_offset + 12 > len(depth_msg.data):
                    continue
                X, Y, Z = struct.unpack_from('fff', depth_msg.data, offset=point_offset)
                if not all(math.isfinite(v) for v in (X, Y, Z)):
                    continue

                try:
                    tf = self.tf_buffer.lookup_transform(
                        'base_footprint', rgb_msg.header.frame_id,
                        rclpy.time.Time(), rclpy.duration.Duration(seconds=0.3))
                    R = _q2R([tf.transform.rotation.w, tf.transform.rotation.x,
                               tf.transform.rotation.y, tf.transform.rotation.z])
                    t = np.array([tf.transform.translation.x,
                                   tf.transform.translation.y,
                                   tf.transform.translation.z])
                    pt = R @ np.array([X, Y, Z]) + t
                except TransformException as e:
                    self.get_logger().warn(f'TF error: {e}')
                    continue

                self._last_time = now

                pose = PoseStamped()
                pose.header.stamp    = rgb_msg.header.stamp
                pose.header.frame_id = 'base_footprint'
                pose.pose.position.x = float(pt[0])
                pose.pose.position.y = float(pt[1])
                pose.pose.position.z = float(pt[2])
                self.pub_pose.publish(pose)

                id_msg = String(); id_msg.data = person_id
                self.pub_id.publish(id_msg)

                victim_img = self.br.cv2_to_imgmsg(debug, encoding='bgr8')
                victim_img.header = rgb_msg.header
                self.pub_image.publish(victim_img)

                self.get_logger().info(
                    f'VICTIM LOGGED: [{person_id}] score={score:.3f} '
                    f'depth=({X:.2f},{Y:.2f},{Z:.2f})m '
                    f'map=({pt[0]:.2f},{pt[1]:.2f},{pt[2]:.2f})m')
                break  # one victim per trigger

        # Always publish debug image
        if self.pub_debug.get_subscription_count() > 0:
            debug_msg = self.br.cv2_to_imgmsg(debug, encoding='bgr8')
            debug_msg.header = rgb_msg.header
            self.pub_debug.publish(debug_msg)


def main(args=None):
    rclpy.init(args=args)
    node = FaceVictimDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
