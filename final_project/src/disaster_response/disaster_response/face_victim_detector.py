import json
import math
import os
import struct

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

HAAR = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
MODEL_DIR = '/root/codes/models'
SFACE_MODEL = os.path.join(MODEL_DIR, 'face_recognition_sface_2021dec.onnx')
ENROLL_FILE = os.path.join(MODEL_DIR, 'enrolled_faces.json')
FACE_SIZE = (112, 112)
RECOGNITION_THRESHOLD = 0.65

STATUS_NO_FACE = 'NO FACE'
STATUS_UNKNOWN = 'UNKNOWN'
STATUS_KNOWN = 'KNOWN'
STATUS_NO_DEPTH = 'NO DEPTH'
STATUS_TF_ERROR = 'TF ERROR'
STATUS_COOLDOWN = 'COOLDOWN'
STATUS_LOW_SCORE = 'LOW SCORE'

_CASCADE = None


def _get_cascade():
    global _CASCADE
    if _CASCADE is None:
        _CASCADE = cv2.CascadeClassifier(HAAR)
    return _CASCADE


def _q2R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


class FaceVictimDetector(Node):

    def __init__(self):
        super().__init__('face_victim_detector')

        self.declare_parameter('cooldown', 5.0)
        self.declare_parameter('min_face_size', 60)
        self.declare_parameter('startup_delay', 30.0)
        self.declare_parameter('min_victim_score', 0.15)
        self.declare_parameter('recognition_threshold', RECOGNITION_THRESHOLD)
        self.declare_parameter('debug_log_period_sec', 3.0)

        self.br = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._last_time = 0.0
        self._last_debug_log_time = 0.0
        self._unknown_count = 0
        self._frame_count = 0
        self._start_time = self.get_clock().now().nanoseconds * 1e-9

        _get_cascade()

        if not os.path.exists(SFACE_MODEL):
            raise RuntimeError(f'Missing SFace model: {SFACE_MODEL}')
        self._recogniser = cv2.FaceRecognizerSF.create(SFACE_MODEL, '')

        self._enrolled = {}
        self._load_enrolled()

        self.pub_pose = self.create_publisher(PoseStamped, '/victim_found_pose', 10)
        self.pub_image = self.create_publisher(Image, '/victim_found_image', 10)
        self.pub_id = self.create_publisher(String, '/victim_found_id', 10)
        self.pub_debug = self.create_publisher(Image, '/face_detector/debug_image', 10)

        self.sub_rgb = Subscriber(self, Image, '/camera/color/image_raw')
        self.sub_depth = Subscriber(self, PointCloud2, '/camera/depth_registered/points')
        self.ts = ApproximateTimeSynchronizer([self.sub_rgb, self.sub_depth], 10, 0.15)
        self.ts.registerCallback(self._camera_cb)

        enrolled_names = list(self._enrolled.keys()) or ['none - run enroll_faces first']
        self.get_logger().info(f'FaceVictimDetector ready | enrolled: {enrolled_names}')
        self.get_logger().info('Debug image -> /face_detector/debug_image')
        self.get_logger().info(
            f'Recognition threshold: {self.get_parameter("recognition_threshold").value:.2f}')

    def _load_enrolled(self):
        if not os.path.exists(ENROLL_FILE):
            self.get_logger().warn('No enrolled faces - all detections will be UNKNOWN')
            return
        with open(ENROLL_FILE) as f:
            data = json.load(f)
        for name, feat_list in data.items():
            self._enrolled[name] = np.array(feat_list, dtype=np.float32).reshape(1, -1)
        self.get_logger().info(
            f'Loaded {len(self._enrolled)} enrolled face(s): {list(self._enrolled.keys())}')

    def _should_log_debug(self, now: float) -> bool:
        period = float(self.get_parameter('debug_log_period_sec').value)
        if period <= 0.0:
            self._last_debug_log_time = now
            return True
        if (now - self._last_debug_log_time) >= period:
            self._last_debug_log_time = now
            return True
        return False

    def _identify(self, feature: np.ndarray):
        best_name, best_score = None, -1.0
        for name, enrolled_feat in self._enrolled.items():
            score = self._recogniser.match(
                feature, enrolled_feat, cv2.FaceRecognizerSF_FR_COSINE)
            if score > best_score:
                best_score, best_name = score, name

        threshold = float(self.get_parameter('recognition_threshold').value)
        if best_name is not None and best_score >= threshold:
            return best_name, best_score, True

        self._unknown_count += 1
        return f'UNKNOWN_{self._unknown_count:03d}', best_score, False

    def _lookup_depth_point(self, depth_msg: PointCloud2, cx: int, cy: int):
        point_offset = cy * depth_msg.row_step + cx * depth_msg.point_step
        if point_offset + 12 > len(depth_msg.data):
            return None
        point = struct.unpack_from('fff', depth_msg.data, offset=point_offset)
        if not all(math.isfinite(v) for v in point):
            return None
        return point

    def _transform_point(self, rgb_msg: Image, point):
        tf = self.tf_buffer.lookup_transform(
            'base_footprint',
            rgb_msg.header.frame_id,
            rclpy.time.Time(),
            rclpy.duration.Duration(seconds=0.3),
        )
        R = _q2R([
            tf.transform.rotation.w,
            tf.transform.rotation.x,
            tf.transform.rotation.y,
            tf.transform.rotation.z,
        ])
        t = np.array([
            tf.transform.translation.x,
            tf.transform.translation.y,
            tf.transform.translation.z,
        ])
        return R @ np.array(point) + t

    def _add_banner(self, image: np.ndarray, text: str, color):
        cv2.rectangle(image, (0, 0), (image.shape[1], 40), (20, 20, 20), -1)
        cv2.putText(image, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def _camera_cb(self, rgb_msg: Image, depth_msg: PointCloud2):
        now = self.get_clock().now().nanoseconds * 1e-9
        if (now - self._start_time) < float(self.get_parameter('startup_delay').value):
            return

        in_cooldown = (now - self._last_time) < float(self.get_parameter('cooldown').value)
        min_size = int(self.get_parameter('min_face_size').value)
        min_score = float(self.get_parameter('min_victim_score').value)
        log_this_frame = self._should_log_debug(now)

        bgr = self.br.imgmsg_to_cv2(rgb_msg, 'bgr8')
        debug = bgr.copy()
        banner_text = STATUS_NO_FACE
        banner_color = (100, 100, 255)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = _get_cascade().detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(min_size, min_size),
        )

        self._frame_count += 1
        if log_this_frame:
            self.get_logger().info(
                f'RGB/depth sync alive: frame={self._frame_count} faces={len(faces)} '
                f'cooldown={in_cooldown}')

        detections = []
        recognized_seen = False
        accepted_detection = None

        if len(faces) == 0:
            if log_this_frame:
                self.get_logger().info('Face detection: no faces detected in RGB frame')

        for (x, y, fw, fh) in faces:
            face_crop = cv2.resize(bgr[y:y + fh, x:x + fw], FACE_SIZE)
            feature = self._recogniser.feature(face_crop)
            person_id, score, known = self._identify(feature)
            status = STATUS_KNOWN if known else STATUS_UNKNOWN
            color = (0, 200, 0) if known else (0, 140, 255)

            if log_this_frame:
                self.get_logger().info(
                    f'Embedding scored: id={person_id} score={score:.3f} '
                    f'known={known} box=({x},{y},{fw},{fh})')

            point = None
            transformed_point = None
            if score < min_score:
                status = STATUS_LOW_SCORE
                if log_this_frame:
                    self.get_logger().info(
                        f'Detection rejected: score {score:.3f} below minimum face score '
                        f'{min_score:.2f}')
            elif not known:
                threshold = float(self.get_parameter('recognition_threshold').value)
                if log_this_frame:
                    self.get_logger().info(
                        f'Detection rejected: score {score:.3f} below recognition threshold '
                        f'{threshold:.2f}')
            else:
                recognized_seen = True
                if in_cooldown:
                    status = STATUS_COOLDOWN
                else:
                    cx, cy = x + fw // 2, y + fh // 2
                    point = self._lookup_depth_point(depth_msg, cx, cy)
                    if point is None:
                        status = STATUS_NO_DEPTH
                        if log_this_frame:
                            self.get_logger().info(
                                f'Detection rejected: depth lookup failed for {person_id} '
                                f'at pixel=({cx},{cy})')
                    else:
                        try:
                            transformed_point = self._transform_point(rgb_msg, point)
                        except TransformException as exc:
                            status = STATUS_TF_ERROR
                            if log_this_frame:
                                self.get_logger().warn(
                                    f'Detection rejected: TF lookup failed for {person_id}: {exc}')
                        else:
                            if accepted_detection is not None:
                                status = STATUS_COOLDOWN
                                transformed_point = None
                                point = None
                                detections.append({
                                    'box': (x, y, fw, fh),
                                    'person_id': person_id,
                                    'score': score,
                                    'known': known,
                                    'status': status,
                                    'color': color,
                                })
                                continue
                            accepted_detection = {
                                'person_id': person_id,
                                'score': score,
                                'point': point,
                                'transformed_point': transformed_point,
                                'stamp': rgb_msg.header.stamp,
                            }

            detections.append({
                'box': (x, y, fw, fh),
                'person_id': person_id,
                'score': score,
                'known': known,
                'status': status,
                'color': color,
            })

        if len(faces) > 0:
            if accepted_detection is not None:
                banner_text = STATUS_KNOWN
                banner_color = (0, 200, 0)
            elif recognized_seen:
                if any(d['status'] == STATUS_TF_ERROR for d in detections):
                    banner_text = STATUS_TF_ERROR
                    banner_color = (0, 100, 255)
                elif any(d['status'] == STATUS_NO_DEPTH for d in detections):
                    banner_text = STATUS_NO_DEPTH
                    banner_color = (0, 165, 255)
                elif in_cooldown:
                    banner_text = STATUS_COOLDOWN
                    banner_color = (255, 215, 0)
                else:
                    banner_text = STATUS_KNOWN
                    banner_color = (0, 200, 0)
            else:
                banner_text = STATUS_UNKNOWN
                banner_color = (0, 140, 255)
                if log_this_frame:
                    self.get_logger().info(
                        'Faces seen but all detections are UNKNOWN or below threshold')

        for detection in detections:
            x, y, fw, fh = detection['box']
            color = detection['color']
            cv2.rectangle(debug, (x, y), (x + fw, y + fh), color, 3)
            label = f"{detection['person_id']} ({detection['score']:.2f})"
            cv2.putText(debug, label, (x, max(24, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(debug, detection['status'], (x, y + fh + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        self._add_banner(debug, banner_text, banner_color)

        if accepted_detection is not None:
            self._last_time = now

            pose = PoseStamped()
            pose.header.stamp = accepted_detection['stamp']
            pose.header.frame_id = 'base_footprint'
            pose.pose.position.x = float(accepted_detection['transformed_point'][0])
            pose.pose.position.y = float(accepted_detection['transformed_point'][1])
            pose.pose.position.z = float(accepted_detection['transformed_point'][2])
            self.pub_pose.publish(pose)

            id_msg = String()
            id_msg.data = accepted_detection['person_id']
            self.pub_id.publish(id_msg)

            victim_img = self.br.cv2_to_imgmsg(debug, encoding='bgr8')
            victim_img.header = rgb_msg.header
            self.pub_image.publish(victim_img)

            point = accepted_detection['point']
            transformed_point = accepted_detection['transformed_point']
            self.get_logger().info(
                f'Detection accepted and published: [{accepted_detection["person_id"]}] '
                f'score={accepted_detection["score"]:.3f} '
                f'depth=({point[0]:.2f},{point[1]:.2f},{point[2]:.2f})m '
                f'map=({transformed_point[0]:.2f},{transformed_point[1]:.2f},'
                f'{transformed_point[2]:.2f})m')

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
