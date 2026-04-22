import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
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


def _q2R(q):
    """Quaternion [w,x,y,z] → 3×3 rotation matrix (Euler-Rodrigues)."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


class VictimDetector(Node):
    """
    Detects disaster victims by colour marker (default: orange) in the RGB image.
    Uses the depth point cloud to localise each victim in the base_footprint frame.
    Publishes:
      /victim_found_pose  — geometry_msgs/PoseStamped  (3-D position in base_footprint)
      /victim_found_image — sensor_msgs/Image           (annotated snapshot)
      /victim_found_id    — std_msgs/String             (victim label)
    """

    # Victim counter so each detection gets a unique ID this session
    _victim_count = 0

    def __init__(self):
        super().__init__('victim_detector')

        # HSV bounds for orange marker — tune via ROS params
        self.declare_parameter('hsv_low',  [5,  120, 120])
        self.declare_parameter('hsv_high', [20, 255, 255])
        self.declare_parameter('min_area', 800)        # px²: ignore tiny blobs
        self.declare_parameter('cooldown', 4.0)        # seconds between repeat detections

        self.br = CvBridge()
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self._last_detection_time = 0.0

        # Publishers
        self.pub_pose  = self.create_publisher(PoseStamped, '/victim_found_pose',  10)
        self.pub_image = self.create_publisher(Image,        '/victim_found_image', 10)
        self.pub_id    = self.create_publisher(String,       '/victim_found_id',    10)

        # Synchronised RGB + depth subscribers
        self.sub_rgb   = Subscriber(self, Image,       '/camera/color/image_raw')
        self.sub_depth = Subscriber(self, PointCloud2, '/camera/depth/points')
        self.ts = ApproximateTimeSynchronizer([self.sub_rgb, self.sub_depth], 10, 0.15)
        self.ts.registerCallback(self._camera_cb)

        self.get_logger().info('VictimDetector ready — watching for colour markers')

    # ------------------------------------------------------------------

    def _camera_cb(self, rgb_msg: Image, depth_msg: PointCloud2):
        now = self.get_clock().now().nanoseconds * 1e-9
        cooldown = self.get_parameter('cooldown').value
        if now - self._last_detection_time < cooldown:
            return

        hsv_low  = np.array(self.get_parameter('hsv_low').value,  dtype=np.uint8)
        hsv_high = np.array(self.get_parameter('hsv_high').value, dtype=np.uint8)
        min_area = self.get_parameter('min_area').value

        bgr = self.br.imgmsg_to_cv2(rgb_msg, 'bgr8')
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, hsv_low, hsv_high)
        # Morphological clean-up to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < min_area:
            return

        x, y, w, h = cv2.boundingRect(largest)
        cx, cy = x + w // 2, y + h // 2

        # Depth lookup at contour centre
        point_offset = cy * depth_msg.row_step + cx * depth_msg.point_step
        if point_offset + 12 > len(depth_msg.data):
            return
        X, Y, Z = struct.unpack_from('fff', depth_msg.data, offset=point_offset)
        if not all(math.isfinite(v) for v in (X, Y, Z)):
            return

        # Transform from camera frame → base_footprint
        try:
            tf = self.tf_buffer.lookup_transform(
                'base_footprint', rgb_msg.header.frame_id,
                rclpy.time.Time(), rclpy.duration.Duration(seconds=0.3))
            R = _q2R([tf.transform.rotation.w,
                      tf.transform.rotation.x,
                      tf.transform.rotation.y,
                      tf.transform.rotation.z])
            t = np.array([tf.transform.translation.x,
                          tf.transform.translation.y,
                          tf.transform.translation.z])
            pt_robot = R @ np.array([X, Y, Z]) + t
        except TransformException as e:
            self.get_logger().warn(f'TF error: {e}')
            return

        VictimDetector._victim_count += 1
        victim_id = f'VICTIM_{VictimDetector._victim_count:03d}'
        self._last_detection_time = now

        # Annotate image
        cv2.rectangle(bgr, (x, y), (x + w, y + h), (0, 69, 255), 3)
        cv2.putText(bgr, victim_id, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 69, 255), 2)

        # Publish pose
        pose = PoseStamped()
        pose.header.stamp    = rgb_msg.header.stamp
        pose.header.frame_id = 'base_footprint'
        pose.pose.position.x = float(pt_robot[0])
        pose.pose.position.y = float(pt_robot[1])
        pose.pose.position.z = float(pt_robot[2])
        self.pub_pose.publish(pose)

        # Publish annotated image
        img_msg = self.br.cv2_to_imgmsg(bgr, encoding='bgr8')
        img_msg.header = rgb_msg.header
        self.pub_image.publish(img_msg)

        # Publish ID string
        id_msg = String()
        id_msg.data = victim_id
        self.pub_id.publish(id_msg)

        self.get_logger().info(
            f'[{victim_id}] detected at robot frame '
            f'({pt_robot[0]:.2f}, {pt_robot[1]:.2f}, {pt_robot[2]:.2f}) m')


def main(args=None):
    rclpy.init(args=args)
    node = VictimDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
