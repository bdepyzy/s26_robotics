import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge
import csv
import os
import datetime
import cv2


# Latched-equivalent QoS so RViz gets markers even if it connects late
_LATCHED = QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)


class VictimLogger(Node):
    """
    Listens to victim detections and:
      • Writes a CSV log  — logs/<session>/victims.csv
      • Saves annotated JPEG snapshots — logs/<session>/<victim_id>.jpg
      • Publishes /victim_markers (MarkerArray, transient-local) so RViz
        shows a red sphere + text label at each victim's map position
    """

    def __init__(self):
        super().__init__('victim_logger')

        self.declare_parameter('log_dir', '/root/codes/S26_roboticsII_ws/logs')

        log_base = self.get_parameter('log_dir').value
        session  = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self._log_dir = os.path.join(log_base, session)
        os.makedirs(self._log_dir, exist_ok=True)

        self._csv_path = os.path.join(self._log_dir, 'victims.csv')
        with open(self._csv_path, 'w', newline='') as f:
            csv.writer(f).writerow(['victim_id', 'timestamp', 'pos_x_m', 'pos_y_m', 'pos_z_m'])

        self.br = CvBridge()
        self._marker_array = MarkerArray()
        self._pending_id   = None
        self._pending_pose = None

        self.pub_markers = self.create_publisher(MarkerArray, '/victim_markers', _LATCHED)

        self.sub_id    = self.create_subscription(String,      '/victim_found_id',    self._id_cb,    10)
        self.sub_pose  = self.create_subscription(PoseStamped, '/victim_found_pose',  self._pose_cb,  10)
        self.sub_image = self.create_subscription(Image,       '/victim_found_image', self._image_cb, 10)

        self.get_logger().info(f'VictimLogger writing to {self._log_dir}')

    # ------------------------------------------------------------------

    def _id_cb(self, msg: String):
        self._pending_id = msg.data

    def _pose_cb(self, msg: PoseStamped):
        self._pending_pose = msg
        if self._pending_id:
            self._write_csv(self._pending_id, msg)
            self._publish_marker(self._pending_id, msg)

    def _image_cb(self, msg: Image):
        if not self._pending_id:
            return
        try:
            bgr = self.br.imgmsg_to_cv2(msg, 'bgr8')
            cv2.imwrite(os.path.join(self._log_dir, f'{self._pending_id}.jpg'), bgr)
        except Exception as e:
            self.get_logger().warn(f'Could not save image for {self._pending_id}: {e}')

    # ------------------------------------------------------------------

    def _write_csv(self, victim_id: str, pose: PoseStamped):
        ts = datetime.datetime.now().isoformat(timespec='seconds')
        x, y, z = pose.pose.position.x, pose.pose.position.y, pose.pose.position.z
        with open(self._csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([victim_id, ts, f'{x:.3f}', f'{y:.3f}', f'{z:.3f}'])
        self.get_logger().info(
            f'\n'
            f'  ╔══════════════════════════════════╗\n'
            f'  ║  VICTIM FOUND: {victim_id:<18}║\n'
            f'  ║  Time : {ts:<26}║\n'
            f'  ║  Pos  : ({x:+.2f}, {y:+.2f}, {z:+.2f}) m    ║\n'
            f'  ╚══════════════════════════════════╝'
        )

    def _publish_marker(self, victim_id: str, pose: PoseStamped):
        marker_id = len(self._marker_array.markers) // 2  # sphere + text = 2 per victim

        # Red sphere at victim location
        sphere = Marker()
        sphere.header.frame_id = 'map'
        sphere.header.stamp    = self.get_clock().now().to_msg()
        sphere.ns     = 'victims'
        sphere.id     = marker_id * 2
        sphere.type   = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position.x = pose.pose.position.x
        sphere.pose.position.y = pose.pose.position.y
        sphere.pose.position.z = 0.2
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.3
        sphere.color.r = 1.0
        sphere.color.g = 0.1
        sphere.color.b = 0.1
        sphere.color.a = 0.9

        # White text label floating above
        label = Marker()
        label.header = sphere.header
        label.ns     = 'victims'
        label.id     = marker_id * 2 + 1
        label.type   = Marker.TEXT_VIEW_FACING
        label.action = Marker.ADD
        label.pose.position.x = pose.pose.position.x
        label.pose.position.y = pose.pose.position.y
        label.pose.position.z = 0.55
        label.pose.orientation.w = 1.0
        label.scale.z = 0.2
        label.color.r = label.color.g = label.color.b = label.color.a = 1.0
        label.text = victim_id

        self._marker_array.markers.append(sphere)
        self._marker_array.markers.append(label)
        self.pub_markers.publish(self._marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = VictimLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
