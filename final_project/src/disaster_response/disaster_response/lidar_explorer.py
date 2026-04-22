import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
import numpy as np
import math

RAD2DEG = 180.0 / math.pi

# Angles are measured from the back of the robot (180°/−180° = front for RPLiDAR A1 mounted rear-facing)
# The yahboomcar convention: |angle| > 160° = front arc
FRONT_MIN_ABS = 160.0   # degrees: front sector threshold
SIDE_ANGLE   = 50.0     # degrees: side sector half-width


class LidarExplorer(Node):
    """
    Reactive LiDAR-based explorer for disaster scenes.

    Drives forward until obstacles are sensed, then turns to find clear space.
    Publishes /cmd_vel. Stops when /JoyState is True (joystick override).
    """

    def __init__(self):
        super().__init__('lidar_explorer')

        self.declare_parameter('linear_speed', 0.3)
        self.declare_parameter('angular_speed', 0.8)
        self.declare_parameter('stop_dist', 0.5)      # metres: danger zone
        self.declare_parameter('slow_dist', 0.9)      # metres: slow-down zone
        self.declare_parameter('front_warn_thresh', 8) # scan points to trigger warning

        self.joy_active  = False
        self._scan_count = 0
        self._last_state = None

        self.pub_vel = self.create_publisher(Twist, '/cmd_vel', 1)
        self.sub_scan = self.create_subscription(LaserScan, '/scan', self._scan_cb, 1)
        self.sub_joy  = self.create_subscription(Bool, '/JoyState', self._joy_cb, 1)

        self.get_logger().info('LidarExplorer ready — waiting for /scan')

    # ------------------------------------------------------------------

    def _joy_cb(self, msg: Bool):
        self.joy_active = msg.data
        if self.joy_active:
            self.pub_vel.publish(Twist())

    def _scan_cb(self, scan: LaserScan):
        if self.joy_active:
            return

        linear  = self.get_parameter('linear_speed').value
        angular = self.get_parameter('angular_speed').value
        stop_d  = self.get_parameter('stop_dist').value
        slow_d  = self.get_parameter('slow_dist').value
        thresh  = self.get_parameter('front_warn_thresh').value

        ranges = np.array(scan.ranges)
        angles = (scan.angle_min + scan.angle_increment * np.arange(len(ranges))) * RAD2DEG

        front_mask = np.abs(angles) > FRONT_MIN_ABS
        left_mask  = (angles > (180.0 - FRONT_MIN_ABS + 10)) & (angles < (180.0 - SIDE_ANGLE))
        right_mask = (angles < -(180.0 - FRONT_MIN_ABS + 10)) & (angles > -(180.0 - SIDE_ANGLE))

        front_ranges = ranges[front_mask]
        left_ranges  = ranges[left_mask]
        right_ranges = ranges[right_mask]

        # Filter out inf/nan
        def close_count(r, dist):
            valid = r[np.isfinite(r)]
            return int(np.sum(valid < dist))

        front_danger = close_count(front_ranges, stop_d)
        front_slow   = close_count(front_ranges, slow_d)
        left_danger  = close_count(left_ranges,  stop_d)
        right_danger = close_count(right_ranges, stop_d)

        cmd = Twist()

        front_valid = front_ranges[np.isfinite(front_ranges)]
        min_front_m = float(front_valid.min()) if len(front_valid) else float('inf')

        if front_danger > thresh:
            turn_dir = 'RIGHT' if right_danger <= left_danger else 'LEFT'
            cmd.angular.z = -angular if turn_dir == 'RIGHT' else angular
            state_key = f'BLOCKED-{turn_dir}'
            state_msg = f'BLOCKED — turning {turn_dir}  front={min_front_m:.2f}m  L={left_danger} R={right_danger} pts'
        elif front_slow > thresh:
            turn_dir = 'right' if right_danger <= left_danger else 'left'
            cmd.linear.x  = linear * 0.4
            cmd.angular.z = (-angular if turn_dir == 'right' else angular) * 0.4
            state_key = 'SLOWING'
            state_msg = f'SLOWING — front={min_front_m:.2f}m  nudging {turn_dir}'
        else:
            cmd.linear.x = linear
            dist_str = f'{min_front_m:.2f}m' if math.isfinite(min_front_m) else 'open'
            state_key = 'FORWARD'
            state_msg = f'FORWARD — nearest={dist_str}'

        # Log only on state-type transitions, not every count fluctuation
        if state_key != self._last_state:
            self.get_logger().info(f'[LiDAR] {state_msg}')
            self._last_state = state_key

        self.pub_vel.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = LidarExplorer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub_vel.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()
