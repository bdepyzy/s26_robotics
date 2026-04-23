import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String
import numpy as np
import math
import signal
import time

RAD2DEG = 180.0 / math.pi
FRONT_MIN_ABS = 160.0
SIDE_ANGLE    = 50.0


def _yaw_from_quat(q):
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny, cosy)


def _angle_diff(a, b):
    d = a - b
    while d >  math.pi: d -= 2 * math.pi
    while d < -math.pi: d += 2 * math.pi
    return d


class LidarExplorer(Node):
    """
    Lawnmower search → APPROACH victim → CIRCLE around victim → RETURN to start.
    """

    def __init__(self):
        super().__init__('lidar_explorer')

        # Grid — 7 rows × 9 cols × 0.5715 m = 4 m wide × 5.14 m forward
        # (arena -2x..+2x, robot at y=-2, top at y=+3 → 5 m forward)
        self.declare_parameter('cell_size',          0.5715)
        self.declare_parameter('grid_cols',          5)
        self.declare_parameter('grid_rows',          5)
        self.declare_parameter('linear_speed',       0.3)
        self.declare_parameter('angular_speed',      0.8)
        self.declare_parameter('stop_dist',          0.5)
        self.declare_parameter('slow_dist',          0.9)
        self.declare_parameter('front_warn_thresh',  8)
        self.declare_parameter('waypoint_tol',       0.30)
        self.declare_parameter('angle_tol',          0.22)
        self.declare_parameter('waypoint_timeout',   15.0)
        self.declare_parameter('startup_delay',      30.0)
        self.declare_parameter('approach_dist',      0.55)
        self.declare_parameter('circle_linear',      0.20)
        self.declare_parameter('circle_angular',     0.40)

        self.joy_active    = False
        self.victim_found  = False
        self._pose         = None
        self._start_pose   = None
        self._victim_odom  = None   # victim position in odom frame
        self._waypoints    = []
        self._wp_idx       = 0
        self._state        = 'WAIT_ODOM'
        self._last_log     = None
        self._scan         = None
        self._wp_start_t   = None
        self._start_time   = self.get_clock().now().nanoseconds * 1e-9
        self._countdown_last = -1
        self._circle_yaw_accum = 0.0
        self._last_circle_yaw  = None

        self.pub_vel = self.create_publisher(Twist, '/cmd_vel', 1)
        self.create_subscription(LaserScan,    '/scan',              self._scan_cb,        1)
        self.create_subscription(Odometry,     '/odom',              self._odom_cb,        1)
        self.create_subscription(Bool,         '/JoyState',          self._joy_cb,         1)
        self.create_subscription(String,       '/victim_found_id',   self._victim_id_cb,   10)
        self.create_subscription(PoseStamped,  '/victim_found_pose', self._victim_pose_cb, 10)

        self.create_timer(0.1, self._control_loop)
        self.get_logger().info('LidarExplorer ready — waiting for odometry')

    # ── subscribers ───────────────────────────────────────────────────────────

    def _joy_cb(self, msg):
        self.joy_active = msg.data
        if self.joy_active:
            self.pub_vel.publish(Twist())

    def _odom_cb(self, msg):
        p = msg.pose.pose.position
        yaw = _yaw_from_quat(msg.pose.pose.orientation)
        self._pose = (p.x, p.y, yaw)
        if self._state == 'WAIT_ODOM':
            self._state = 'STARTUP'

    def _scan_cb(self, msg):
        self._scan = msg

    def _victim_pose_cb(self, msg: PoseStamped):
        """Convert victim pose from base_footprint to odom frame and store it."""
        if self._victim_odom is not None or self._pose is None:
            return
        px, py, pyaw = self._pose
        # msg is in base_footprint: x=forward, y=left
        bfx = msg.pose.position.x
        bfy = msg.pose.position.y
        vx = px + math.cos(pyaw) * bfx - math.sin(pyaw) * bfy
        vy = py + math.sin(pyaw) * bfx + math.cos(pyaw) * bfy
        self._victim_odom = (vx, vy)
        self.get_logger().info(
            f'Victim odom position stored: ({vx:.2f}, {vy:.2f})')

    def _victim_id_cb(self, msg: String):
        victim_id = msg.data
        if victim_id.startswith('UNKNOWN_'):
            self.get_logger().info(f'Ignored victim trigger [{victim_id}]')
            return
        if not self.victim_found:
            self.victim_found = True
            self.get_logger().info(f'Accepted victim trigger [{victim_id}] — switching to APPROACH')
            self._state = 'APPROACH'

    # ── waypoint generation ───────────────────────────────────────────────────

    def _build_waypoints(self):
        cell = self.get_parameter('cell_size').value
        cols = self.get_parameter('grid_cols').value
        rows = self.get_parameter('grid_rows').value
        sx, sy, syaw = self._start_pose

        fwd = np.array([math.cos(syaw), math.sin(syaw)])
        rgt = np.array([math.cos(syaw - math.pi / 2), math.sin(syaw - math.pi / 2)])

        wps = []
        for c in range(cols):
            f = (c + 0.5) * cell
            row_order = range(rows) if c % 2 == 0 else range(rows - 1, -1, -1)
            for r in row_order:
                lateral = (r - (rows - 1) / 2.0) * cell
                pt = np.array([sx, sy]) + fwd * f + rgt * lateral
                wps.append((float(pt[0]), float(pt[1])))

        wps.append((sx, sy))
        self._waypoints = wps
        self._wp_idx    = 0

    # ── obstacle sensing ──────────────────────────────────────────────────────

    def _obstacle_info(self):
        if self._scan is None:
            return False, 0, 0, float('inf')

        stop_d = self.get_parameter('stop_dist').value
        thresh = self.get_parameter('front_warn_thresh').value
        scan   = self._scan

        ranges = np.array(scan.ranges)
        angles = (scan.angle_min + scan.angle_increment * np.arange(len(ranges))) * RAD2DEG

        front_mask = np.abs(angles) > FRONT_MIN_ABS
        left_mask  = (angles >  (180.0 - FRONT_MIN_ABS + 10)) & (angles <  (180.0 - SIDE_ANGLE))
        right_mask = (angles < -(180.0 - FRONT_MIN_ABS + 10)) & (angles > -(180.0 - SIDE_ANGLE))

        def n_close(mask, d):
            v = ranges[mask]; v = v[np.isfinite(v)]
            return int(np.sum(v < d))

        fv = ranges[front_mask]; fv = fv[np.isfinite(fv)]
        min_front = float(fv.min()) if len(fv) else float('inf')

        return (n_close(front_mask, stop_d) > thresh,
                n_close(left_mask,  stop_d),
                n_close(right_mask, stop_d),
                min_front)

    # ── control loop (10 Hz) ─────────────────────────────────────────────────

    def _control_loop(self):
        if self.joy_active or self._pose is None:
            return
        if self._state in ('WAIT_ODOM', 'DONE'):
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        startup_delay = self.get_parameter('startup_delay').value
        remaining = startup_delay - (now - self._start_time)

        if remaining > 0:
            tick = int(remaining)
            if tick != self._countdown_last:
                if tick <= 10:
                    self.get_logger().info(f'Starting in {tick}s — place robot on ground now...')
                elif tick % 5 == 0:
                    self.get_logger().info(f'Startup hold: {tick}s remaining — enroll faces now')
                self._countdown_last = tick
            self.pub_vel.publish(Twist())
            return

        if self._start_pose is None:
            self._start_pose = self._pose
            self._build_waypoints()
            self._state = 'SEARCH'
            sx, sy, syaw = self._start_pose
            self.get_logger().info(
                f'Start: ({sx:.2f}, {sy:.2f}) yaw={math.degrees(syaw):.1f}° — '
                f'{len(self._waypoints)} waypoints')

        px, py, pyaw = self._pose
        linear  = self.get_parameter('linear_speed').value
        angular = self.get_parameter('angular_speed').value
        ang_tol = self.get_parameter('angle_tol').value
        cmd = Twist()

        # ── APPROACH: steer toward known victim odom position ─────────────
        if self._state == 'APPROACH':
            approach_dist = self.get_parameter('approach_dist').value

            if self._victim_odom is None:
                # Pose not received yet — drive slowly forward
                cmd.linear.x = linear * 0.4
                log_key = 'APPROACH waiting for pose'
            else:
                tx, ty = self._victim_odom
                dx, dy = tx - px, ty - py
                dist   = math.hypot(dx, dy)

                if dist < approach_dist:
                    self.get_logger().info(
                        f'Victim reached at {dist:.2f}m — starting circle')
                    self._state = 'CIRCLE'
                    self._last_circle_yaw  = pyaw
                    self._circle_yaw_accum = 0.0
                    self.pub_vel.publish(Twist())
                    return

                target_yaw = math.atan2(dy, dx)
                yaw_err    = _angle_diff(target_yaw, pyaw)

                if abs(yaw_err) > ang_tol:
                    spd = angular * min(1.0, abs(yaw_err) / math.radians(45))
                    cmd.angular.z = math.copysign(max(spd, 0.10), yaw_err)
                else:
                    cmd.linear.x  = linear * 0.5
                    cmd.angular.z = angular * 0.8 * (yaw_err / math.pi)

                log_key = f'APPROACH dist={dist:.2f}m'

            tag = f'[APPROACH] {log_key}'
            if tag != self._last_log:
                self.get_logger().info(tag)
                self._last_log = tag
            self.pub_vel.publish(cmd)
            return

        # ── CIRCLE: orbit for one full revolution ─────────────────────────
        if self._state == 'CIRCLE':
            front_blocked, _, _, _ = self._obstacle_info()
            if front_blocked:
                self.get_logger().warn('Obstacle during circle — returning to start')
                self._waypoints = [self._start_pose[:2]]
                self._wp_idx    = 0
                self._state     = 'RETURN'
                self.pub_vel.publish(Twist())
                return

            delta = _angle_diff(pyaw, self._last_circle_yaw)
            self._circle_yaw_accum += delta
            self._last_circle_yaw = pyaw

            if self._circle_yaw_accum >= 2 * math.pi:
                self.get_logger().info('Circle complete — returning to start')
                self._waypoints = [self._start_pose[:2]]
                self._wp_idx    = 0
                self._state     = 'RETURN'
                self.pub_vel.publish(Twist())
            else:
                cmd.linear.x  = self.get_parameter('circle_linear').value
                cmd.angular.z = self.get_parameter('circle_angular').value
                deg = math.degrees(self._circle_yaw_accum)
                log_key = f'CIRCLE {deg:.0f}/360°'
                if log_key != self._last_log:
                    self.get_logger().info(f'[CIRCLE] {log_key}')
                    self._last_log = log_key
                self.pub_vel.publish(cmd)
            return

        # ── SEARCH / RETURN: waypoint navigation ─────────────────────────
        wp_tol     = self.get_parameter('waypoint_tol').value
        wp_timeout = self.get_parameter('waypoint_timeout').value
        front_blocked, left_d, right_d, min_front = self._obstacle_info()

        if self._wp_idx >= len(self._waypoints):
            self._state = 'DONE'
            self.pub_vel.publish(Twist())
            self.get_logger().info('DONE — stopped at start position.')
            return

        if self._wp_start_t is None:
            self._wp_start_t = now

        tx, ty = self._waypoints[self._wp_idx]
        dx, dy = tx - px, ty - py
        dist   = math.hypot(dx, dy)

        if (now - self._wp_start_t) > wp_timeout:
            self.get_logger().warn(
                f'[{self._state}] waypoint {self._wp_idx+1} timed out — skipping')
            self._wp_idx += 1
            self._wp_start_t = None
            self.pub_vel.publish(Twist())
            return

        if dist < wp_tol:
            self._wp_idx += 1
            self._wp_start_t = None
            self.get_logger().info(
                f'[{self._state}] waypoint {self._wp_idx}/{len(self._waypoints)} reached')
            self.pub_vel.publish(Twist())
            return

        target_yaw = math.atan2(dy, dx)
        yaw_err    = _angle_diff(target_yaw, pyaw)

        if front_blocked:
            turn = -angular if right_d <= left_d else angular
            cmd.angular.z = turn
            log_key = 'AVOID'
        elif abs(yaw_err) > ang_tol:
            spd = angular * min(1.0, abs(yaw_err) / math.radians(45))
            cmd.angular.z = math.copysign(max(spd, 0.10), yaw_err)
            log_key = f'TURN  yaw_err={math.degrees(yaw_err):+.1f}°'
        else:
            cmd.linear.x  = linear
            cmd.angular.z = angular * 0.8 * (yaw_err / math.pi)
            log_key = f'DRIVE dist={dist:.2f}m front={min_front:.2f}m'

        tag = f'[{self._state} wp{self._wp_idx+1}/{len(self._waypoints)}] {log_key}'
        if tag != self._last_log:
            self.get_logger().info(tag)
            self._last_log = tag
        self.pub_vel.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = LidarExplorer()

    def _send_stop(n):
        stop = Twist()
        for _ in range(5):
            try:
                n.pub_vel.publish(stop)
            except Exception:
                break
            time.sleep(0.05)

    def _sigterm(signum, frame):
        _send_stop(node)
        rclpy.shutdown()

    signal.signal(signal.SIGTERM, _sigterm)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        _send_stop(node)
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
