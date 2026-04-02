import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from tf2_ros import TransformException, Buffer, TransformListener
import numpy as np
import math

## Functions for quaternion and rotation matrix conversion
## The code is adapted from the general_robotics_toolbox package
## Code reference: https://github.com/rpiRobotics/rpi_general_robotics_toolbox_py
def hat(k):
    khat=np.zeros((3,3))
    khat[0,1]=-k[2]
    khat[0,2]=k[1]
    khat[1,0]=k[2]
    khat[1,2]=-k[0]
    khat[2,0]=-k[1]
    khat[2,1]=k[0]
    return khat

def q2R(q):
    I = np.identity(3)
    qhat = hat(q[1:4])
    qhat2 = qhat.dot(qhat)
    return I + 2*q[0]*qhat + 2*qhat2

def euler_from_quaternion(q):
    w=q[0]; x=q[1]; y=q[2]; z=q[3]
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(2 * (w * y - z * x))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return [roll,pitch,yaw]

class TrackingNode(Node):
    # State machine constants
    STATE_IDLE = 0
    STATE_GO_TO_GOAL = 1
    STATE_RETURN_TO_START = 2
    STATE_ROTATE_TO_START_ORIENTATION = 3
    STATE_COMPLETED = 4
    
    def __init__(self):
        super().__init__('tracking_node')
        self.get_logger().info('Tracking Node Started')
        
        # State machine
        self.state = self.STATE_IDLE
        self.start_position = None  # Store initial position
        self.start_yaw = None  # Store initial orientation
        self.last_goal_dist = float('inf')
        self.goal_reached_counter = 0  # Confirm goal is reached
        
        # Current object pose (in robot frame)
        self.obs_pose = None
        self.goal_pose = None
        
        # ROS parameters
        self.declare_parameter('world_frame_id', 'odom')
        self.odom_id = self.get_parameter('world_frame_id').get_parameter_value().string_value

        # Create a transform listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Create publisher for the control command
        self.pub_control_cmd = self.create_publisher(Twist, '/track_cmd_vel', 10)
        # Create subscribers
        self.sub_detected_obs_pose = self.create_subscription(PoseStamped, 'detected_color_object_pose', self.detected_obs_pose_callback, 10)
        self.sub_detected_goal_pose = self.create_subscription(PoseStamped, 'detected_color_goal_pose', self.detected_goal_pose_callback, 10)

        # Create timer, running at 100Hz
        self.timer = self.create_timer(0.01, self.timer_update)
        
        # Potential field parameters
        self.GOAL_THRESHOLD = 0.5   # Stop when within 0.5m of goal (increased for braking)
        self.SLOW_DIST = 1.0        # Start slowing at 1.0m
        self.ATTRACTION_GAIN = 0.5  # Attractive force gain
        self.REPULSION_GAIN = 0.6   # Repulsive force gain (increased)
        self.REPULSION_DIST = 0.9   # Distance at which repulsion activates (increased)
        self.MAX_LINEAR_VEL = 0.3   # Max forward velocity
        self.MAX_ANGULAR_VEL = 0.8  # Max turning velocity
        self.ANGULAR_TOLERANCE = 0.1  # Radians for orientation matching
        
        self.get_logger().info('Tracking Node initialized in IDLE state')
    
    def detected_obs_pose_callback(self, msg):
        center_points = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        
        # Filter: ignore if too far or too high
        if np.linalg.norm(center_points) > 3 or center_points[2] > 0.7:
            self.obs_pose = None
            return
        
        try:
            transform = self.tf_buffer.lookup_transform(self.odom_id, msg.header.frame_id, rclpy.time.Time(), rclpy.duration.Duration(seconds=0.1))
            t_R = q2R(np.array([transform.transform.rotation.w, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z]))
            cp_world = t_R @ center_points + np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z])
        except TransformException as e:
            return
        
        self.obs_pose = cp_world

    def detected_goal_pose_callback(self, msg):
        center_points = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        
        try:
            transform = self.tf_buffer.lookup_transform(self.odom_id, msg.header.frame_id, rclpy.time.Time(), rclpy.duration.Duration(seconds=0.1))
            t_R = q2R(np.array([transform.transform.rotation.w, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z]))
            cp_world = t_R @ center_points + np.array([transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z])
        except TransformException as e:
            return
        
        self.goal_pose = cp_world
        
    def get_robot_pose_in_world(self):
        """Get current robot position in world frame"""
        try:
            transform = self.tf_buffer.lookup_transform(self.odom_id, 'base_footprint', rclpy.time.Time())
            x = transform.transform.translation.x
            y = transform.transform.translation.y
            q = transform.transform.rotation
            _, _, yaw = euler_from_quaternion([q.w, q.x, q.y, q.z])
            return np.array([x, y]), yaw
        except TransformException:
            return None, None
            
    def get_poses_in_robot_frame(self):
        """Get obstacle and goal poses in robot frame (for control)"""
        try:
            transform = self.tf_buffer.lookup_transform('base_footprint', self.odom_id, rclpy.time.Time())
            robot_world_x = transform.transform.translation.x
            robot_world_y = transform.transform.translation.y
            robot_world_z = transform.transform.translation.z
            robot_world_R = q2R([transform.transform.rotation.w, transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z])
            
            obs_in_robot = None
            goal_in_robot = None
            
            if self.obs_pose is not None:
                #obs_in_robot = robot_world_R @ self.obs_pose + np.array([robot_world_x, robot_world_y, robot_world_z])
                obs_in_robot = robot_world_R.T @ (self.obs_pose - np.array([robot_world_x, robot_world_y, robot_world_z]))
            
            if self.goal_pose is not None:
                #goal_in_robot = robot_world_R @ self.goal_pose + np.array([robot_world_x, robot_world_y, robot_world_z])
                goal_in_robot = robot_world_R.T @ (self.goal_pose - np.array([robot_world_x, robot_world_y, robot_world_z]))
                
            return obs_in_robot, goal_in_robot
        except TransformException as e:
            self.get_logger().error(f'Transform error: {e}')
            return None, None
    
    def potential_field_controller(self, goal_in_robot, obs_in_robot):
        """
        Potential field controller:
        - Attractive force toward goal
        - Repulsive force away from obstacles
        Returns: (linear_vel, angular_vel)
        """
        if goal_in_robot is None:
            return 0.0, 0.0
        
        # Distance and angle to goal
        goal_dist = np.linalg.norm(goal_in_robot[:2])
        if goal_dist < 0.01:
            return 0.0, 0.0
        
        goal_angle = math.atan2(goal_in_robot[1], goal_in_robot[0])
        
        # Attractive force (proportional to distance)
        # F_att = K_att * (goal - current)
        f_att_x = self.ATTRACTION_GAIN * goal_in_robot[0]
        f_att_y = self.ATTRACTION_GAIN * goal_in_robot[1]
        
        # Repulsive force from obstacles
        f_rep_x = 0.0
        f_rep_y = 0.0
        
        if obs_in_robot is not None:
            obs_dist = np.linalg.norm(obs_in_robot[:2])
            if obs_dist < self.REPULSION_DIST and obs_dist > 0.01:
                # Repulsive force: push AWAY from obstacle
                # If obstacle is at (ox, oy), force should point in direction (-ox, -oy)
                repulsion_strength = self.REPULSION_GAIN * (1.0/obs_dist - 1.0/self.REPULSION_DIST) / (obs_dist ** 2)
                # Force points away from obstacle (opposite direction)
                # Boost lateral avoidance for obstacles to the side
                lateral_boost = 2.0 if abs(obs_in_robot[1]) > abs(obs_in_robot[0]) else 1.0
                f_rep_x = -repulsion_strength * obs_in_robot[0] / obs_dist
                f_rep_y = -repulsion_strength * obs_in_robot[1] / obs_dist * lateral_boost

                # Debug: print what's happening
                # self.get_logger().info(f'Obs: ({obs_in_robot[0]:.2f}, {obs_in_robot[1]:.2f}) Rep: ({f_rep_x:.2f}, {f_rep_y:.2f})')
        
        # Total force
        f_total_x = f_att_x + f_rep_x
        f_total_y = f_att_y + f_rep_y
        
        # Convert to velocity commands
        # Linear velocity: magnitude of force in forward direction
        desired_angle = math.atan2(f_total_y, f_total_x)

        # Angular velocity: align with desired direction
        angular_vel = 2.0 * desired_angle  # Proportional control for heading

        # Linear velocity: faster when aligned, slower when turning
        forward_component = math.cos(desired_angle)

        # Gradual braking: slow down as we approach goal
        if goal_dist < self.SLOW_DIST:
            speed_factor = goal_dist / self.SLOW_DIST  # 0 to 1 scaling
        else:
            speed_factor = 1.0

        linear_vel = self.MAX_LINEAR_VEL * speed_factor * max(0.0, forward_component)

        # Limit velocities
        linear_vel = max(-self.MAX_LINEAR_VEL, min(self.MAX_LINEAR_VEL, linear_vel))
        angular_vel = max(-self.MAX_ANGULAR_VEL, min(self.MAX_ANGULAR_VEL, angular_vel))
        
        return linear_vel, angular_vel
    
    def navigate_to_start(self, current_pos, start_pos):
        """Navigate back to start position using odometry"""
        if current_pos is None or start_pos is None:
            return 0.0, 0.0
        
        # Vector to start
        dx = start_pos[0] - current_pos[0]
        dy = start_pos[1] - current_pos[1]
        dist = np.linalg.norm([dx, dy])
        
        if dist < 0.1:  # Close enough to start
            return 0.0, 0.0
        
        # Get current heading
        _, yaw = self.get_robot_pose_in_world()
        if yaw is None:
            return 0.0, 0.0
        
        # Desired heading
        desired_yaw = math.atan2(dy, dx)
        angle_error = desired_yaw - yaw
        
        # Normalize angle to [-pi, pi]
        while angle_error > math.pi:
            angle_error -= 2 * math.pi
        while angle_error < -math.pi:
            angle_error += 2 * math.pi
        
        # Simple P-controller
        angular_vel = 1.0 * angle_error
        linear_vel = self.MAX_LINEAR_VEL * 0.5 * math.cos(angle_error)
        
        # Limit
        linear_vel = max(0.0, min(self.MAX_LINEAR_VEL, linear_vel))
        angular_vel = max(-self.MAX_ANGULAR_VEL, min(self.MAX_ANGULAR_VEL, angular_vel))
        
        return linear_vel, angular_vel
    
    def timer_update(self):
        # State machine
        if self.state == self.STATE_IDLE:
            # Check if we should start (goal detected)
            if self.goal_pose is not None:
                # Store start position and orientation
                current_pos, current_yaw = self.get_robot_pose_in_world()
                if current_pos is not None:
                    self.start_position = current_pos
                    self.start_yaw = current_yaw
                    self.state = self.STATE_GO_TO_GOAL
                    self.get_logger().info(f'Starting navigation from {self.start_position}, yaw: {self.start_yaw:.2f}')

            # Stop robot
            cmd_vel = Twist()
            self.pub_control_cmd.publish(cmd_vel)
            return

        elif self.state == self.STATE_GO_TO_GOAL:
            if self.goal_pose is None:
                # Lost goal, stop but stay in state
                cmd_vel = Twist()
                self.pub_control_cmd.publish(cmd_vel)
                return

            # Get poses in robot frame
            obs_in_robot, goal_in_robot = self.get_poses_in_robot_frame()

            if goal_in_robot is None:
                cmd_vel = Twist()
                self.pub_control_cmd.publish(cmd_vel)
                return

            # Check if goal reached
            goal_dist = np.linalg.norm(goal_in_robot[:2])

            if goal_dist <= self.GOAL_THRESHOLD:
                self.goal_reached_counter += 1
                if self.goal_reached_counter > 10:  # Confirm for 10 cycles
                    self.state = self.STATE_RETURN_TO_START
                    self.get_logger().info('Goal reached! Returning to start...')
                    self.goal_reached_counter = 0
            else:
                self.goal_reached_counter = 0

            # Use potential field controller
            linear_vel, angular_vel = self.potential_field_controller(goal_in_robot, obs_in_robot)

            cmd_vel = Twist()
            cmd_vel.linear.x = linear_vel
            cmd_vel.angular.z = angular_vel
            self.pub_control_cmd.publish(cmd_vel)

        elif self.state == self.STATE_RETURN_TO_START:
            # Navigate back with obstacle avoidance
            current_pos, current_yaw = self.get_robot_pose_in_world()

            if current_pos is None or current_yaw is None or self.start_position is None:
                cmd_vel = Twist()
                self.pub_control_cmd.publish(cmd_vel)
                return

            # Get obstacle detection
            obs_in_robot, _ = self.get_poses_in_robot_frame()

            # Calculate start position in robot frame
            dx = self.start_position[0] - current_pos[0]
            dy = self.start_position[1] - current_pos[1]

            # Rotate world frame offset to robot frame
            cos_yaw = math.cos(-current_yaw)
            sin_yaw = math.sin(-current_yaw)
            start_in_robot_x = dx * cos_yaw - dy * sin_yaw
            start_in_robot_y = dx * sin_yaw + dy * cos_yaw
            start_in_robot = np.array([start_in_robot_x, start_in_robot_y, 0.0])

            # Check if close to start position
            dist_to_start = np.linalg.norm([dx, dy])
            if dist_to_start < 0.15:
                self.state = self.STATE_ROTATE_TO_START_ORIENTATION
                self.get_logger().info('At start position! Rotating to match start orientation...')
                cmd_vel = Twist()
                self.pub_control_cmd.publish(cmd_vel)
                return

            # Use potential field controller with start as goal
            linear_vel, angular_vel = self.potential_field_controller(start_in_robot, obs_in_robot)

            cmd_vel = Twist()
            cmd_vel.linear.x = linear_vel
            cmd_vel.angular.z = angular_vel
            self.pub_control_cmd.publish(cmd_vel)

        elif self.state == self.STATE_ROTATE_TO_START_ORIENTATION:
            # Rotate in place to match starting orientation
            _, current_yaw = self.get_robot_pose_in_world()

            if current_yaw is None or self.start_yaw is None:
                cmd_vel = Twist()
                self.pub_control_cmd.publish(cmd_vel)
                return

            # Calculate angle difference
            angle_diff = self.start_yaw - current_yaw

            # Normalize to [-pi, pi]
            while angle_diff > math.pi:
                angle_diff -= 2 * math.pi
            while angle_diff < -math.pi:
                angle_diff += 2 * math.pi

            # Check if aligned
            if abs(angle_diff) < self.ANGULAR_TOLERANCE:
                self.state = self.STATE_COMPLETED
                self.get_logger().info('Orientation matched! Task complete.')
                cmd_vel = Twist()
                self.pub_control_cmd.publish(cmd_vel)
                return

            # P-controller for rotation
            angular_vel = 1.5 * angle_diff
            angular_vel = max(-self.MAX_ANGULAR_VEL, min(self.MAX_ANGULAR_VEL, angular_vel))

            cmd_vel = Twist()
            cmd_vel.linear.x = 0.0
            cmd_vel.angular.z = angular_vel
            self.pub_control_cmd.publish(cmd_vel)

        elif self.state == self.STATE_COMPLETED:
            # Auto-reset to IDLE if goal is detected again (tracking toggled back on)
            if self.goal_pose is not None:
                self.state = self.STATE_IDLE
                self.start_position = None
                self.start_yaw = None
                self.get_logger().info('Tracking reset to IDLE - ready for new run')

            # Stop
            cmd_vel = Twist()
            self.pub_control_cmd.publish(cmd_vel)

def main(args=None):
    rclpy.init(args=args)
    tracking_node = TrackingNode()
    rclpy.spin(tracking_node)
    tracking_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
