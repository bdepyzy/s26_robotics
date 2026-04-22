import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource, PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():

    rviz_cfg = os.path.join(
        get_package_share_directory('disaster_response'),
        'rviz', 'disaster_response.rviz')

    # ── 1. Robot base + IMU + EKF + joystick ──────────────────────────
    bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('yahboomcar_bringup'),
            'launch', 'yahboomcar_bringup_X3_launch.py'))
    )

    # ── 2. RPLiDAR A1 driver → /scan ──────────────────────────────────
    # frame_id must match the URDF link name (laser_link) so gmapping
    # can look up the TF from the scan frame to base_footprint
    lidar = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('sllidar_ros2'),
            'launch', 'sllidar_launch.py')),
        launch_arguments={'frame_id': 'laser_link'}.items()
    )

    # ── 3. Astra Pro RGB-D camera (XML launch file — needs AnyLaunchDescriptionSource)
    camera = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(os.path.join(
            get_package_share_directory('astra_camera'),
            'launch', 'astro_pro_plus.launch.xml'))
    )

    # ── 4. SLAM gmapping → /map ────────────────────────────────────────
    slam = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('slam_gmapping'),
            'launch', 'slam_gmapping.launch.py'))
    )

    # ── 5. Disaster response nodes ────────────────────────────────────
    explorer = Node(
        package='disaster_response',
        executable='lidar_explorer',
        name='lidar_explorer',
        output='screen',
        parameters=[{
            'linear_speed':      0.3,
            'angular_speed':     0.8,
            'stop_dist':         0.5,
            'slow_dist':         0.9,
            'front_warn_thresh': 8,
        }]
    )

    detector = Node(
        package='disaster_response',
        executable='face_victim_detector',
        name='face_victim_detector',
        output='screen',
        parameters=[{
            'cooldown':       5.0,
            'conf_threshold': 0.6,
            'min_face_size':  30,
        }]
    )

    logger = Node(
        package='disaster_response',
        executable='victim_logger',
        name='victim_logger',
        output='screen',
        parameters=[{
            'log_dir': '/root/codes/S26_roboticsII_ws/logs',
        }]
    )

    # ── 6. RViz2 — live map + scan + victim markers ───────────────────
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_cfg],
        output='screen',
    )

    return LaunchDescription([
        bringup,
        lidar,
        camera,
        slam,
        explorer,
        detector,
        logger,
        rviz,
    ])
