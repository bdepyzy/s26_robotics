Task 1: Camera RGB and Depth
6
• The usage of the depth camera is different for different types of robots
• Open a new terminal and run ./run_docker.sh
• For each of the steps below, you will open a new terminal and
enter this docker container by running:
• ./exec_docker.sh
• Run ros2 launch astra_camera astra_pro.launch.xml
• ros2 topic list
• rviz2
• Unplug the camera USB cable and plug it back in
• Open a new terminal and run ./run_docker.sh
• For each of the steps below, you will open a new terminal and
enter this docker container by running:
• ./exec_docker.sh
• Run ros2 launch astra_camera
astro_pro_plus.launch.xml
• ros2 topic list
• rviz2
Robots 1-6
Robots 7+
Task 2: Lidar
7
You can start and visualize the lidar scan using the following command:
• ros2 launch sllidar_ros2 view_sllidar_launch.py
You can run obstacle avoidance using lidar as well:
• Start the robot: ros2 run yahboomcar_bringup Mcnamu_driver_X3
• Start the lidar: ros2 launch sllidar_ros2 sllidar_launch.py
• Start the avoidance node: ros2 run yahboomcar_laser laser_Avoidance_a1_X3
• Start the keyboard control: ros2 run yahboomcar_ctrl yahboom_keyboard
Task 3: Robot State Estimation
8
• Run the robot:
• ros2 launch yahboomcar_bringup yahboomcar_bringup_X3_launch.py
• To drive the robot via keyboard, run:
• ros2 run yahboomcar_ctrl yahboom_keyboard
• To observe raw odometry data, run:
• ros2 topic echo /odom_raw
• To observe the odometry data filtered by EKF, run:
• ros2 topic echo /odom
• Observe the nodes and topics:
• ros2 run rqt_graph rqt_graph
• Discuss the questions related to this task
Task 4: Mapping and SLAM
9
1. Complete the step 6.2.2 of the tutorial “25.6 gmapping mapping algorithm”.
2. Save some images of the map you created
