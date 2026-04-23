# S26 Robotics — Disaster Response

ROS2 autonomous search-and-rescue robot. Uses LiDAR-based SLAM exploration and RGB-D face recognition to locate and identify victims.

## OpenCV Models

The face recognition node requires the SFace ONNX model. Place it at `/root/codes/models/face_recognition_sface_2021dec.onnx` inside the Docker container.

Download: https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx

The Haar cascade (`haarcascade_frontalface_default.xml`) is bundled with OpenCV and requires no manual download.

## Usage

```bash
ros2 launch disaster_response disaster_response_launch.py
```
