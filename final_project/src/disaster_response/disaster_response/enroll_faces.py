"""
Standalone enrollment — no ROS launch needed.

Usage:
    python3 -m disaster_response.enroll_faces Daniel
  OR via ros2:
    ros2 run disaster_response enroll_faces --ros-args -p name:=Daniel
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import json
import sys
import select
import time

MODEL_DIR   = '/root/codes/models'
SFACE_MODEL = os.path.join(MODEL_DIR, 'face_recognition_sface_2021dec.onnx')
ENROLL_FILE = os.path.join(MODEL_DIR, 'enrolled_faces.json')
FACE_SIZE   = (112, 112)
HAAR        = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'


def _detect_and_crop(bgr):
    cascade = cv2.CascadeClassifier(HAAR)
    gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None, 'No face detected — make sure your face is clearly visible and well-lit.'
    if len(faces) > 1:
        return None, f'{len(faces)} faces detected — only one person should be in frame.'
    x, y, w, h = faces[0]
    return cv2.resize(bgr[y:y+h, x:x+w], FACE_SIZE), None


def _save_enrollment(name, recogniser, face_img):
    feature  = recogniser.feature(face_img)
    enrolled = {}
    if os.path.exists(ENROLL_FILE):
        with open(ENROLL_FILE) as f:
            enrolled = json.load(f)
    enrolled[name] = feature.flatten().tolist()
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(ENROLL_FILE, 'w') as f:
        json.dump(enrolled, f)
    return enrolled


def enroll_from_camera(name, device=0):
    """Standalone: open camera directly, no ROS required."""
    if not os.path.exists(SFACE_MODEL):
        print(f'ERROR: Missing SFace model: {SFACE_MODEL}')
        sys.exit(1)
    recogniser = cv2.FaceRecognizerSF.create(SFACE_MODEL, '')

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        print(f'ERROR: Cannot open camera device {device}')
        sys.exit(1)

    # Warm up
    for _ in range(15):
        cap.read()

    print(f'Camera open (device {device})')
    print(f'Enrolling: "{name}"')
    print('Press Enter to capture, Ctrl+C to cancel.')

    while True:
        input('>> Press Enter when face is in frame … ')
        for _ in range(3):   # grab a fresh frame (flush buffer)
            ret, frame = cap.read()
        if not ret:
            print('Failed to grab frame, try again.')
            continue

        face_img, err = _detect_and_crop(frame)
        if face_img is None:
            print(f'  ✗ {err}')
            continue

        enrolled = _save_enrollment(name, recogniser, face_img)
        print(f'  ✓ Enrolled "{name}"')
        print(f'  All enrolled: {list(enrolled.keys())}')
        break

    cap.release()


# ── ROS node (used when launched via ros2 run) ────────────────────────────────

class EnrollFaces(Node):
    def __init__(self):
        super().__init__('enroll_faces')
        self.declare_parameter('name',       '')
        self.declare_parameter('image_path', '')
        self.declare_parameter('camera_device', 0)

        self._name = self.get_parameter('name').value.strip()
        if not self._name:
            self.get_logger().error('Parameter "name" is required. Pass --ros-args -p name:=<Name>')
            raise SystemExit(1)

        if not os.path.exists(SFACE_MODEL):
            raise RuntimeError(f'Missing SFace model: {SFACE_MODEL}')
        self._recogniser = cv2.FaceRecognizerSF.create(SFACE_MODEL, '')
        self._done = False

        image_path = self.get_parameter('image_path').value.strip()
        if image_path:
            self._enroll_from_file(image_path)
            return

        # Try direct camera first (no ROS topics needed)
        device = self.get_parameter('camera_device').value
        self.get_logger().info(
            f'Enrolling "{self._name}" — opening camera device {device} directly …')
        self._enroll_from_device(device)

    def _enroll_from_device(self, device):
        cap = cv2.VideoCapture(device)
        if not cap.isOpened():
            self.get_logger().error(f'Cannot open /dev/video{device} — falling back to ROS topic')
            self._start_ros_subscriber()
            return

        for _ in range(15):
            cap.read()

        self.get_logger().info('Camera open. Press Enter when your face is in frame …')

        while not self._done:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                sys.stdin.readline()
                for _ in range(3):
                    ret, frame = cap.read()
                if not ret:
                    self.get_logger().warn('Failed to grab frame, try again.')
                    continue

                face_img, err = _detect_and_crop(frame)
                if face_img is None:
                    self.get_logger().warn(f'{err}  — try again.')
                    continue

                enrolled = _save_enrollment(self._name, self._recogniser, face_img)
                self.get_logger().info(f'Enrolled "{self._name}" → {ENROLL_FILE}')
                self.get_logger().info(f'All enrolled: {list(enrolled.keys())}')
                self._done = True

        cap.release()

    def _enroll_from_file(self, path):
        if not os.path.exists(path):
            self.get_logger().error(f'Image not found: {path}')
            self._done = True
            return
        bgr = cv2.imread(path)
        if bgr is None:
            self.get_logger().error(f'Could not read image: {path}')
            self._done = True
            return
        face_img, err = _detect_and_crop(bgr)
        if face_img is None:
            self.get_logger().error(f'Enrollment failed: {err}')
            self._done = True
            return
        enrolled = _save_enrollment(self._name, self._recogniser, face_img)
        self.get_logger().info(f'Enrolled "{self._name}" → {ENROLL_FILE}')
        self.get_logger().info(f'All enrolled: {list(enrolled.keys())}')
        self._done = True

    def _start_ros_subscriber(self):
        self._br = CvBridge()
        self._capture_next = False
        self._sub = self.create_subscription(
            Image, '/camera/color/image_raw', self._image_cb, 10)
        self._timer = self.create_timer(0.1, self._check_input)
        self.get_logger().info('Waiting for /camera/color/image_raw … press Enter to capture.')

    def _check_input(self):
        if select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
            self._capture_next = True
            self.get_logger().info('Capturing …')

    def _image_cb(self, msg):
        if self._done or not self._capture_next:
            return
        self._capture_next = False
        bgr = self._br.imgmsg_to_cv2(msg, 'bgr8')
        face_img, err = _detect_and_crop(bgr)
        if face_img is None:
            self.get_logger().warn(f'{err}  — press Enter to try again.')
            return
        enrolled = _save_enrollment(self._name, self._recogniser, face_img)
        self.get_logger().info(f'Enrolled "{self._name}" → {ENROLL_FILE}')
        self.get_logger().info(f'All enrolled: {list(enrolled.keys())}')
        self._done = True


def main(args=None):
    # Allow running as a plain Python script: python3 enroll_faces.py Daniel
    if args is None and len(sys.argv) > 1 and not sys.argv[1].startswith('--ros-args'):
        enroll_from_camera(sys.argv[1], device=int(sys.argv[2]) if len(sys.argv) > 2 else 0)
        return

    rclpy.init(args=args)
    try:
        node = EnrollFaces()
    except SystemExit:
        rclpy.shutdown()
        return

    try:
        while rclpy.ok() and not node._done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
