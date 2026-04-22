"""
Run on your laptop:
    python3 capture_face.py --name YourName

Counts down 3 seconds, then snaps a photo from your webcam (no GUI needed).
Saves face_enroll.jpg, then SCPs it to the Jetson and enrolls it.
"""
import cv2
import subprocess
import argparse
import sys
import time

JETSON_HOST   = 'jetson@192.168.1.129'
JETSON_PASS   = 'yahboom'
REMOTE_TMP    = '/home/jetson/face_enroll.jpg'
CONTAINER     = 'nice_robinson'
CONTAINER_IMG = '/tmp/face_enroll.jpg'
WS_SETUP      = '/root/codes/S26_roboticsII_ws/install/setup.bash'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name',   required=True, help='Your name')
    ap.add_argument('--camera', type=int, default=0, help='Camera index (default 0)')
    ap.add_argument('--delay',  type=int, default=3, help='Countdown seconds (default 3)')
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print('ERROR: could not open camera', args.camera)
        sys.exit(1)

    # Warm up the camera (first few frames are often dark)
    for _ in range(10):
        cap.read()

    print(f'Look straight at the camera …')
    for i in range(args.delay, 0, -1):
        print(f'  {i}…', flush=True)
        time.sleep(1)

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print('ERROR: failed to grab frame')
        sys.exit(1)

    saved = 'face_enroll.jpg'
    cv2.imwrite(saved, frame)
    print(f'Captured → {saved}')

    # SCP to Jetson host filesystem
    print(f'\nCopying to Jetson {JETSON_HOST}:{REMOTE_TMP} …')
    r = subprocess.run(
        ['sshpass', '-p', JETSON_PASS, 'scp', saved, f'{JETSON_HOST}:{REMOTE_TMP}'],
        capture_output=True, text=True)
    if r.returncode != 0:
        print('SCP failed:', r.stderr)
        sys.exit(1)
    print('SCP OK')

    # Copy from host into container
    print('Copying into Docker container …')
    r = subprocess.run(
        ['sshpass', '-p', JETSON_PASS, 'ssh', JETSON_HOST,
         f'docker cp {REMOTE_TMP} {CONTAINER}:{CONTAINER_IMG}'],
        capture_output=True, text=True)
    if r.returncode != 0:
        print('docker cp failed:', r.stderr)
        sys.exit(1)

    # Run enrollment inside container
    print(f'Enrolling "{args.name}" …')
    enroll_cmd = (
        f'source {WS_SETUP} && '
        f'ros2 run disaster_response enroll_faces '
        f'--ros-args -p name:={args.name} -p image_path:={CONTAINER_IMG}'
    )
    r = subprocess.run(
        ['sshpass', '-p', JETSON_PASS, 'ssh', JETSON_HOST,
         f'docker exec {CONTAINER} bash -c "{enroll_cmd}"'],
        capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print('Enrollment error:', r.stderr)
        sys.exit(1)

    print(f'\nDone — "{args.name}" is enrolled on the Jetson.')

if __name__ == '__main__':
    main()
