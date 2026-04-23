"""
Run on the Jetson host:
    python3 capture_face.py --name YourName

Snaps a photo, copies it into the running Docker container, and enrolls it.
"""
import cv2
import subprocess
import argparse
import sys
import time

CONTAINER_IMG = '/tmp/face_enroll.jpg'
WS_SETUP      = '/root/codes/S26_roboticsII_ws/install/setup.bash'


def get_container():
    r = subprocess.run(
        ['docker', 'ps', '--filter', 'ancestor=erickiemvp/robotics2_docker_image', '--format', '{{.Names}}'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    name = r.stdout.decode().strip().splitlines()[0] if r.stdout.strip() else None
    if not name:
        print('ERROR: no running robotics2 container found — run ./run_docker.sh first')
        sys.exit(1)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name',  required=True, help='Person name to enroll')
    ap.add_argument('--image', default=None,  help='Path to existing photo (skips camera)')
    args = ap.parse_args()

    saved = '/tmp/face_enroll.jpg'

    if args.image:
        if not os.path.exists(args.image):
            print(f'ERROR: image not found: {args.image}')
            sys.exit(1)
        import shutil
        shutil.copy(args.image, saved)
        print(f'Using image: {args.image}')
    else:
        print('ERROR: Astra camera is not accessible from the Jetson host.')
        print('Take a photo with your phone, copy it here, then run:')
        print(f'  python3 capture_face.py --name {args.name} --image /path/to/photo.jpg')
        sys.exit(1)

    container = get_container()
    print(f'Using container: {container}')

    r = subprocess.run(['docker', 'cp', saved, f'{container}:{CONTAINER_IMG}'],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        print('docker cp failed:', r.stderr.decode())
        sys.exit(1)

    print(f'Enrolling "{args.name}" …')
    enroll_cmd = (
        f'source {WS_SETUP} && '
        f'ros2 run disaster_response enroll_faces '
        f'--ros-args -p name:={args.name} -p image_path:={CONTAINER_IMG}'
    )
    r = subprocess.run(['docker', 'exec', container, 'bash', '-c', enroll_cmd],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(r.stdout.decode())
    if r.returncode != 0:
        print('Enrollment error:', r.stderr.decode())
        sys.exit(1)

    print(f'Done — "{args.name}" is enrolled.')


if __name__ == '__main__':
    main()
