from setuptools import setup
import os
from glob import glob

package_name = 'disaster_response'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'rviz'),   glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='student',
    maintainer_email='student@example.com',
    description='Disaster search-and-rescue robot',
    license='MIT',
    entry_points={
        'console_scripts': [
            'lidar_explorer       = disaster_response.lidar_explorer:main',
            'victim_detector      = disaster_response.victim_detector:main',
            'victim_logger        = disaster_response.victim_logger:main',
            'face_victim_detector = disaster_response.face_victim_detector:main',
            'enroll_faces         = disaster_response.enroll_faces:main',
        ],
    },
)
