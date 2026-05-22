from setuptools import find_packages, setup
import os
from glob import glob

package_name = "speed_stack_yolo_seg"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "weights"), glob("weights/*.pt")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="ssu",
    maintainer_email="ssu@example.com",
    description="ROS2 YOLO segmentation node for speed stack cup top segmentation",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "yolo_seg_node = speed_stack_yolo_seg.yolo_seg_node:main",
            "fallen_cup_pose_node = speed_stack_yolo_seg.fallen_cup_pose_node:main",
            "fallen_cup_tracker_node = speed_stack_yolo_seg.fallen_cup_tracker_node:main",
        ],
    },
)