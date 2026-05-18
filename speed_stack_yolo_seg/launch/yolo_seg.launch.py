from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    pkg_share = get_package_share_directory("speed_stack_yolo_seg")
    default_weights = os.path.join(pkg_share, "weights", "best.pt")

    return LaunchDescription([
        DeclareLaunchArgument(
            "weights_path",
            default_value=default_weights,
            description="Path to YOLO segmentation .pt weight file",
        ),
        DeclareLaunchArgument(
            "image_topic",
            default_value="/camera/color/image_raw",
            description="Input camera image topic",
        ),
        DeclareLaunchArgument(
            "imgsz",
            default_value="1280",
            description="YOLO inference image size",
        ),
        DeclareLaunchArgument(
            "conf",
            default_value="0.25",
            description="YOLO confidence threshold",
        ),
        DeclareLaunchArgument(
            "iou",
            default_value="0.45",
            description="YOLO IoU threshold",
        ),
        DeclareLaunchArgument(
            "device",
            default_value="0",
            description="'0' for CUDA GPU 0, or 'cpu'",
        ),
        DeclareLaunchArgument(
            "half",
            default_value="true",
            description="Use FP16 inference on CUDA",
        ),
        DeclareLaunchArgument(
            "frame_skip",
            default_value="0",
            description="0 means process every frame. 1 means process one and skip one.",
        ),

        Node(
            package="speed_stack_yolo_seg",
            executable="yolo_seg_node",
            name="yolo_seg_node",
            output="screen",
            parameters=[{
                "weights_path": LaunchConfiguration("weights_path"),
                "image_topic": LaunchConfiguration("image_topic"),
                "overlay_topic": "/yolo_seg/overlay",
                "mask_topic": "/yolo_seg/mask",
                "count_topic": "/yolo_seg/count",
                "imgsz": LaunchConfiguration("imgsz"),
                "conf": LaunchConfiguration("conf"),
                "iou": LaunchConfiguration("iou"),
                "device": LaunchConfiguration("device"),
                "half": LaunchConfiguration("half"),
                "retina_masks": True,
                "frame_skip": LaunchConfiguration("frame_skip"),
            }],
        ),
    ])