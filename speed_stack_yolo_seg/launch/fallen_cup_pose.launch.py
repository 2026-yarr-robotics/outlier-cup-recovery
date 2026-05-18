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
        ),
        DeclareLaunchArgument(
            "image_topic",
            default_value="/camera/camera/color/image_raw",
        ),
        DeclareLaunchArgument(
            "depth_topic",
            default_value="/camera/camera/aligned_depth_to_color/image_raw",
        ),
        DeclareLaunchArgument(
            "camera_info_topic",
            default_value="/camera/camera/color/camera_info",
        ),
        DeclareLaunchArgument(
            "imgsz",
            default_value="640",
        ),
        DeclareLaunchArgument(
            "conf",
            default_value="0.25",
        ),
        DeclareLaunchArgument(
            "iou",
            default_value="0.45",
        ),
        DeclareLaunchArgument(
            "device",
            default_value="cpu",
        ),
        DeclareLaunchArgument(
            "half",
            default_value="false",
        ),
        DeclareLaunchArgument(
            "mode",
            default_value="auto",
            description="auto, silhouette, or two_face",
        ),
        DeclareLaunchArgument(
            "use_depth",
            default_value="false",
        ),
        DeclareLaunchArgument(
            "grip_offset_m",
            default_value="0.025",
            description="grip point offset from top center to bottom direction. 0.025 = 2.5cm",
        ),
        DeclareLaunchArgument(
            "top_diameter_m",
            default_value="0.045",
            description="real diameter of smaller top face in meters",
        ),

        Node(
            package="speed_stack_yolo_seg",
            executable="fallen_cup_pose_node",
            name="fallen_cup_pose_node",
            output="screen",
            parameters=[{
                "weights_path": LaunchConfiguration("weights_path"),
                "image_topic": LaunchConfiguration("image_topic"),
                "depth_topic": LaunchConfiguration("depth_topic"),
                "camera_info_topic": LaunchConfiguration("camera_info_topic"),

                "debug_image_topic": "/fallen_cup/debug_image",
                "pose2d_topic": "/fallen_cup/pose2d",
                "grasp_pose_topic": "/fallen_cup/grasp_pose",

                "imgsz": LaunchConfiguration("imgsz"),
                "conf": LaunchConfiguration("conf"),
                "iou": LaunchConfiguration("iou"),
                "device": LaunchConfiguration("device"),
                "half": LaunchConfiguration("half"),

                "mode": LaunchConfiguration("mode"),
                "use_depth": LaunchConfiguration("use_depth"),

                "grip_offset_m": LaunchConfiguration("grip_offset_m"),
                "top_diameter_m": LaunchConfiguration("top_diameter_m"),

                "pixels_per_meter": 0.0,
                "min_mask_area": 300.0,
                "min_pair_distance_px": 20.0,
                "max_pair_distance_px": 10000.0,
            }],
        ),
    ])