"""
fallen_cup_tracker.launch.py

Phase 1: 컵 트래킹/시각화 노드만 실행.
인식 노드(fallen_cup_pose_node)는 별도로 띄워야 한다 (기존 fallen_cup_pose.launch.py).

사용 예:
  # 터미널 A: 카메라
  ros2 launch realsense2_camera rs_align_depth_launch.py \
      depth_module.depth_profile:=640x480x30 \
      rgb_camera.color_profile:=640x480x30 \
      align_depth.enable:=true

  # 터미널 B: 인식 (기존 노드)
  ros2 launch speed_stack_yolo_seg fallen_cup_pose.launch.py \
      use_depth:=true imgsz:=1280 conf:=0.70

  # 터미널 C: 트래커 (이 파일)
  ros2 launch speed_stack_yolo_seg fallen_cup_tracker.launch.py

  # 시각화: rqt_image_view -> /fallen_cup/tracker_debug_image
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            "window_sec",
            default_value="0.5",
            description="속도 추정용 finite-difference window 길이 (s).",
        ),
        DeclareLaunchArgument(
            "stale_timeout_sec",
            default_value="1.0",
            description="이 시간 동안 detection 없으면 STALE 표시.",
        ),
        DeclareLaunchArgument(
            "arrow_scale_px_per_cm_s",
            default_value="3.0",
            description="속도 화살표 시각화 스케일 (현재 미사용, 표시 horizon은 0.3s 고정).",
        ),
        DeclareLaunchArgument(
            "grasp_pose_topic",
            default_value="/fallen_cup/grasp_pose",
        ),
        DeclareLaunchArgument(
            "pose2d_topic",
            default_value="/fallen_cup/pose2d",
        ),
        DeclareLaunchArgument(
            "debug_image_in_topic",
            default_value="/fallen_cup/debug_image",
        ),
        DeclareLaunchArgument(
            "tracked_state_topic",
            default_value="/fallen_cup/tracked_state",
        ),
        DeclareLaunchArgument(
            "tracker_debug_topic",
            default_value="/fallen_cup/tracker_debug_image",
        ),

        Node(
            package="speed_stack_yolo_seg",
            executable="fallen_cup_tracker_node",
            name="fallen_cup_tracker_node",
            output="screen",
            parameters=[{
                "window_sec": LaunchConfiguration("window_sec"),
                "stale_timeout_sec": LaunchConfiguration("stale_timeout_sec"),
                "arrow_scale_px_per_cm_s":
                    LaunchConfiguration("arrow_scale_px_per_cm_s"),
                "grasp_pose_topic": LaunchConfiguration("grasp_pose_topic"),
                "pose2d_topic": LaunchConfiguration("pose2d_topic"),
                "debug_image_in_topic":
                    LaunchConfiguration("debug_image_in_topic"),
                "tracked_state_topic":
                    LaunchConfiguration("tracked_state_topic"),
                "tracker_debug_topic":
                    LaunchConfiguration("tracker_debug_topic"),
            }],
        ),
    ])
