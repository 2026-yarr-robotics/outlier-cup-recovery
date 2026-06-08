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
            "target_class_name",
            default_value="fallen-cup",
            description=(
                "방향벡터를 뽑을 YOLO 클래스 이름. "
                "이 클래스의 mask만 사용하고 나머지(upright-cup 등)는 무시한다. "
                "빈 문자열이면 필터링 끔."
            ),
        ),
        DeclareLaunchArgument(
            "use_depth",
            default_value="false",
        ),
        DeclareLaunchArgument(
            "grip_offset_m",
            default_value="0.015",
            description="grip point offset from top center toward bottom (m). "
                        "Default 0.015 grips near the narrow tip so the cup can be "
                        "tilted upright without the wide end colliding with the table.",
        ),
        DeclareLaunchArgument(
            "top_diameter_m",
            default_value="0.045",
            description="real diameter of smaller top face in meters",
        ),
        DeclareLaunchArgument(
            "min_pair_diameter_ratio",
            default_value="1.3",
            description=(
                "two_face pair 의 narrow/wide 직경 비 최소값. "
                "한 컵의 두 face 는 ≥~1.5, 별개 두 컵은 ≈1.0 이라 이 값으로 "
                "cross-cup pairing 을 차단한다. 너무 키우면 valid 한 two_face 도 거부됨."
            ),
        ),
        DeclareLaunchArgument(
            "enable_axis_length_filter",
            default_value="true",
            description=(
                "축 길이 sanity 필터 on/off. 넘어진 컵 top→bottom 축 실제 길이는 "
                "거의 일정 → 정상 밴드 밖(두 컵 병합/cross-cup)이면 거부. "
                "튜닝 전엔 false 로 두고 로그의 axis_cm 측정 후 켜는 것을 권장."
            ),
        ),
        DeclareLaunchArgument(
            "expected_axis_length_m",
            default_value="0.075",
            description="정상 넘어진 컵의 top→bottom 축 실제 길이(m). 측정값 7.5cm "
                        "(silhouette 단일 컵, 2026-06-08). 로그 axis_cm 으로 재측정해 보정 가능.",
        ),
        DeclareLaunchArgument(
            "axis_length_tol_m",
            default_value="0.02",
            description="허용 오차(±m). expected±tol 밖이면 거부. 기본 0.02=±2cm.",
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
                "target_class_name": LaunchConfiguration("target_class_name"),
                "use_depth": LaunchConfiguration("use_depth"),

                "grip_offset_m": LaunchConfiguration("grip_offset_m"),
                "top_diameter_m": LaunchConfiguration("top_diameter_m"),

                "pixels_per_meter": 0.0,
                "min_mask_area": 300.0,
                "min_pair_distance_px": 20.0,
                "max_pair_distance_px": 10000.0,
                "min_pair_diameter_ratio": LaunchConfiguration("min_pair_diameter_ratio"),

                "enable_axis_length_filter": LaunchConfiguration("enable_axis_length_filter"),
                "expected_axis_length_m": LaunchConfiguration("expected_axis_length_m"),
                "axis_length_tol_m": LaunchConfiguration("axis_length_tol_m"),
            }],
        ),
    ])