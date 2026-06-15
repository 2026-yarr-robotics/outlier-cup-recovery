from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder


def _fail_launch_on_nonzero_exit(event, context):
    """outlier_cup_recovery 노드가 비정상 종료(코드!=0)하면 launch 자체를 실패시킨다.

    ROS2 launch 는 자식 노드가 non-zero 로 죽어도 `ros2 launch` 종료코드를 0 으로
    두므로, 여기서 예외를 던져 LaunchService 가 종료코드 1 을 반환하게 한다. 이
    코드가 wrapper launch 를 거쳐 서버 LaunchManager 까지 전파돼 task 가 failed
    로 잡힌다. (stand_fallen_cup.launch.py 와 동일 계약.)
    """
    if event.returncode != 0:
        raise RuntimeError(
            f"outlier_cup_recovery 노드 비정상 종료(exit={event.returncode}) "
            "— recovery 실패를 launch 종료코드로 전파"
        )
    return None


def generate_launch_description():
    # Doosan M0609 MoveIt 기본 설정 (두 스킬이 공유하는 단일 MoveItPy 가 사용).
    moveit_config = (
        MoveItConfigsBuilder(
            robot_name="m0609",
            package_name="dsr_moveit_config_m0609",
        )
        .robot_description()
        .robot_description_semantic(file_path="config/dsr.srdf")
        .robot_description_kinematics()
        .joint_limits()
        .trajectory_execution()
        .planning_scene_monitor()
        .to_moveit_configs()
    )

    moveit_py_params = PathJoinSubstitution(
        [FindPackageShare("dsr_practice"), "config", "moveit_py.yaml"]
    )

    # ── 공통 + fallen 파라미터 ──
    dry_run = LaunchConfiguration("dry_run")
    use_current_as_home = LaunchConfiguration("use_current_as_home")
    sim = LaunchConfiguration("sim")
    robot_namespace = LaunchConfiguration("robot_namespace")
    joint_state_topic = LaunchConfiguration("joint_state_topic")

    cup_yaw_override_deg = LaunchConfiguration("cup_yaw_override_deg")
    mode = LaunchConfiguration("mode")
    place_flange_side = LaunchConfiguration("place_flange_side")
    place_flange_yaw_deg = LaunchConfiguration("place_flange_yaw_deg")
    place_flange_yaw_auto_extra_deg = LaunchConfiguration(
        "place_flange_yaw_auto_extra_deg"
    )
    stand_cup_margin_m = LaunchConfiguration("stand_cup_margin_m")
    place_base_yaw_deg = LaunchConfiguration("place_base_yaw_deg")
    place_cup_tilt_deg = LaunchConfiguration("place_cup_tilt_deg")
    place_plus_y_auto_swing = LaunchConfiguration("place_plus_y_auto_swing")
    place_plus_y_side = LaunchConfiguration("place_plus_y_side")
    place_plus_y_base_yaw_deg = LaunchConfiguration("place_plus_y_base_yaw_deg")
    place_plus_y_cup_tilt_deg = LaunchConfiguration("place_plus_y_cup_tilt_deg")
    place_x = LaunchConfiguration("place_x")
    place_y = LaunchConfiguration("place_y")
    # multi_cup 은 오케스트레이터가 강제로 True 로 둔다(fallen 먼저·최근접 우선).
    # 여기선 부수 파라미터만 노출.
    multi_cup_max_iterations = LaunchConfiguration("multi_cup_max_iterations")
    multi_cup_cluster_radius_m = LaunchConfiguration("multi_cup_cluster_radius_m")
    multi_cup_blacklist_radius_m = LaunchConfiguration("multi_cup_blacklist_radius_m")
    multi_cup_min_samples_per_cluster = LaunchConfiguration(
        "multi_cup_min_samples_per_cluster"
    )
    place_in_place = LaunchConfiguration("place_in_place")
    place_spot_candidates = LaunchConfiguration("place_spot_candidates")
    place_spot_avoid_radius_m = LaunchConfiguration("place_spot_avoid_radius_m")
    upright_boxes_topic = LaunchConfiguration("upright_boxes_topic")
    pyramid_avoid = LaunchConfiguration("pyramid_avoid")
    pyramid_config_url = LaunchConfiguration("pyramid_config_url")
    pyramid_stack_topic = LaunchConfiguration("pyramid_stack_topic")
    pyramid_sync_poll_period_s = LaunchConfiguration("pyramid_sync_poll_period_s")
    pyramid_obstacle_margin_m = LaunchConfiguration("pyramid_obstacle_margin_m")
    pyramid_stack_wait_s = LaunchConfiguration("pyramid_stack_wait_s")
    avoid_upright_cups = LaunchConfiguration("avoid_upright_cups")
    upright_obstacle_radius_m = LaunchConfiguration("upright_obstacle_radius_m")
    upright_obstacle_height_m = LaunchConfiguration("upright_obstacle_height_m")
    preflight_reach_check = LaunchConfiguration("preflight_reach_check")

    # ── mouth-up 전용 파라미터 ──
    approach_side = LaunchConfiguration("approach_side")
    grip_tilt_deg = LaunchConfiguration("grip_tilt_deg")
    grip_z_offset = LaunchConfiguration("grip_z_offset")

    outlier_node = Node(
        package="dsr_practice",
        executable="outlier_cup_recovery",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            moveit_py_params,
            {
                # 공통 (두 스킬 모두 declare → /** 와일드카드로 동일 적용)
                "dry_run": ParameterValue(dry_run, value_type=bool),
                "use_current_as_home": ParameterValue(
                    use_current_as_home, value_type=bool),
                "sim": ParameterValue(sim, value_type=bool),
                "robot_namespace": ParameterValue(
                    robot_namespace, value_type=str),
                "planning_scene_monitor_options.joint_state_topic": (
                    ParameterValue(joint_state_topic, value_type=str)),

                # fallen 전용 (mouth-up 은 declare 안 함 → 무시)
                "cup_yaw_override_deg": ParameterValue(
                    cup_yaw_override_deg, value_type=float),
                "mode": ParameterValue(mode, value_type=str),
                "place_flange_side": ParameterValue(
                    place_flange_side, value_type=str),
                "place_flange_yaw_deg": ParameterValue(
                    place_flange_yaw_deg, value_type=float),
                "place_flange_yaw_auto_extra_deg": ParameterValue(
                    place_flange_yaw_auto_extra_deg, value_type=float),
                "stand_cup_margin_m": ParameterValue(
                    stand_cup_margin_m, value_type=float),
                "place_base_yaw_deg": ParameterValue(
                    place_base_yaw_deg, value_type=float),
                "place_cup_tilt_deg": ParameterValue(
                    place_cup_tilt_deg, value_type=float),
                "place_plus_y_auto_swing": ParameterValue(
                    place_plus_y_auto_swing, value_type=bool),
                "place_plus_y_side": ParameterValue(
                    place_plus_y_side, value_type=str),
                "place_plus_y_base_yaw_deg": ParameterValue(
                    place_plus_y_base_yaw_deg, value_type=float),
                "place_plus_y_cup_tilt_deg": ParameterValue(
                    place_plus_y_cup_tilt_deg, value_type=float),
                "place_x": ParameterValue(place_x, value_type=float),
                "place_y": ParameterValue(place_y, value_type=float),
                "multi_cup_max_iterations": ParameterValue(
                    multi_cup_max_iterations, value_type=int),
                "multi_cup_cluster_radius_m": ParameterValue(
                    multi_cup_cluster_radius_m, value_type=float),
                "multi_cup_blacklist_radius_m": ParameterValue(
                    multi_cup_blacklist_radius_m, value_type=float),
                "multi_cup_min_samples_per_cluster": ParameterValue(
                    multi_cup_min_samples_per_cluster, value_type=int),
                "place_in_place": ParameterValue(
                    place_in_place, value_type=bool),
                "place_spot_candidates": ParameterValue(
                    place_spot_candidates, value_type=str),
                "place_spot_avoid_radius_m": ParameterValue(
                    place_spot_avoid_radius_m, value_type=float),
                "upright_boxes_topic": ParameterValue(
                    upright_boxes_topic, value_type=str),
                "pyramid_avoid": ParameterValue(
                    pyramid_avoid, value_type=bool),
                "pyramid_config_url": ParameterValue(
                    pyramid_config_url, value_type=str),
                "pyramid_stack_topic": ParameterValue(
                    pyramid_stack_topic, value_type=str),
                "pyramid_sync_poll_period_s": ParameterValue(
                    pyramid_sync_poll_period_s, value_type=float),
                "pyramid_obstacle_margin_m": ParameterValue(
                    pyramid_obstacle_margin_m, value_type=float),
                "pyramid_stack_wait_s": ParameterValue(
                    pyramid_stack_wait_s, value_type=float),
                "avoid_upright_cups": ParameterValue(
                    avoid_upright_cups, value_type=bool),
                "upright_obstacle_radius_m": ParameterValue(
                    upright_obstacle_radius_m, value_type=float),
                "upright_obstacle_height_m": ParameterValue(
                    upright_obstacle_height_m, value_type=float),
                "preflight_reach_check": ParameterValue(
                    preflight_reach_check, value_type=bool),

                # mouth-up 전용 (fallen 은 declare 안 함 → 무시)
                "approach_side": ParameterValue(approach_side, value_type=str),
                "grip_tilt_deg": ParameterValue(grip_tilt_deg, value_type=float),
                "grip_z_offset": ParameterValue(grip_z_offset, value_type=float),
            },
        ],
    )

    return LaunchDescription([
        # ── 공통 ──
        DeclareLaunchArgument(
            "dry_run", default_value="false",
            description="True면 접근 자세까지만, gripper/insert/lift 스킵 (양 스킬 공통)"),
        DeclareLaunchArgument(
            "use_current_as_home", default_value="false",
            description="False(기본)면 코드 고정 원점 HOME_JOINTS 사용. "
                        "True면 launch 시점 현재 자세를 세션 HOME 으로 캡처"),
        DeclareLaunchArgument(
            "sim", default_value="false",
            description="True면 카메라/그리퍼 HW 우회 (MoveIt virtual). 통합 "
                        "모듈은 실로봇 용 — sim 컵 좌표는 각 스킬 기본값 사용."),
        DeclareLaunchArgument(
            "robot_namespace", default_value="",
            description="namespaced bringup(name:=dsr01)이면 :=dsr01. 기본 ''(루트)."),
        DeclareLaunchArgument(
            "joint_state_topic", default_value="/joint_states",
            description="planning scene monitor 가 구독할 joint state 토픽."),

        # ── fallen 전용 ──
        DeclareLaunchArgument("cup_yaw_override_deg", default_value="nan"),
        DeclareLaunchArgument("mode", default_value="drop"),
        DeclareLaunchArgument("place_flange_side", default_value="right"),
        DeclareLaunchArgument("place_flange_yaw_deg", default_value="nan"),
        DeclareLaunchArgument(
            "place_flange_yaw_auto_extra_deg", default_value="nan"),
        DeclareLaunchArgument("stand_cup_margin_m", default_value="-0.05"),
        DeclareLaunchArgument("place_base_yaw_deg", default_value="nan"),
        DeclareLaunchArgument("place_cup_tilt_deg", default_value="0.0"),
        DeclareLaunchArgument("place_plus_y_auto_swing", default_value="true"),
        DeclareLaunchArgument("place_plus_y_side", default_value="left"),
        DeclareLaunchArgument("place_plus_y_base_yaw_deg", default_value="60.0"),
        DeclareLaunchArgument("place_plus_y_cup_tilt_deg", default_value="25.0"),
        DeclareLaunchArgument("place_x", default_value="nan"),
        DeclareLaunchArgument("place_y", default_value="nan"),
        DeclareLaunchArgument("multi_cup_max_iterations", default_value="10"),
        DeclareLaunchArgument("multi_cup_cluster_radius_m", default_value="0.04"),
        DeclareLaunchArgument(
            "multi_cup_blacklist_radius_m", default_value="0.06"),
        DeclareLaunchArgument(
            "multi_cup_min_samples_per_cluster", default_value="3"),
        DeclareLaunchArgument("place_in_place", default_value="false"),
        DeclareLaunchArgument(
            "place_spot_candidates",
            default_value="0.30:0.10,0.30:0.00,0.30:-0.10"),
        DeclareLaunchArgument("place_spot_avoid_radius_m", default_value="0.09"),
        DeclareLaunchArgument(
            "upright_boxes_topic", default_value="/hand_eye/boxes"),
        DeclareLaunchArgument("pyramid_avoid", default_value="true"),
        DeclareLaunchArgument(
            "pyramid_config_url",
            default_value="https://yarr-api-31.simplyimg.com/api/robot/config/pyramid"),
        DeclareLaunchArgument("pyramid_stack_topic", default_value="/stack"),
        DeclareLaunchArgument(
            "pyramid_sync_poll_period_s", default_value="5.0"),
        DeclareLaunchArgument("pyramid_obstacle_margin_m", default_value="0.02"),
        DeclareLaunchArgument("pyramid_stack_wait_s", default_value="1.5"),
        DeclareLaunchArgument("avoid_upright_cups", default_value="true"),
        DeclareLaunchArgument("upright_obstacle_radius_m", default_value="0.04"),
        DeclareLaunchArgument("upright_obstacle_height_m", default_value="0.12"),
        DeclareLaunchArgument("preflight_reach_check", default_value="true"),

        # ── mouth-up 전용 ──
        DeclareLaunchArgument(
            "approach_side", default_value="auto",
            description="mouth-up 수평 접근 방향: 'auto' / 'left'(+Y) / 'right'(-Y)"),
        DeclareLaunchArgument(
            "grip_tilt_deg", default_value="15.0",
            description="mouth-up grab 시 ee_z 하향 기울기(카메라-바닥 클리어)."),
        DeclareLaunchArgument(
            "grip_z_offset", default_value="-0.03",
            description="mouth-up 그립 높이 미세조정(m). 기본 -0.03(기하중심 아래 3cm)."),

        outlier_node,
        # 노드가 non-zero 로 죽으면 launch 종료코드도 non-zero 로 전파.
        RegisterEventHandler(
            OnProcessExit(
                target_action=outlier_node,
                on_exit=_fail_launch_on_nonzero_exit,
            )
        ),
    ])
