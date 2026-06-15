from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
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

    dry_run = LaunchConfiguration("dry_run")
    use_current_as_home = LaunchConfiguration("use_current_as_home")
    approach_side = LaunchConfiguration("approach_side")
    grip_tilt_deg = LaunchConfiguration("grip_tilt_deg")
    robot_namespace = LaunchConfiguration("robot_namespace")
    joint_state_topic = LaunchConfiguration("joint_state_topic")
    sim = LaunchConfiguration("sim")
    sim_cup_x = LaunchConfiguration("sim_cup_x")
    sim_cup_y = LaunchConfiguration("sim_cup_y")
    sim_cup_z = LaunchConfiguration("sim_cup_z")

    return LaunchDescription([
        DeclareLaunchArgument(
            "dry_run", default_value="false",
            description="True면 접근 자세까지만, gripper/insert/lift 스킵"),
        DeclareLaunchArgument(
            "use_current_as_home", default_value="false",
            description="False(기본)면 코드 고정 원점 HOME_JOINTS 사용. "
                        "True면 launch 시점 현재 자세를 세션 HOME 으로 캡처"),
        DeclareLaunchArgument(
            "approach_side", default_value="auto",
            description="수평 접근 방향: 'auto'(컵 base Y 부호) / 'left'(+Y) / 'right'(-Y)"),
        DeclareLaunchArgument(
            "grip_tilt_deg", default_value="15.0",
            description="grab 시 ee_z 의 수평 아래 기울기(카메라-바닥 클리어용). 순수 joint_6 "
                        "180° flip 후 mouth-down 컵 기울기 = 2·이 값. 0 이면 정확한 수직 "
                        "mouth-down(단 grab 시 바닥 클리어 확인 필요)."),
        DeclareLaunchArgument(
            "robot_namespace", default_value="",
            description="namespaced bringup(name:=dsr01)이면 :=dsr01. 기본 ''(루트)."),
        DeclareLaunchArgument(
            "joint_state_topic", default_value="/joint_states",
            description="planning scene monitor 가 구독할 joint state 토픽. "
                        "루트 bringup 이면 /joint_states, namespaced 면 /<ns>/joint_states."),
        DeclareLaunchArgument(
            "sim", default_value="false",
            description="True면 카메라/그리퍼 HW 우회 (MoveIt virtual)"),
        DeclareLaunchArgument("sim_cup_x", default_value="0.45"),
        DeclareLaunchArgument("sim_cup_y", default_value="0.20"),
        DeclareLaunchArgument("sim_cup_z", default_value="0.145"),

        Node(
            package="dsr_practice",
            executable="place_mouth_up_cup",
            output="screen",
            parameters=[
                moveit_config.to_dict(),
                moveit_py_params,
                {
                    "dry_run": ParameterValue(dry_run, value_type=bool),
                    "use_current_as_home": ParameterValue(
                        use_current_as_home, value_type=bool),
                    "approach_side": ParameterValue(
                        approach_side, value_type=str),
                    "grip_tilt_deg": ParameterValue(
                        grip_tilt_deg, value_type=float),
                    "robot_namespace": ParameterValue(
                        robot_namespace, value_type=str),
                    "sim": ParameterValue(sim, value_type=bool),
                    "sim_cup_x": ParameterValue(sim_cup_x, value_type=float),
                    "sim_cup_y": ParameterValue(sim_cup_y, value_type=float),
                    "sim_cup_z": ParameterValue(sim_cup_z, value_type=float),
                    "planning_scene_monitor_options.joint_state_topic": (
                        ParameterValue(joint_state_topic, value_type=str)),
                },
            ],
        )
    ])
