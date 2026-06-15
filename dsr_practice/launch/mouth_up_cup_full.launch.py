"""
mouth_up_cup_full.launch.py

mouth-up-cup 처리 전체를 **한 번의 런치**로:
  (1) place_mouth_up_cup  — 컵을 옆에서 잡아 ~90° 롤로 눕혀 작업영역에 내려놓고 종료
  (2) stand_fallen_cup    — (1)이 종료되면 자동 시작, 누운 컵을 vision 재검출해 세움

두 노드를 동시에 띄우지 않고, (1)이 프로세스 종료하면 OnProcessExit 이벤트로 (2)를
시작한다(각자 자기 MoveItPy 를 띄우므로 순차 실행이라야 충돌이 없다). 기존 단독 런치
(place_mouth_up_cup.launch.py / stand_fallen_cup.launch.py)는 그대로 두고, 이 파일만 추가.

전제: 비전 노드는 별도 실행 — (1)용 mouth_up_cup_pose_node 와 (2)용 fallen_cup_pose
노드(speed_stack_yolo_seg, yolo_ws)가 떠 있어야 한다. 둘은 서로 다른 토픽을 쓰므로
동시에 켜 둬도 무방하다. sim:=true 면 vision/HW 없이 가상 컵으로 체인 동작만 확인.

예) 실로봇 한 방:
  ros2 launch dsr_practice mouth_up_cup_full.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler, LogInfo
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    # 두 노드가 공유하는 Doosan M0609 MoveIt 설정.
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
    moveit_dict = moveit_config.to_dict()

    moveit_py_params = PathJoinSubstitution(
        [FindPackageShare("dsr_practice"), "config", "moveit_py.yaml"]
    )

    sim = LaunchConfiguration("sim")
    dry_run = LaunchConfiguration("dry_run")
    robot_namespace = LaunchConfiguration("robot_namespace")
    joint_state_topic = LaunchConfiguration("joint_state_topic")
    recover_mode = LaunchConfiguration("recover_mode")
    sim_cup_x = LaunchConfiguration("sim_cup_x")
    sim_cup_y = LaunchConfiguration("sim_cup_y")
    sim_cup_z = LaunchConfiguration("sim_cup_z")

    # robot_namespace + joint_state_topic 는 두 노드가 동일해야 한다(루트 bringup 기본
    # ""/"/joint_states", namespaced 면 둘 다 dsr01).
    common_ns = {
        "robot_namespace": ParameterValue(robot_namespace, value_type=str),
        "planning_scene_monitor_options.joint_state_topic": ParameterValue(
            joint_state_topic, value_type=str
        ),
    }

    # (1) 눕히기 노드.
    laydown_node = Node(
        package="dsr_practice",
        executable="place_mouth_up_cup",
        name="place_mouth_up_cup",
        output="screen",
        parameters=[
            moveit_dict,
            moveit_py_params,
            {
                "dry_run": ParameterValue(dry_run, value_type=bool),
                "sim": ParameterValue(sim, value_type=bool),
                "sim_cup_x": ParameterValue(sim_cup_x, value_type=float),
                "sim_cup_y": ParameterValue(sim_cup_y, value_type=float),
                "sim_cup_z": ParameterValue(sim_cup_z, value_type=float),
                **common_ns,
            },
        ],
    )

    # (2) recovery 노드. mode 기본 'place'(작업영역에 세우기). place_plus_y_auto_swing
    #     등 나머지는 stand_fallen_cup 노드의 기본값(auto_swing=true 등)을 그대로 쓴다.
    recover_node = Node(
        package="dsr_practice",
        executable="stand_fallen_cup",
        name="stand_fallen_cup",
        output="screen",
        parameters=[
            moveit_dict,
            moveit_py_params,
            {
                "mode": ParameterValue(recover_mode, value_type=str),
                "sim": ParameterValue(sim, value_type=bool),
                **common_ns,
            },
        ],
    )

    # (1) 프로세스가 종료되면 (2) 시작. (1)이 실패해 컵을 못 눕혔어도 (2)는 그냥
    #     '누운 컵 없음'으로 graceful 종료하므로 무조건 체인해도 안전하다.
    chain_recover = RegisterEventHandler(
        OnProcessExit(
            target_action=laydown_node,
            on_exit=[
                LogInfo(msg="[full] 눕히기 노드 종료 → fallen-cup recovery 시작"),
                recover_node,
            ],
        )
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            "sim", default_value="false",
            description="True면 카메라/그리퍼 HW 우회(가상 컵). 체인 동작만 확인용."),
        DeclareLaunchArgument(
            "dry_run", default_value="false",
            description="눕히기 단계 dry_run(접근까지만). recovery 에는 전달 안 함."),
        DeclareLaunchArgument(
            "robot_namespace", default_value="",
            description="namespaced bringup(name:=dsr01)이면 :=dsr01. 기본 ''(루트). "
                        "두 노드에 동일 적용."),
        DeclareLaunchArgument(
            "joint_state_topic", default_value="/joint_states",
            description="planning scene monitor 구독 토픽. 루트면 /joint_states, "
                        "namespaced 면 /<ns>/joint_states. 두 노드에 동일 적용."),
        DeclareLaunchArgument(
            "recover_mode", default_value="place",
            description="recovery 동작: 'place'(작업영역에 세우기) / 'drop'(제자리 세우기)."),
        DeclareLaunchArgument("sim_cup_x", default_value="0.45"),
        DeclareLaunchArgument("sim_cup_y", default_value="0.20"),
        DeclareLaunchArgument("sim_cup_z", default_value="0.145"),

        LogInfo(msg="[full] mouth-up-cup 눕히기 → fallen-cup recovery 파이프라인 시작"),
        laydown_node,
        chain_recover,
    ])
