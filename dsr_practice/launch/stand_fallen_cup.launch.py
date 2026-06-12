from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder


def _fail_launch_on_nonzero_exit(event, context):
    """stand_fallen_cup 노드가 비정상 종료(코드!=0)하면 launch 자체를 실패시킨다.

    ROS2 launch 는 자식 노드가 non-zero 로 죽어도 `ros2 launch` 종료코드를
    0 으로 두기 때문에(LaunchService.run 은 launch 시스템 예외일 때만 1 반환),
    여기서 예외를 던져 LaunchService 가 종료코드 1 을 반환하게 만든다. 이 코드가
    cup_stack wrapper launch 를 거쳐 서버 LaunchManager 까지 전파돼야
    /api/robot/status 의 fallen_cup_recovery task 가 failed 로 잡힌다.
    """
    if event.returncode != 0:
        raise RuntimeError(
            f"stand_fallen_cup 노드 비정상 종료(exit={event.returncode}) "
            "— recovery 실패를 launch 종료코드로 전파"
        )
    return None


def generate_launch_description():
    # Doosan M0609 MoveIt 기본 설정 (URDF, SRDF, kinematics, controllers 등)
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
        # .sensors_3d() 제거: 3D 센서(sensors_3d.yaml = sensors:[]) 미구성이라
        # octomap 모니터가 "No 3D sensor plugin(s)" ERROR 만 찍고 실제 octomap
        # 충돌회피는 안 함. stand_fallen_cup 은 octomap 미사용 → 노이즈 로그 제거.
        .to_moveit_configs()
    )

    # MoveItPy 전용 YAML
    moveit_py_params = PathJoinSubstitution(
        [FindPackageShare("dsr_practice"), "config", "moveit_py.yaml"]
    )

    dry_run = LaunchConfiguration("dry_run")
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
    multi_cup = LaunchConfiguration("multi_cup")
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
    sim = LaunchConfiguration("sim")
    sim_cup_x = LaunchConfiguration("sim_cup_x")
    sim_cup_y = LaunchConfiguration("sim_cup_y")
    sim_cup_z = LaunchConfiguration("sim_cup_z")
    sim_cup_yaw_deg = LaunchConfiguration("sim_cup_yaw_deg")
    robot_namespace = LaunchConfiguration("robot_namespace")
    joint_state_topic = LaunchConfiguration("joint_state_topic")
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

    stand_node = Node(
        package="dsr_practice",
        executable="stand_fallen_cup",
        output="screen",
        parameters=[
            moveit_config.to_dict(),
            moveit_py_params,
            {
                "dry_run": ParameterValue(dry_run, value_type=bool),
                "cup_yaw_override_deg": ParameterValue(
                    cup_yaw_override_deg, value_type=float
                ),
                "mode": ParameterValue(mode, value_type=str),
                "place_flange_side": ParameterValue(
                    place_flange_side, value_type=str
                ),
                "place_flange_yaw_deg": ParameterValue(
                    place_flange_yaw_deg, value_type=float
                ),
                "place_flange_yaw_auto_extra_deg": ParameterValue(
                    place_flange_yaw_auto_extra_deg, value_type=float
                ),
                "stand_cup_margin_m": ParameterValue(
                    stand_cup_margin_m, value_type=float
                ),
                "place_base_yaw_deg": ParameterValue(
                    place_base_yaw_deg, value_type=float
                ),
                "place_cup_tilt_deg": ParameterValue(
                    place_cup_tilt_deg, value_type=float
                ),
                "place_plus_y_auto_swing": ParameterValue(
                    place_plus_y_auto_swing, value_type=bool
                ),
                "place_plus_y_side": ParameterValue(
                    place_plus_y_side, value_type=str
                ),
                "place_plus_y_base_yaw_deg": ParameterValue(
                    place_plus_y_base_yaw_deg, value_type=float
                ),
                "place_plus_y_cup_tilt_deg": ParameterValue(
                    place_plus_y_cup_tilt_deg, value_type=float
                ),
                "place_x": ParameterValue(
                    place_x, value_type=float
                ),
                "place_y": ParameterValue(
                    place_y, value_type=float
                ),
                "multi_cup": ParameterValue(
                    multi_cup, value_type=bool
                ),
                "multi_cup_max_iterations": ParameterValue(
                    multi_cup_max_iterations, value_type=int
                ),
                "multi_cup_cluster_radius_m": ParameterValue(
                    multi_cup_cluster_radius_m, value_type=float
                ),
                "multi_cup_blacklist_radius_m": ParameterValue(
                    multi_cup_blacklist_radius_m, value_type=float
                ),
                "multi_cup_min_samples_per_cluster": ParameterValue(
                    multi_cup_min_samples_per_cluster, value_type=int
                ),
                "place_in_place": ParameterValue(
                    place_in_place, value_type=bool
                ),
                "place_spot_candidates": ParameterValue(
                    place_spot_candidates, value_type=str
                ),
                "place_spot_avoid_radius_m": ParameterValue(
                    place_spot_avoid_radius_m, value_type=float
                ),
                "upright_boxes_topic": ParameterValue(
                    upright_boxes_topic, value_type=str
                ),
                "sim": ParameterValue(sim, value_type=bool),
                "sim_cup_x": ParameterValue(sim_cup_x, value_type=float),
                "sim_cup_y": ParameterValue(sim_cup_y, value_type=float),
                "sim_cup_z": ParameterValue(sim_cup_z, value_type=float),
                "sim_cup_yaw_deg": ParameterValue(
                    sim_cup_yaw_deg, value_type=float
                ),
                "robot_namespace": ParameterValue(
                    robot_namespace, value_type=str
                ),
                # moveit_py.yaml의 절대경로 joint_state_topic을 override.
                # parameters 리스트에서 yaml(moveit_py_params)보다 뒤에
                # 오므로 우선 적용된다. 네임스페이스 bringup에서
                # planning_scene_monitor가 /<ns>/joint_states 를 구독해야
                # 현재 관절을 읽어 plan/IK 가 된다.
                "planning_scene_monitor_options.joint_state_topic": (
                    ParameterValue(joint_state_topic, value_type=str)
                ),
                "pyramid_avoid": ParameterValue(
                    pyramid_avoid, value_type=bool
                ),
                "pyramid_config_url": ParameterValue(
                    pyramid_config_url, value_type=str
                ),
                "pyramid_stack_topic": ParameterValue(
                    pyramid_stack_topic, value_type=str
                ),
                "pyramid_sync_poll_period_s": ParameterValue(
                    pyramid_sync_poll_period_s, value_type=float
                ),
                "pyramid_obstacle_margin_m": ParameterValue(
                    pyramid_obstacle_margin_m, value_type=float
                ),
                "pyramid_stack_wait_s": ParameterValue(
                    pyramid_stack_wait_s, value_type=float
                ),
                "avoid_upright_cups": ParameterValue(
                    avoid_upright_cups, value_type=bool
                ),
                "upright_obstacle_radius_m": ParameterValue(
                    upright_obstacle_radius_m, value_type=float
                ),
                "upright_obstacle_height_m": ParameterValue(
                    upright_obstacle_height_m, value_type=float
                ),
                "preflight_reach_check": ParameterValue(
                    preflight_reach_check, value_type=bool
                ),
            },
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "dry_run",
                default_value="false",
                description="True면 approach 자세까지만, gripper/descend/lift 스킵",
            ),
            DeclareLaunchArgument(
                "cup_yaw_override_deg",
                default_value="nan",
                description="NaN이 아니면 인식 yaw 무시하고 강제 값 사용",
            ),
            DeclareLaunchArgument(
                "mode",
                default_value="drop",
                description="lift 후 동작: 'drop' (3s 대기 후 release) / "
                            "'place' (작업공간으로 옮겨 세우기)",
            ),
            DeclareLaunchArgument(
                "place_flange_side",
                default_value="right",
                description="place 모드: flange를 PLACE의 어느 쪽에 둘지. "
                            "'right'=EE_Z→+Y (flange가 -Y 쪽, +Y obstacle 회피) / "
                            "'left'=EE_Z→-Y. place_flange_yaw_deg가 set이면 무시됨.",
            ),
            DeclareLaunchArgument(
                "place_flange_yaw_deg",
                default_value="nan",
                description="place 모드: standing 시 EE_Z 방향을 base XY 평면에서 "
                            "이 각도(deg)로 직접 강제. NaN(기본)이면 "
                            "place_flange_side에 따른 ±90° 사용. "
                            "권장 범위: side=right이면 [+90, +180].",
            ),
            DeclareLaunchArgument(
                "place_flange_yaw_auto_extra_deg",
                default_value="nan",
                description="place 모드: '문제 케이스(컵이 강제 방향 반대로 누움)' "
                            "자동 감지 시 baseline target_angle 에 sign*extra(deg) 추가. "
                            "NaN(기본)이면 비활성. 예: 45 → 90°→135°.",
            ),
            DeclareLaunchArgument(
                "stand_cup_margin_m",
                default_value="-0.05",
                description="place 모드: 컵 바닥-테이블 사이 여유 (m). "
                            "closing_z = TABLE_Z + CUP_HEIGHT + 이 값. "
                            "양수로 키우면 flange Z 같이 올라가 elbow 가 테이블에서 멀어짐. "
                            "기본 -0.05 (컵 바닥 가까이 release). 권장 +0.05~+0.10.",
            ),
            DeclareLaunchArgument(
                "place_base_yaw_deg",
                default_value="nan",
                description="place 모드: standing/retreat IK 시 joint_1을 이 값(deg)으로 "
                            "강제 (seed override). 로봇 전체 yaw를 회전시켜 elbow를 "
                            "workspace 밖으로 swing. NaN(기본)=미사용. "
                            "권장 시작값: ±30~±60° (방향은 obstacle 반대 쪽).",
            ),
            DeclareLaunchArgument(
                "place_cup_tilt_deg",
                default_value="0.0",
                description="place 모드: cup을 vertical에서 -EE_Z 방향으로 α° 기울여서 "
                            "release. flange Z가 sin α × TOOL_LENGTH 만큼 상승 → "
                            "elbow 자연 회피. α=20°에서 약 +68mm. "
                            "기본 0 (수직). 권장 10~20°. 30°+ 면 cup 넘어질 위험.",
            ),
            DeclareLaunchArgument(
                "place_plus_y_auto_swing",
                default_value="true",
                description="place 모드: cup wide가 +Y 영역으로 누운 케이스를 자동 감지 "
                            "(sin(cup_yaw) > 0.5) 하여 swing strategy 적용. "
                            "감지 시 side/base_yaw/cup_tilt를 plus_y_* 값으로 일괄 override. "
                            "true(기본)면 항상 켜짐 — +Y 미감지 시 no-op이라 다른 컵엔 영향 없음. "
                            "끄려면 :=false. 한 명령으로 ±Y 양쪽 케이스 처리.",
            ),
            DeclareLaunchArgument(
                "place_plus_y_side",
                default_value="left",
                description="auto_swing 발동 시 사용할 place_flange_side. "
                            "기본 'left' (사용자 검증값).",
            ),
            DeclareLaunchArgument(
                "place_plus_y_base_yaw_deg",
                default_value="60.0",
                description="auto_swing 발동 시 사용할 place_base_yaw_deg (deg). "
                            "기본 +60 (사용자 검증값).",
            ),
            DeclareLaunchArgument(
                "place_plus_y_cup_tilt_deg",
                default_value="25.0",
                description="auto_swing 발동 시 사용할 place_cup_tilt_deg (deg). "
                            "기본 25 (사용자 검증값).",
            ),
            DeclareLaunchArgument(
                "place_x",
                default_value="nan",
                description="single-cup PLACE x 좌표 override (base_link, m). "
                            "nan(기본)이면 모듈 상수 PLACE_X 사용.",
            ),
            DeclareLaunchArgument(
                "place_y",
                default_value="nan",
                description="single-cup PLACE y 좌표 override (base_link, m). "
                            "nan(기본)이면 모듈 상수 PLACE_Y 사용.",
            ),
            DeclareLaunchArgument(
                "multi_cup",
                default_value="false",
                description="한 프레임에 여러 fallen cup 이 있을 때 가까운 순서로 "
                            "순차 처리 (in-place stand). false(기본)면 single-cup 동작.",
            ),
            DeclareLaunchArgument(
                "multi_cup_max_iterations",
                default_value="10",
                description="multi_cup 모드 안전 limit. 이 횟수 도달하면 cup 남아도 종료.",
            ),
            DeclareLaunchArgument(
                "multi_cup_cluster_radius_m",
                default_value="0.04",
                description="multi_cup 클러스터링: 카메라 frame 에서 같은 cup 으로 묶을 "
                            "거리 임계값(m). 기본 0.04m=4cm.",
            ),
            DeclareLaunchArgument(
                "multi_cup_blacklist_radius_m",
                default_value="0.06",
                description="multi_cup blacklist: PLACE 위치에서 이 반지름(m) 내 신규 "
                            "감지된 cup 은 '이미 세움'으로 간주. 기본 0.06m=6cm.",
            ),
            DeclareLaunchArgument(
                "multi_cup_min_samples_per_cluster",
                default_value="3",
                description="cluster 최소 sample 수. 이 미만이면 noise 로 무시.",
            ),
            DeclareLaunchArgument(
                "place_in_place",
                default_value="false",
                description="multi_cup 에서 cup 을 제자리에 세움(legacy). false(기본)면 "
                            "작업영역 빈 안전지점에 세움.",
            ),
            DeclareLaunchArgument(
                "place_spot_candidates",
                default_value="0.30:0.10,0.30:0.00,0.30:-0.10",
                description="빈 안전지점 후보 'x:y,x:y,...' (base_link, 1순위부터). "
                            "기본 = 검증값 (0.30,0.10) + 좌우 2개.",
            ),
            DeclareLaunchArgument(
                "place_spot_avoid_radius_m",
                default_value="0.09",
                description="후보를 '점유'로 볼 회피 반경(m) — 정상 컵/기점유/피라미드 공통.",
            ),
            DeclareLaunchArgument(
                "upright_boxes_topic",
                default_value="/hand_eye/boxes",
                description="정상(세워진) 컵 위치 토픽 (upright_cup_pose_node 발행, "
                            "base_link MarkerArray).",
            ),
            DeclareLaunchArgument(
                "sim",
                default_value="false",
                description="True면 카메라/그리퍼 HW 우회 (MoveIt virtual용)",
            ),
            DeclareLaunchArgument(
                "sim_cup_x",
                default_value="0.28",
                description="sim 모드 가상 컵 base x (m). 넘어진-컵 작업영역(피라미드 정면 0.45,0 에서 옆·뒤로 빠짐)",
            ),
            DeclareLaunchArgument(
                "sim_cup_y",
                default_value="0.20",
                description="sim 모드 가상 컵 base y (m). +Y 작업영역",
            ),
            DeclareLaunchArgument(
                "sim_cup_z",
                default_value="0.10",
                description="sim 모드 가상 컵 base z (m)",
            ),
            DeclareLaunchArgument(
                "sim_cup_yaw_deg",
                default_value="0.0",
                description="sim 모드 가상 컵 yaw (deg)",
            ),
            DeclareLaunchArgument(
                "robot_namespace",
                default_value="",
                description="bringup이 네임스페이스(예: dsr01) 아래에서 도는 경우 "
                            "MoveItPy 내부 노드들을 같은 네임스페이스로 옮기기 위한 값. "
                            "dsr_bringup2_moveit.launch.py는 name 기본값이 ''(루트)이므로 "
                            "기본 ''(루트). namespaced bringup(name:=dsr01)이면 :=dsr01 로 준다.",
            ),
            DeclareLaunchArgument(
                "joint_state_topic",
                default_value="/joint_states",
                description="MoveItPy planning scene monitor가 구독할 joint state "
                            "토픽. moveit_py.yaml의 절대 토픽 '/joint_states'는 "
                            "name_space remap으로 바뀌지 않으므로 직접 override 한다. "
                            "dsr_bringup2_moveit(name 기본 '')는 /joint_states(루트)이므로 "
                            "기본 /joint_states. 네임스페이스 bringup이면 :=/<ns>/joint_states.",
            ),
            DeclareLaunchArgument(
                "pyramid_avoid",
                default_value="true",
                description="쌓인 피라미드(/stack 점유 슬롯)를 MoveIt collision "
                            "object 로 등록해 recovery 궤적이 회피. center/degree 는 "
                            "API polling 으로 동기화. vision 스택/서버 미가용이면 "
                            "graceful no-op(장애물 0개, 동작 변화 없음). 끄려면 :=false.",
            ),
            DeclareLaunchArgument(
                "pyramid_config_url",
                default_value="https://yarr-api-31.simplyimg.com/api/robot/config/pyramid",
                description="GET 으로 center{x,y}+degree 를 받아오는 FastAPI 엔드포인트. "
                            "verifier 와 동일 소스. 빈 문자열이면 polling 비활성.",
            ),
            DeclareLaunchArgument(
                "pyramid_stack_topic",
                default_value="/stack",
                description="피라미드 슬롯 점유 토픽 (std_msgs/String JSON "
                            "{slot: color|null}). verifier_node 가 publish.",
            ),
            DeclareLaunchArgument(
                "pyramid_sync_poll_period_s",
                default_value="5.0",
                description="config(center/degree) API polling 주기(s).",
            ),
            DeclareLaunchArgument(
                "pyramid_obstacle_margin_m",
                default_value="0.02",
                description="장애물 박스 xy 인플레이션 여유(m). Pilz 는 회피 재계획을 "
                            "안 하고 OMPL 만 피하므로 약간의 margin 으로 안전 여유. 기본 2cm.",
            ),
            DeclareLaunchArgument(
                "pyramid_stack_wait_s",
                default_value="1.5",
                description="등록 시 /stack 첫 수신을 기다리는 시간(s). sticky 라 "
                            "vision 이 떠 있으면 즉시 도착. 미수신이면 장애물 스킵.",
            ),
            DeclareLaunchArgument(
                "avoid_upright_cups",
                default_value="true",
                description="정상(세워진) 컵(/hand_eye/boxes)을 실린더 collision "
                            "object 로 등록해 궤적이 회피. 매 sense 직후 갱신.",
            ),
            DeclareLaunchArgument(
                "upright_obstacle_radius_m",
                default_value="0.04",
                description="정상 컵 장애물 실린더 반경(m). 컵 반경+여유. 기본 4cm.",
            ),
            DeclareLaunchArgument(
                "upright_obstacle_height_m",
                default_value="0.12",
                description="정상 컵 장애물 실린더 높이(m). 컵 높이+여유, 테이블에서 "
                            "위로. 기본 12cm.",
            ),
            DeclareLaunchArgument(
                "preflight_reach_check",
                default_value="true",
                description="pick 전에 approach/descend IK 해를 미리 검사. 해가 없으면 "
                            "그 컵을 집지 않고 blacklist 처리 + /fallen_cup/unreachable "
                            "토픽으로 통보.",
            ),
            stand_node,
            # 노드가 non-zero 로 죽으면 launch 종료코드도 non-zero 가 되게 함
            # (서버 LaunchManager 가 task 를 failed 로 표시하도록).
            RegisterEventHandler(
                OnProcessExit(
                    target_action=stand_node,
                    on_exit=_fail_launch_on_nonzero_exit,
                )
            ),
        ]
    )
