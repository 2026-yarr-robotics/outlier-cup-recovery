from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder


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
        .sensors_3d()
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
    multi_cup = LaunchConfiguration("multi_cup")
    multi_cup_max_iterations = LaunchConfiguration("multi_cup_max_iterations")
    multi_cup_cluster_radius_m = LaunchConfiguration("multi_cup_cluster_radius_m")
    multi_cup_blacklist_radius_m = LaunchConfiguration("multi_cup_blacklist_radius_m")
    multi_cup_min_samples_per_cluster = LaunchConfiguration(
        "multi_cup_min_samples_per_cluster"
    )
    sim = LaunchConfiguration("sim")
    sim_cup_x = LaunchConfiguration("sim_cup_x")
    sim_cup_y = LaunchConfiguration("sim_cup_y")
    sim_cup_z = LaunchConfiguration("sim_cup_z")
    sim_cup_yaw_deg = LaunchConfiguration("sim_cup_yaw_deg")
    robot_namespace = LaunchConfiguration("robot_namespace")

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
                "sim",
                default_value="false",
                description="True면 카메라/그리퍼 HW 우회 (MoveIt virtual용)",
            ),
            DeclareLaunchArgument(
                "sim_cup_x",
                default_value="0.40",
                description="sim 모드 가상 컵 base x (m)",
            ),
            DeclareLaunchArgument(
                "sim_cup_y",
                default_value="0.0",
                description="sim 모드 가상 컵 base y (m)",
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
                default_value="dsr01",
                description="bringup이 네임스페이스(예: dsr01) 아래에서 도는 경우 "
                            "MoveItPy 내부 노드들을 같은 네임스페이스로 옮기기 위한 값. "
                            "bringup_real_31.sh(dsr_bringup2, name=dsr01)와 정합되도록 "
                            "기본 'dsr01'. 루트 네임스페이스 bringup이면 :=\"\" 로 비운다.",
            ),
            Node(
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
                    },
                ],
            )
        ]
    )
