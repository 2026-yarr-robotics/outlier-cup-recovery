#!/usr/bin/env python3
"""
stand_fallen_cup.py

YOLO 인식 노드(fallen_cup_pose_node)가 publish하는
  /fallen_cup/grasp_pose  (PoseStamped, camera optical frame)
  /fallen_cup/pose2d      (Float32MultiArray, image yaw 포함)
를 받아서 넘어진 컵의 윗부분을 옆에서 잡고 들어 올리는 노드.

옵션1: lift-only (그리퍼가 컵 윗부분을 옆에서 잡고 들어올리면
컵이 자연스럽게 매달려서 수직이 됨)

전제:
  - click_pick_two.py와 동일한 환경
  - T_gripper2camera.npy hand-eye calibration 완료
  - RG2 그리퍼, OnRobot quick changer 192.168.1.1:502
  - dsr_bringup2_moveit.launch.py가 별도 터미널에서 떠 있음
  - speed_stack_yolo_seg의 fallen_cup_pose.launch.py가 use_depth:=true로 떠 있음

컵 치수 (사용자 측정):
  - 아래 넓은 원 지름  ≈ 7.5 cm
  - 위  좁은 부분 지름 ≈ 5.0 cm
"""

import math
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray

from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy, PlanRequestParameters

from .onrobot import RG


# ─────────────────────────────────────────────────────────
#  설정 (click_pick_two.py와 동일한 환경 가정)
# ─────────────────────────────────────────────────────────
GROUP_NAME = "manipulator"
BASE_FRAME = "base_link"
EE_LINK    = "link_6"

HOME_JOINTS = {
    "joint_1": math.radians(0.0),
    "joint_2": math.radians(0.0),
    "joint_3": math.radians(90.0),
    "joint_4": math.radians(0.0),
    "joint_5": math.radians(90.0),
    "joint_6": math.radians(90.0),
}

# 안전 작업 영역 (m, base_link 기준) — click_pick_two와 동일
SAFE_X_MIN = 0.0
SAFE_Y_MIN = -0.30
SAFE_Y_MAX =  0.30
SAFE_Z_MIN =  0.05   # link_6 flange 기준 최저 안전 z
                     # (TOOL_LENGTH_M 보정 후 손가락 기준이 아닌 flange 기준)

# z 보정 — bz(=depth가 읽은 컵 표면)에서 flange 목표 z로 변환
#   flange_at_grip = bz - CUP_R_AT_GRIP + TOOL_LENGTH_M
TOOL_LENGTH_M = 0.20     # link_6 flange → 그리퍼 손가락 closing 평면
                         # (RG2 + 툴체인저). click_pick_two의 Z_OFFSET와 동일 개념.
CUP_R_AT_GRIP = 0.0225   # grip 지점 컵 반경. depth(컵 표면) → 컵 축 보정.
                         # grip이 narrow 끝에서 1.5cm 들어간 지점 → 반경 약 2.25cm
                         # (narrow 지름 4.5~5.0cm 가정, 살짝 안쪽이라 거의 동일).
                         # 컵 모양이 더 가파르게 좁아지면 이 값 키워야.

# 동작 파라미터 (모두 flange_at_grip 기준 상대값)
APPROACH_OFFSET = 0.10   # 컵 위 접근 높이 (m)
GRIP_Z_MARGIN   = 0.020  # 그립 지점에서 위로 띄울 마진 (m). 5mm → 20mm로 키워서
                         # 그리퍼가 컵을 누르며 트립되는 것 방지.
LIFT_Z          = 0.45   # 들어 올리는 최종 z (절대값, flange 기준)
LIFT_HOLD_SEC   = 1.0    # 들어 올린 채 매달림 안정화 대기 시간 (s)

# 컵 놓기 (drop / place 모드 공용)
PLACE_X       = 0.55     # 컵을 세울 위치 (base frame, m).
                         # ⚠ place 모드에서 그리퍼가 수평으로 향하므로 flange는
                         # PLACE에서 TOOL_LENGTH(0.20m) 만큼 -EE_Z 방향에 위치한다.
                         # 컵 방향(cup_dir_base)에 따라 EE_Z가 ±X 어느 쪽이든 향할 수
                         # 있으므로, PLACE_X가 너무 작으면 worst case에 flange가
                         # robot 베이스에 너무 가까워져 IK 실패.
                         # 안전 마진: PLACE_X >= 0.50 권장.
PLACE_Y       = 0.0      # 픽업 위치와 겹치지 않게. 정면이 가장 reach 여유 큼.
TABLE_Z       = 0.05     # 테이블 표면 z (base frame, m). 실측으로 조정.
CUP_HEIGHT    = 0.10     # 컵 높이 (m). 실측으로 조정.
DROP_HOLD_SEC = 3.0      # drop 모드에서 lift 후 release까지 대기 시간 (s)

# place 모드 (방법 2: 손목 pitch 90° 회전으로 세우기) 전용
STAND_PITCH_SIGN_OVERRIDE = None  # None=자동, +1/-1로 강제 (cup 거꾸로 서면 부호 바꿈)
STAND_CUP_MARGIN_M        = -0.02 # standing 시 컵 바닥과 테이블 사이 여유 (m).
                                  # 음수면 closing_z를 더 낮춰서 release. 컵이 튕기지
                                  # 않도록 바닥 가까이에서 놓고 싶을 때 음수 사용.
                                  # 단, TABLE_Z가 정확해야 안전. 너무 음수면 충돌.

# 그리퍼 (raw 단위: 1/10 mm)
GRIPPER_NAME     = "rg2"
TOOLCHARGER_IP   = "192.168.1.1"
TOOLCHARGER_PORT = 502

# 컵 위 좁은 부분 외경 5 cm → 50 raw=mm→500
# open: 외경보다 +20mm 정도 여유 (70mm)
# close: 외경보다 -5~-7mm 정도 압박 (43~45mm)
# 압박이 너무 세면 컵 변형. 너무 약하면 떨어짐.
GRIP_OPEN_WIDTH  = 700   # 70.0 mm
GRIP_CLOSE_WIDTH = 450   # 45.0 mm  ← 컵 두께/재질에 맞춰 미세조정
GRIP_FORCE       = 200   # 약 13 N (얇은 컵이므로 약하게)

# 그리퍼 부호 보정 (실험으로 결정)
# 그리퍼 두 손가락이 컵 긴축에 "수직"이 되도록 회전.
# 만약 dry-run에서 손가락이 컵 축과 평행이 되면 부호 뒤집어야 함.
YAW_OFFSET_DEG = 90.0   # joint_6=0° HOME 기준에서 RG2가 EE_X 따라 닫히는 결과가 되어
                        # 컵 축과 평행해지므로 +90 으로 90° 회전시켜 수직 grip 만든다.
                        # (이전 joint_6=90° HOME 에서는 0 이었음.)
                        # 손가락이 컵 축에 또 평행이면 -90 으로 부호 반대로.

# 인식 안정화
SAMPLE_COLLECT_SEC = 8.0   # grasp_pose 수집 대기 시간 (최대)
MIN_SAMPLES        = 3     # 최소 샘플 개수 (이 만큼 모이면 조기 종료)

DOWN_ORI = {"x": 0.0, "y": 1.0, "z": 0.0, "w": 0.0}  # EE -Z 방향

# Hand-eye calibration 잔여 오차 보정 (base frame, m).
# 측정으로 결정: dry_run 출력의 base p 와 실측의 차이.
#   code p - real p = (BASE_OFFSET_X, BASE_OFFSET_Y, BASE_OFFSET_Z)
# 코드에서 p_base[i] -= BASE_OFFSET_i 로 빼준다.
# 카메라 마운팅이 바뀌거나 calibration 재실행 시 다시 0으로 두고 측정.
BASE_OFFSET_X = 0.0   # joint_6=0° HOME 기준으로 재측정 후 결정 예정
BASE_OFFSET_Y = 0.0   # joint_6=0° HOME 기준으로 재측정 후 결정 예정
BASE_OFFSET_Z = 0.080 # Z 오차는 joint_6 무관하게 +8cm 일정 (그리퍼 z 보정)


# ─────────────────────────────────────────────────────────
#  유틸
# ─────────────────────────────────────────────────────────
def clamp_to_safe_workspace(x, y, z, logger):
    if x < SAFE_X_MIN:
        logger.warning(f"x={x:.3f} clamped to {SAFE_X_MIN}")
        x = SAFE_X_MIN
    if y < SAFE_Y_MIN:
        logger.warning(f"y={y:.3f} clamped to {SAFE_Y_MIN}")
        y = SAFE_Y_MIN
    elif y > SAFE_Y_MAX:
        logger.warning(f"y={y:.3f} clamped to {SAFE_Y_MAX}")
        y = SAFE_Y_MAX
    if z < SAFE_Z_MIN:
        logger.warning(f"z={z:.3f} clamped to {SAFE_Z_MIN}")
        z = SAFE_Z_MIN
    return x, y, z


def plan_and_execute(robot, arm, logger, pose_goal=None,
                     state_goal=None, params=None):
    arm.set_start_state_to_current_state()

    if pose_goal is not None:
        x = pose_goal.pose.position.x
        y = pose_goal.pose.position.y
        z = pose_goal.pose.position.z
        sx, sy, sz = clamp_to_safe_workspace(x, y, z, logger)
        pose_goal.pose.position.x = sx
        pose_goal.pose.position.y = sy
        pose_goal.pose.position.z = sz
        arm.set_goal_state(pose_stamped_msg=pose_goal, pose_link=EE_LINK)
    elif state_goal is not None:
        arm.set_goal_state(robot_state=state_goal)
    else:
        logger.error("pose/state 없음")
        return False

    plan_result = (arm.plan(parameters=params)
                   if params is not None else arm.plan())
    if not plan_result:
        logger.error("Planning 실패")
        return False

    robot.execute(group_name=GROUP_NAME,
                  robot_trajectory=plan_result.trajectory,
                  blocking=True)
    return True


def make_pose(x, y, z, ori):
    p = PoseStamped()
    p.header.frame_id = BASE_FRAME
    p.pose.position.x = float(x)
    p.pose.position.y = float(y)
    p.pose.position.z = float(z)
    p.pose.orientation.x = ori["x"]
    p.pose.orientation.y = ori["y"]
    p.pose.orientation.z = ori["z"]
    p.pose.orientation.w = ori["w"]
    return p


def get_ee_matrix(moveit_robot):
    psm = moveit_robot.get_planning_scene_monitor()
    with psm.read_only() as scene:
        T = scene.current_state.get_global_link_transform(EE_LINK)
    return np.asarray(T, dtype=float)


def rotmat_to_quat_xyzw(R):
    """3x3 회전행렬 → quaternion (x, y, z, w). scipy 대체."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return float(x), float(y), float(z), float(w)


# DOWN_ORI 쿼터니언 (0,1,0,0) = Y축 180° 회전 → 회전행렬로 미리 캐시
R_DOWN = np.array([
    [-1.0, 0.0,  0.0],
    [ 0.0, 1.0,  0.0],
    [ 0.0, 0.0, -1.0],
])


# ─────────────────────────────────────────────────────────
#  Circular statistics (yaw 안정화용)
# ─────────────────────────────────────────────────────────
def circular_mean(angles):
    c = sum(math.cos(a) for a in angles) / len(angles)
    s = sum(math.sin(a) for a in angles) / len(angles)
    return math.atan2(s, c)


def circular_R(angles):
    # |mean vector|. 1.0 = 완전 일치, 0.0 = 완전 분산
    c = sum(math.cos(a) for a in angles) / len(angles)
    s = sum(math.sin(a) for a in angles) / len(angles)
    return math.hypot(c, s)


def angular_diff(a, b):
    d = a - b
    while d > math.pi:
        d -= 2.0 * math.pi
    while d < -math.pi:
        d += 2.0 * math.pi
    return d


# ─────────────────────────────────────────────────────────
#  Node
# ─────────────────────────────────────────────────────────
class StandFallenCupNode(Node):
    def __init__(self):
        super().__init__("stand_fallen_cup")
        log = self.get_logger()

        # ROS parameters (dry-run / yaw override 용)
        self.declare_parameter("dry_run", False)
        self.declare_parameter("cup_yaw_override_deg", float("nan"))
        self.declare_parameter("mode", "drop")
        # use_current_as_home: true면 launch 시점의 robot 상태를 세션 HOME으로 저장.
        #   티치펜던트에서 설정한 HOME 위치를 코드 수정 없이 그대로 사용 가능.
        #   초기 HOME 이동을 스킵하고 종료 시 그 자세로 정확히 복귀.
        # false면 코드에 박힌 HOME_JOINTS 사용 (기존 동작).
        self.declare_parameter("use_current_as_home", True)
        # sim: 카메라/그리퍼 하드웨어 없이 MoveIt virtual에서 동작 시각화
        self.declare_parameter("sim", False)
        self.declare_parameter("sim_cup_x", 0.40)
        self.declare_parameter("sim_cup_y", 0.00)
        self.declare_parameter("sim_cup_z", 0.10)
        self.declare_parameter("sim_cup_yaw_deg", 0.0)

        self.dry_run = bool(self.get_parameter("dry_run").value)
        self.use_current_as_home = bool(
            self.get_parameter("use_current_as_home").value
        )
        self.cup_yaw_override_deg = float(
            self.get_parameter("cup_yaw_override_deg").value
        )
        self.mode = str(self.get_parameter("mode").value).lower()
        if self.mode not in ("drop", "place"):
            log.warn(f"unknown mode '{self.mode}' → 'drop' 사용")
            self.mode = "drop"
        self.sim = bool(self.get_parameter("sim").value)
        self.sim_cup_x = float(self.get_parameter("sim_cup_x").value)
        self.sim_cup_y = float(self.get_parameter("sim_cup_y").value)
        self.sim_cup_z = float(self.get_parameter("sim_cup_z").value)
        self.sim_cup_yaw_deg = float(self.get_parameter("sim_cup_yaw_deg").value)

        log.info(f"=== POST-LIFT MODE: {self.mode} ===")
        if self.sim:
            log.warn(
                f"=== SIM MODE: 인식/그리퍼 우회. "
                f"cup=({self.sim_cup_x:.3f},{self.sim_cup_y:.3f},{self.sim_cup_z:.3f}), "
                f"yaw={self.sim_cup_yaw_deg:.1f}deg ==="
            )
        if self.dry_run:
            log.warn("=== DRY-RUN MODE: approach 자세까지만, gripper close/lift 없음 ===")
        if not math.isnan(self.cup_yaw_override_deg):
            log.warn(f"=== cup_yaw OVERRIDE: {self.cup_yaw_override_deg} deg (인식 무시) ===")

        # Hand-Eye
        calib_file = (
            Path(get_package_share_directory("dsr_practice"))
            / "config" / "T_gripper2camera.npy"
        )
        self.gripper2cam = np.load(str(calib_file)).astype(float)
        self.gripper2cam[:3, 3] /= 1000.0  # mm → m
        log.info(f"Hand-Eye 로드: {calib_file}")

        # 그리퍼 (sim 모드면 HW 연결 실패해도 무시)
        try:
            self.gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)
        except Exception as e:
            if self.sim:
                log.warn(f"[sim] gripper HW 연결 안 됨 (정상): {e}")
                self.gripper = None
            else:
                raise

        # MoveIt
        log.info("MoveItPy 초기화 중…")
        self.robot = MoveItPy(node_name="stand_fallen_cup_moveit_py")
        self.arm = self.robot.get_planning_component(GROUP_NAME)
        self.robot_model = self.robot.get_robot_model()
        log.info("MoveItPy 초기화 완료")

        # Plan 파라미터
        self.ompl_params = PlanRequestParameters(self.robot)
        self.ompl_params.planning_pipeline = "ompl"
        self.ompl_params.planner_id = "RRTConnect"
        self.ompl_params.max_velocity_scaling_factor = 0.2
        self.ompl_params.max_acceleration_scaling_factor = 0.1
        self.ompl_params.planning_time = 2.0

        self.pilz_params = PlanRequestParameters(self.robot)
        self.pilz_params.planning_pipeline = "pilz_industrial_motion_planner"
        self.pilz_params.planner_id = "PTP"
        self.pilz_params.max_velocity_scaling_factor = 0.10
        self.pilz_params.max_acceleration_scaling_factor = 0.05
        self.pilz_params.planning_time = 2.0

        # LIN: orientation 유지 + Cartesian 직선 (descend / lift 용)
        self.lin_params = PlanRequestParameters(self.robot)
        self.lin_params.planning_pipeline = "pilz_industrial_motion_planner"
        self.lin_params.planner_id = "LIN"
        self.lin_params.max_velocity_scaling_factor = 0.05
        self.lin_params.max_acceleration_scaling_factor = 0.03
        self.lin_params.planning_time = 2.0

        # 인식 결과 버퍼
        self.grasp_samples = []    # list of PoseStamped
        self.last_pose2d = None    # Float32MultiArray.data (latest)
        self.pose2d_samples = []   # yaw 샘플 (circular mean 용)

        self.create_subscription(
            PoseStamped, "/fallen_cup/grasp_pose",
            self._grasp_cb, 10)
        self.create_subscription(
            Float32MultiArray, "/fallen_cup/pose2d",
            self._pose2d_cb, 10)

        # sim 모드: 가상 컵 marker
        self.cup_marker_pub = None
        self._cup_state = "initial"       # initial | attached | placed | removed
        self._cup_axis_ee_sign = +1       # attached 상태에서 cup이 ±EE_X 어느 쪽인지
        self._cup_place_xy = (PLACE_X, PLACE_Y)
        if self.sim:
            self.cup_marker_pub = self.create_publisher(
                MarkerArray, "/sim_cup", 1
            )
            self.create_timer(0.3, self._publish_cup_marker)

    def _publish_cup_marker(self):
        """sim 모드에서 컵 시각화. 상태(initial/attached/placed)에 따라 자세 갱신."""
        if not self.sim or self.cup_marker_pub is None:
            return
        if self._cup_state == "removed":
            # delete marker
            m = Marker()
            m.header.frame_id = BASE_FRAME
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "sim_cup"
            m.id = 0
            m.action = Marker.DELETE
            ma = MarkerArray()
            ma.markers.append(m)
            self.cup_marker_pub.publish(ma)
            return

        m = Marker()
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "sim_cup"
        m.id = 0
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        m.scale.x = 0.060    # diameter (avg)
        m.scale.y = 0.060
        m.scale.z = CUP_HEIGHT
        m.color.r = 0.2
        m.color.g = 0.6
        m.color.b = 1.0
        m.color.a = 0.7

        if self._cup_state == "attached":
            # link_6 좌표계, closing plane이 +Z 쪽 TOOL_LENGTH_M 떨어져 있음.
            # cup은 그 closing plane에서 ±EE_X 방향으로 매달림.
            sign = self._cup_axis_ee_sign
            m.header.frame_id = EE_LINK
            m.pose.position.x = sign * CUP_HEIGHT / 2.0
            m.pose.position.y = 0.0
            m.pose.position.z = TOOL_LENGTH_M
            # cylinder local +Z → ±EE_X 방향 (90° 회전 around ±Y)
            m.pose.orientation.x = 0.0
            m.pose.orientation.y = sign * 0.7071
            m.pose.orientation.z = 0.0
            m.pose.orientation.w = 0.7071
        elif self._cup_state == "placed":
            # 세워서 놓은 상태: base frame, 수직
            px, py = self._cup_place_xy
            m.header.frame_id = BASE_FRAME
            m.pose.position.x = px
            m.pose.position.y = py
            m.pose.position.z = TABLE_Z + CUP_HEIGHT / 2.0
            m.pose.orientation.w = 1.0  # cylinder +Z = base +Z = up ✓
        else:  # "initial"
            # 옆으로 누워 있음. base frame.
            yaw = math.radians(self.sim_cup_yaw_deg)
            # 그립 위치는 narrow 끝, 컵은 yaw 방향으로 length만큼 뻗음
            # cup center = grip + (length/2) * (cos yaw, sin yaw, 0)
            m.header.frame_id = BASE_FRAME
            m.pose.position.x = (
                self.sim_cup_x + 0.5 * CUP_HEIGHT * math.cos(yaw)
            )
            m.pose.position.y = (
                self.sim_cup_y + 0.5 * CUP_HEIGHT * math.sin(yaw)
            )
            m.pose.position.z = self.sim_cup_z
            # cylinder local +Z → (cos yaw, sin yaw, 0)
            # 90° rotation around axis (-sin yaw, cos yaw, 0)
            m.pose.orientation.x = -0.7071 * math.sin(yaw)
            m.pose.orientation.y =  0.7071 * math.cos(yaw)
            m.pose.orientation.z = 0.0
            m.pose.orientation.w = 0.7071

        ma = MarkerArray()
        ma.markers.append(m)
        self.cup_marker_pub.publish(ma)

    # ── 콜백 ─────────────────────────────
    def _grasp_cb(self, msg: PoseStamped):
        if len(self.grasp_samples) == 0:
            self.get_logger().info(
                f"[recv] first /fallen_cup/grasp_pose: "
                f"frame={msg.header.frame_id} "
                f"p=({msg.pose.position.x:.3f},{msg.pose.position.y:.3f},"
                f"{msg.pose.position.z:.3f})"
            )
        self.grasp_samples.append(msg)
        if len(self.grasp_samples) > 30:
            self.grasp_samples.pop(0)

    def _pose2d_cb(self, msg: Float32MultiArray):
        if self.last_pose2d is None:
            self.get_logger().info("[recv] first /fallen_cup/pose2d")
        self.last_pose2d = list(msg.data)
        if len(msg.data) >= 7:
            self.pose2d_samples.append(float(msg.data[6]))
            if len(self.pose2d_samples) > 30:
                self.pose2d_samples.pop(0)

    # ── 좌표 변환 ─────────────────────────
    def compute_target(self):
        """
        return: (p_base[3], cup_yaw_base[rad])  or  None
        """
        log = self.get_logger()
        if len(self.grasp_samples) < MIN_SAMPLES:
            log.error(f"grasp_pose 샘플 부족: {len(self.grasp_samples)} < {MIN_SAMPLES}")
            return None
        if self.last_pose2d is None or len(self.last_pose2d) < 7:
            log.error("pose2d 미수신 — yaw 사용 불가")
            return None

        # 위치 평균 (camera optical frame)
        ps = np.array([
            [s.pose.position.x, s.pose.position.y, s.pose.position.z]
            for s in self.grasp_samples
        ])
        p_cam = ps.mean(axis=0)

        # T_base_cam = T_base_ee @ T_ee_cam  ← 현재 EE pose 기준
        T_base_ee = get_ee_matrix(self.robot)
        T_ee_cam  = self.gripper2cam
        T_base_cam = T_base_ee @ T_ee_cam

        p_base = (T_base_cam @ np.append(p_cam, 1.0))[:3]
        # Y-flip은 wrist joint(joint_6) 위치에 따라 달라지는 보정이라 비활성화.
        # 새 calibration이 link_6 기준으로 잘 풀려있다면 raw transform만으로 충분.
        # p_base[1] = -p_base[1]

        # 잔여 오차 상수 보정 (재측정으로 결정).
        p_base[0] -= BASE_OFFSET_X
        p_base[1] -= BASE_OFFSET_Y
        p_base[2] -= BASE_OFFSET_Z

        # 컵 축 방향: pose2d 샘플들에서 circular mean + outlier rejection
        yaws = list(self.pose2d_samples)
        if not yaws:
            log.error("pose2d yaw 샘플 없음 — 사용 불가")
            return None
        m1 = circular_mean(yaws)
        thresh = math.radians(30.0)
        yaws_in = [y for y in yaws if abs(angular_diff(y, m1)) <= thresh]
        n_out = len(yaws) - len(yaws_in)
        if not yaws_in:
            log.error("yaw inlier 0개 — 인식 매우 불안정")
            return None
        cam_yaw = circular_mean(yaws_in)
        R = circular_R(yaws_in)
        log.info(
            f"[yaw] n={len(yaws)} (outlier {n_out}) R={R:.3f} "
            f"mean={math.degrees(cam_yaw):.1f}deg"
        )
        if R < 0.9:
            log.warn(
                f"[yaw] 분산 큼 (R={R:.3f}) — 인식 흔들리는 중, "
                "결과 신뢰도 낮음"
            )

        v_cam  = np.array([math.cos(cam_yaw), math.sin(cam_yaw), 0.0])
        v_base = T_base_cam[:3, :3] @ v_cam
        # 위치와 동일하게 Y-flip 비활성 (joint_6 의존성 회피).
        # v_base[1] = -v_base[1]
        cup_yaw_base = math.atan2(v_base[1], v_base[0])

        log.info(
            f"camera p=({p_cam[0]:.3f},{p_cam[1]:.3f},{p_cam[2]:.3f}) "
            f"yaw={math.degrees(cam_yaw):.1f}deg"
        )
        log.info(
            f"base   p=({p_base[0]:.3f},{p_base[1]:.3f},{p_base[2]:.3f}) "
            f"cup_yaw={math.degrees(cup_yaw_base):.1f}deg"
        )
        return p_base, cup_yaw_base

    def grip_orientation(self, cup_yaw_base):
        """
        EE는 -Z 방향(아래)을 보고, base Z축 기준으로 grip_yaw만큼 회전.
        그리퍼 두 손가락은 컵 긴축에 수직이 되도록 +90도(또는 -90도) 보정.

        ⚠️ wrist-flip 방지: 평행턱 그리퍼는 180° 회전해도 동일한 grip이므로,
        grip_yaw를 [-π/2, π/2] 로 wrap 해서 joint 6이 HOME 근처에 머물도록 함.
        이렇게 하면 IK가 항상 동일한 branch를 선택 → 경로 도중 wrist 회전 X.
        """
        grip_yaw = cup_yaw_base + math.radians(YAW_OFFSET_DEG)

        # [-π/2, π/2] 로 wrap (180° 회전은 평행턱 그리퍼에 대해 동등)
        wrapped = math.atan2(math.sin(grip_yaw), math.cos(grip_yaw))  # [-π, π]
        if wrapped > math.pi / 2:
            wrapped -= math.pi
        elif wrapped < -math.pi / 2:
            wrapped += math.pi
        if abs(wrapped - grip_yaw) > 1e-6:
            self.get_logger().info(
                f"[grip_yaw] {math.degrees(grip_yaw):.1f}° → "
                f"{math.degrees(wrapped):.1f}° (wrist-flip 방지)"
            )
        grip_yaw = wrapped

        c, s = math.cos(grip_yaw), math.sin(grip_yaw)
        R_yaw = np.array([
            [c, -s, 0.0],
            [s,  c, 0.0],
            [0.0, 0.0, 1.0],
        ])
        R = R_yaw @ R_DOWN
        qx, qy, qz, qw = rotmat_to_quat_xyzw(R)
        return {"x": qx, "y": qy, "z": qz, "w": qw}

    # ── 그리퍼 호출 (sim 모드면 skip) ─────
    def _gripper_move(self, width, force):
        if self.gripper is not None:
            self.gripper.move_gripper(width, force)
        else:
            self.get_logger().info(
                f"[sim] gripper.move_gripper({width}, {force}) skipped"
            )

    # ── 현재 joint 상태 읽기 ──────────────
    def _read_current_joints(self):
        """planning_scene_monitor에서 현재 joint positions를 읽어 dict로 반환."""
        psm = self.robot.get_planning_scene_monitor()
        with psm.read_only() as scene:
            joints = dict(scene.current_state.joint_positions)
        return joints

    # ── IK 잠금 헬퍼 ──────────────────────
    def ik_state_with_current_seed(self, pose_stamped, timeout=1.0):
        """
        현재 관절 상태를 seed로 IK를 풀어 RobotState를 만든다.
        목적: descend / lift에서 IK가 다른 branch를 골라 wrist가 회전하는 것 방지.
        반환: 성공 시 RobotState, 실패 시 None.
        """
        log = self.get_logger()
        psm = self.robot.get_planning_scene_monitor()
        with psm.read_only() as scene:
            current_joints = dict(scene.current_state.joint_positions)

        target_state = RobotState(self.robot_model)
        target_state.joint_positions = current_joints  # seed = 현재 관절
        target_state.update()

        ok = target_state.set_from_ik(
            GROUP_NAME,
            pose_stamped.pose,
            EE_LINK,
            timeout,
        )
        if not ok:
            log.error(
                f"IK 실패: pose=({pose_stamped.pose.position.x:.3f},"
                f"{pose_stamped.pose.position.y:.3f},"
                f"{pose_stamped.pose.position.z:.3f})"
            )
            return None
        target_state.update()
        return target_state

    # ── 메인 ──────────────────────────────
    def run(self):
        log = self.get_logger()

        # controller action server 연결 대기
        log.info("[Init] controller 연결 대기 3s")
        time.sleep(3.0)

        # 1) HOME 결정 + (필요 시) 이동
        if self.use_current_as_home:
            # 티치펜던트에서 설정한 현재 자세를 그대로 세션 HOME으로 채택.
            # 초기 HOME 이동 없이 현재 joint state를 저장만 한다.
            self._session_home_joints = self._read_current_joints()
            log.info(
                "[Init] use_current_as_home=true → 현재 자세를 세션 HOME으로 캡처 "
                "(초기 HOME 이동 스킵)"
            )
            for jn, jv in self._session_home_joints.items():
                log.info(f"  {jn} = {math.degrees(jv):+.2f}°")
        else:
            log.info("[Init] HOME 이동 (코드 HOME_JOINTS 사용)")
            home_state = RobotState(self.robot_model)
            home_state.joint_positions = HOME_JOINTS
            home_state.update()
            if not plan_and_execute(self.robot, self.arm, log,
                                    state_goal=home_state,
                                    params=self.ompl_params):
                log.error("HOME 이동 실패 — 종료")
                return
            self._session_home_joints = dict(HOME_JOINTS)

        # 시작 시점의 link_6 pose를 저장 (종료 시 동일한 자세로 복귀했는지 검증용)
        self._start_T_base_ee = get_ee_matrix(self.robot)
        sp = self._start_T_base_ee[:3, 3]
        log.info(
            f"[Init] HOME link_6 pose: "
            f"pos=({sp[0]:.3f},{sp[1]:.3f},{sp[2]:.3f})"
        )

        # 2) 그리퍼 열기
        self._gripper_move(GRIP_OPEN_WIDTH, GRIP_FORCE)
        time.sleep(1.0)

        if self.sim:
            # sim: 인식 우회, 파라미터로 받은 가상 컵 좌표 사용
            log.info(
                f"[sim] sense 스킵, cup_base=({self.sim_cup_x:.3f},"
                f"{self.sim_cup_y:.3f},{self.sim_cup_z:.3f}), "
                f"yaw={self.sim_cup_yaw_deg:.1f}deg"
            )
            p_base = np.array([self.sim_cup_x, self.sim_cup_y, self.sim_cup_z])
            cup_yaw = math.radians(self.sim_cup_yaw_deg)
        else:
            # 2.5) HOME settle 대기 + sense 직전 버퍼 초기화
            # (HOME 정착 진동/이전 transient 샘플 제거 → yaw 정확도 ↑)
            log.info("[Sense] HOME settle 대기 1s")
            time.sleep(1.0)
            self.grasp_samples.clear()
            self.pose2d_samples.clear()
            self.last_pose2d = None

            # 3) 인식 결과 수집 (최대 SAMPLE_COLLECT_SEC, MIN_SAMPLES 모이면 조기 종료)
            log.info(
                f"[Sense] /fallen_cup/grasp_pose 수집 시작 "
                f"(최대 {SAMPLE_COLLECT_SEC}s, 최소 {MIN_SAMPLES}개)"
            )
            t0 = time.time()
            last_status_log = 0.0
            while rclpy.ok() and time.time() - t0 < SAMPLE_COLLECT_SEC:
                rclpy.spin_once(self, timeout_sec=0.05)
                if len(self.grasp_samples) >= MIN_SAMPLES:
                    log.info(
                        f"[Sense] 샘플 {len(self.grasp_samples)}개 확보 "
                        f"({time.time() - t0:.1f}s) — 조기 종료"
                    )
                    break
                # 1초마다 진행상황 로그
                if time.time() - last_status_log > 1.0:
                    log.info(
                        f"[Sense] grasp_samples={len(self.grasp_samples)}, "
                        f"pose2d={'received' if self.last_pose2d else 'NONE'}"
                    )
                    last_status_log = time.time()

            if len(self.grasp_samples) == 0:
                log.error("=== /fallen_cup/grasp_pose 한 개도 못 받음 ===")
                log.error("  1) 인식 노드가 use_depth:=true 인가?")
                log.error("  2) ros2 topic hz /fallen_cup/grasp_pose 직접 확인")
                log.error("  3) 컵이 카메라 화각 안, 거리 30~80cm, depth가 잡히는 표면인가?")
                return

            target = self.compute_target()
            if target is None:
                log.error("target 계산 실패 — 종료")
                return
            p_base, cup_yaw = target

        # cup_yaw override (dry-run/정적 테스트용)
        if not math.isnan(self.cup_yaw_override_deg):
            cup_yaw = math.radians(self.cup_yaw_override_deg)
            log.warn(f"cup_yaw override 적용 → {self.cup_yaw_override_deg:.1f} deg")

        ori = self.grip_orientation(cup_yaw)
        bx, by, bz = float(p_base[0]), float(p_base[1]), float(p_base[2])

        # bz(=depth가 읽은 컵 표면 z) → flange(link_6) 목표 z 로 보정
        #   - depth 표면 → 컵 축: -CUP_R_AT_GRIP
        #   - 컵 축 → flange:    +TOOL_LENGTH_M
        flange_at_grip = bz - CUP_R_AT_GRIP + TOOL_LENGTH_M

        log.info(
            f"[Plan] grip=({bx:.3f},{by:.3f},{bz:.3f}) "
            f"flange_at_grip={flange_at_grip:.3f} "
            f"cup_yaw={math.degrees(cup_yaw):.1f}deg "
            f"yaw_offset={YAW_OFFSET_DEG}deg"
        )

        # dry-run에서는 더 높이 띄움 (안전)
        approach_z = flange_at_grip + (0.25 if self.dry_run else APPROACH_OFFSET)

        # 4) APPROACH (컵 위)
        log.info(f"[1] Approach @ z={approach_z:.3f}")
        if not plan_and_execute(
                self.robot, self.arm, log,
                pose_goal=make_pose(bx, by, approach_z, ori),
                params=self.pilz_params):
            return

        if self.dry_run:
            log.warn("=== DRY-RUN: approach 자세 도달. 30초 정지 ===")
            log.warn("  → 그리퍼 손가락 평면이 컵 긴축에 [수직]이면 OK")
            log.warn("  → [평행]이면 YAW_OFFSET_DEG 부호 반대로 (90→-90)")
            t0 = time.time()
            while rclpy.ok() and time.time() - t0 < 30.0:
                rclpy.spin_once(self, timeout_sec=0.1)
            log.info("=== DRY-RUN 종료 ===")
            return

        # 5) DESCEND — IK를 현재 관절 seed로 잠가서 wrist 회전 차단
        descend_z = flange_at_grip + GRIP_Z_MARGIN
        if descend_z < SAFE_Z_MIN:
            log.warn(f"descend_z={descend_z:.3f} → {SAFE_Z_MIN} (clamp)")
            descend_z = SAFE_Z_MIN
        descend_pose = make_pose(bx, by, descend_z, ori)
        descend_state = self.ik_state_with_current_seed(descend_pose)
        if descend_state is None:
            log.error("descend IK 실패 — 종료")
            return
        log.info(
            f"[2] Descend @ z={descend_z:.3f} "
            "(joint goal, orientation 잠금)"
        )
        if not plan_and_execute(
                self.robot, self.arm, log,
                state_goal=descend_state,
                params=self.pilz_params):
            return

        # 6) CLOSE
        log.info("[3] Gripper CLOSE")
        self._gripper_move(GRIP_CLOSE_WIDTH, GRIP_FORCE)
        time.sleep(1.0)

        # sim: cup을 그리퍼에 attach (link_6 따라 움직이게)
        if self.sim:
            cup_dir_base_ = np.array([math.cos(cup_yaw), math.sin(cup_yaw), 0.0])
            cy_, sy_ = math.cos(cup_yaw + math.radians(YAW_OFFSET_DEG)), \
                       math.sin(cup_yaw + math.radians(YAW_OFFSET_DEG))
            # wrap된 grip_yaw 사용
            gyw_ = cup_yaw + math.radians(YAW_OFFSET_DEG)
            _w_ = math.atan2(math.sin(gyw_), math.cos(gyw_))
            if _w_ > math.pi / 2: _w_ -= math.pi
            elif _w_ < -math.pi / 2: _w_ += math.pi
            cy_, sy_ = math.cos(_w_), math.sin(_w_)
            R_yaw_ = np.array([[cy_, -sy_, 0], [sy_, cy_, 0], [0, 0, 1]])
            EE_X_lift_ = (R_yaw_ @ R_DOWN)[:, 0]
            dot_ = float(np.dot(EE_X_lift_[:2], cup_dir_base_[:2]))
            self._cup_axis_ee_sign = +1 if dot_ > 0 else -1
            self._cup_state = "attached"
            log.info(f"[sim] cup attached (axis_ee_sign={self._cup_axis_ee_sign:+d})")

        # 7) LIFT — IK 현재 관절 seed로 잠금 (잡은 컵이 휘둘리지 않게)
        lift_pose = make_pose(bx, by, LIFT_Z, ori)
        lift_state = self.ik_state_with_current_seed(lift_pose)
        if lift_state is None:
            log.error("lift IK 실패 — 종료")
            return
        log.info(f"[4] Lift @ z={LIFT_Z:.3f} (joint goal, orientation 잠금)")
        if not plan_and_execute(
                self.robot, self.arm, log,
                state_goal=lift_state,
                params=self.pilz_params):
            return

        log.info(f"[Hold] {LIFT_HOLD_SEC}s 대기 (컵 매달림 안정화)")
        time.sleep(LIFT_HOLD_SEC)

        # 8) MODE 분기: drop (그자리 떨어뜨림) / place (작업공간에 세우기)
        if self.mode == "drop":
            log.info(f"[5] drop: {DROP_HOLD_SEC}s 대기 후 그리퍼 open")
            time.sleep(DROP_HOLD_SEC)
            self._gripper_move(GRIP_OPEN_WIDTH, GRIP_FORCE)
            if self.sim:
                self._cup_state = "removed"
            time.sleep(0.5)
            log.info("=== drop 완료 ===")

        elif self.mode == "place":
            # === 컵을 수직(wide 끝이 바닥)으로 세우기 ===
            # YAW_OFFSET_DEG 값에 무관하도록 cross-product 기반 axis-angle 회전 사용.
            # cup_dir_base (lift 시 컵 narrow→wide 방향) 를 base의 -Z 방향에 맞추는
            # 최소 회전 R_align 을 구하고, R_stand = R_align @ R_lift_mat.

            # (a) lift 자세의 회전행렬 R_lift = R_yaw(grip_yaw_wrapped) @ R_DOWN
            grip_yaw_w = cup_yaw + math.radians(YAW_OFFSET_DEG)
            _w = math.atan2(math.sin(grip_yaw_w), math.cos(grip_yaw_w))
            if _w > math.pi / 2:
                _w -= math.pi
            elif _w < -math.pi / 2:
                _w += math.pi
            grip_yaw_w = _w

            cup_dir_base = np.array([math.cos(cup_yaw), math.sin(cup_yaw), 0.0])
            cy_, sy_ = math.cos(grip_yaw_w), math.sin(grip_yaw_w)
            R_yaw_mat = np.array([
                [cy_, -sy_, 0.0],
                [sy_,  cy_, 0.0],
                [0.0,  0.0, 1.0],
            ])
            R_lift_mat = R_yaw_mat @ R_DOWN

            # (b) cup_dir_base → -base_Z (wide 끝이 바닥 향함) 로 매핑하는 최소 회전
            final_cup_dir = np.array([0.0, 0.0, -1.0])
            v_cross = np.cross(cup_dir_base, final_cup_dir)
            v_dot = float(np.dot(cup_dir_base, final_cup_dir))
            cross_norm = float(np.linalg.norm(v_cross))

            if cross_norm < 1e-6:
                if v_dot > 0.0:
                    R_align = np.eye(3)
                    align_angle_deg = 0.0
                else:
                    # 컵이 이미 위로 향함 (드문 케이스). base X축 기준 180° flip.
                    R_align = np.diag([1.0, -1.0, -1.0])
                    align_angle_deg = 180.0
            else:
                axis = v_cross / cross_norm
                angle = math.atan2(cross_norm, v_dot)
                align_angle_deg = math.degrees(angle)
                K = np.array([
                    [0.0, -axis[2], axis[1]],
                    [axis[2], 0.0, -axis[0]],
                    [-axis[1], axis[0], 0.0],
                ])
                R_align = (np.eye(3)
                           + math.sin(angle) * K
                           + (1.0 - math.cos(angle)) * (K @ K))

            R_pre_twist = R_align @ R_lift_mat

            # (b.5) IK reach 안정성: gripper EE_Z를 ±base_Y 방향(옆 접근)으로 강제.
            #       컵은 base -Z 방향(수직)이므로 base_Z 축 회전을 더해도 컵 자세 유지.
            #       gripper가 정면(+X)이 아닌 측면(±Y)에서 접근하면 wrist가
            #       singular 영역을 피할 수 있어 IK가 더 잘 풀린다.
            EE_Z_after = R_pre_twist[:, 2]
            ez_xy_norm = float(np.linalg.norm(EE_Z_after[:2]))
            if ez_xy_norm > 1e-6:
                cur_angle = math.atan2(EE_Z_after[1], EE_Z_after[0])
                # 가장 가까운 ±Y 방향 선택
                target_angle = math.pi / 2 if EE_Z_after[1] >= 0 else -math.pi / 2
                twist = target_angle - cur_angle
                twist = math.atan2(math.sin(twist), math.cos(twist))
                cz, sz = math.cos(twist), math.sin(twist)
                R_twist = np.array([
                    [cz, -sz, 0.0],
                    [sz,  cz, 0.0],
                    [0.0, 0.0, 1.0],
                ])
                R_stand = R_twist @ R_pre_twist
                twist_deg = math.degrees(twist)
            else:
                R_stand = R_pre_twist
                twist_deg = 0.0

            sqx, sqy, sqz, sqw = rotmat_to_quat_xyzw(R_stand)
            stand_ori = {"x": sqx, "y": sqy, "z": sqz, "w": sqw}
            log.info(
                f"[place] cup_dir_base=({cup_dir_base[0]:+.2f},"
                f"{cup_dir_base[1]:+.2f},{cup_dir_base[2]:+.2f}) → "
                f"align={align_angle_deg:.1f}° + Z-twist={twist_deg:+.1f}°"
            )

            # (d) standing flange 위치
            # closing plane을 (PLACE_X, PLACE_Y, TABLE_Z+CUP_HEIGHT+margin) 로
            # → flange = closing_plane - TOOL_LENGTH_M * EE_Z_in_base
            EE_Z_stand = R_stand[:, 2]
            closing_z  = TABLE_Z + CUP_HEIGHT + STAND_CUP_MARGIN_M
            stand_fx   = PLACE_X - TOOL_LENGTH_M * EE_Z_stand[0]
            stand_fy   = PLACE_Y - TOOL_LENGTH_M * EE_Z_stand[1]
            stand_fz   = closing_z - TOOL_LENGTH_M * EE_Z_stand[2]

            log.info(
                f"[5] place + pitch tilt: "
                f"closing=({PLACE_X:+.3f},{PLACE_Y:+.3f},{closing_z:+.3f}), "
                f"flange=({stand_fx:+.3f},{stand_fy:+.3f},{stand_fz:+.3f}), "
                f"EE_Z_base=({EE_Z_stand[0]:+.2f},{EE_Z_stand[1]:+.2f},{EE_Z_stand[2]:+.2f})"
            )

            # (e) lift → standing 한 번에 plan (IK seed 우선, 실패 시 pose goal)
            stand_pose  = make_pose(stand_fx, stand_fy, stand_fz, stand_ori)
            stand_state = self.ik_state_with_current_seed(stand_pose, timeout=2.0)
            if stand_state is not None:
                ok = plan_and_execute(self.robot, self.arm, log,
                                      state_goal=stand_state,
                                      params=self.pilz_params)
            else:
                log.warn("standing IK with seed 실패 → pose goal로 fallback")
                ok = plan_and_execute(self.robot, self.arm, log,
                                      pose_goal=stand_pose,
                                      params=self.pilz_params)
            if not ok:
                log.error("standing 동작 실패 — 종료")
                return

            # (f) release
            log.info("[6] 그리퍼 open (cup release at standing)")
            self._gripper_move(GRIP_OPEN_WIDTH, GRIP_FORCE)
            if self.sim:
                self._cup_place_xy = (PLACE_X, PLACE_Y)
                self._cup_state = "placed"
                log.info("[sim] cup placed (서있음)")
            time.sleep(1.0)

            # (g) retreat (위로 — 그리퍼 stand orientation 유지)
            log.info(f"[7] 후퇴 위로 @ z={LIFT_Z:.3f}")
            retreat_pose  = make_pose(stand_fx, stand_fy, LIFT_Z, stand_ori)
            retreat_state = self.ik_state_with_current_seed(retreat_pose, timeout=2.0)
            if retreat_state is not None:
                plan_and_execute(self.robot, self.arm, log,
                                 state_goal=retreat_state,
                                 params=self.pilz_params)
            else:
                plan_and_execute(self.robot, self.arm, log,
                                 pose_goal=retreat_pose,
                                 params=self.pilz_params)
            log.info("=== place (방법2: pitch tilt) 완료 ===")

        # 9) HOME 복귀 — 시작과 동일한 자세로 강제 복귀 (재시도 + 검증)
        log.info("[Final] HOME 복귀 (시작 자세로)")
        home_back = RobotState(self.robot_model)
        home_back.joint_positions = self._session_home_joints
        home_back.update()

        # 1차: OMPL (충돌 회피 강함, 자유로운 경로)
        home_ok = plan_and_execute(self.robot, self.arm, log,
                                   state_goal=home_back,
                                   params=self.ompl_params)

        # 2차 fallback: Pilz PTP (joint-space 직선, OMPL 실패해도 reachable이면 풀림)
        if not home_ok:
            log.warn("[Final] OMPL HOME 복귀 실패 — Pilz PTP로 재시도")
            home_ok = plan_and_execute(self.robot, self.arm, log,
                                       state_goal=home_back,
                                       params=self.pilz_params)

        # 그리퍼도 시작과 동일하게 — 열림 상태
        self._gripper_move(GRIP_OPEN_WIDTH, GRIP_FORCE)
        time.sleep(0.5)

        # 종료 자세와 시작 자세 비교 (link_6 pose 차이 확인)
        if hasattr(self, "_start_T_base_ee"):
            end_T_base_ee = get_ee_matrix(self.robot)
            dp = end_T_base_ee[:3, 3] - self._start_T_base_ee[:3, 3]
            dp_norm = float(np.linalg.norm(dp))
            # 회전 차이도 측정 (Frobenius norm)
            dR = end_T_base_ee[:3, :3] - self._start_T_base_ee[:3, :3]
            dR_norm = float(np.linalg.norm(dR))
            ep = end_T_base_ee[:3, 3]
            log.info(
                f"[Final] 종료 link_6 pos=({ep[0]:.3f},{ep[1]:.3f},{ep[2]:.3f}) "
                f"| Δpos={dp_norm*1000:.1f}mm, Δrot={dR_norm:.3f}"
            )
            if dp_norm < 0.005 and dR_norm < 0.01:
                log.info("[Final] ✓ 시작 자세와 종료 자세 일치")
            else:
                log.warn(
                    "[Final] ⚠ 시작/종료 자세 차이 큼 — HOME 복귀가 완전히 안 됐을 수 있음"
                )

        if not home_ok:
            log.error("[Final] HOME 복귀 모든 시도 실패 — 로봇 자세 수동 확인 필요")


def main(args=None):
    rclpy.init(args=args)
    node = StandFallenCupNode()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
