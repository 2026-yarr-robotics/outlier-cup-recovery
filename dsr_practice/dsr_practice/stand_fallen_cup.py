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

import json
import math
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory

from geometry_msgs.msg import Pose, PoseStamped, PoseArray
from std_msgs.msg import Float32MultiArray, String
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import CollisionObject, AttachedCollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene
from visualization_msgs.msg import Marker, MarkerArray

from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy, PlanRequestParameters

from .onrobot import RG


class TopicRG:
    """SimRG(토픽) 그리퍼 백엔드 — Isaac 디지털 트윈 경로.

    OnRobot Modbus 대신 cup_stack gripper_node(SimRG)와 동일한 토픽 계약을
    쓴다: /gripper/target_width 에 목표 폭(mm)을 발행하면 Isaac 의
    gripper_bridge 가 손가락을 애니메이션하고 sticky attach/release 를
    수행한다. move_gripper 의 width 단위는 OnRobot 계약(1/10 mm)을 그대로
    받아 mm 로 변환한다 (GRIP_OPEN_WIDTH=700 → 70.0 mm).
    """

    def __init__(self, node, topic: str = "/gripper/target_width") -> None:
        from std_msgs.msg import Float32
        self._pub = node.create_publisher(Float32, topic, 10)
        self._log = node.get_logger()

    def move_gripper(self, width_val, force_val=400):  # noqa: ARG002 — force는 sim에서 무의미
        from std_msgs.msg import Float32
        width_mm = float(width_val) / 10.0
        self._pub.publish(Float32(data=width_mm))
        self._log.info(f"[topic-gripper] target_width {width_mm:.1f} mm")


# ─────────────────────────────────────────────────────────
#  설정 (click_pick_two.py와 동일한 환경 가정)
# ─────────────────────────────────────────────────────────
GROUP_NAME = "manipulator"
BASE_FRAME = "base_link"
EE_LINK    = "link_6"

# 고정 원점(HOME) 자세. 2026-06-09 사용자가 실로봇으로 직접 만든 자세를
# /joint_states 에서 캡처한 라디안 원값. (참고 도: j1 -2.84, j2 -14.92,
# j3 88.48, j4 -2.96, j5 107.60, j6 86.61). fallen-cup / mouth-up-cup 두
# task 모두 이 값을 공유(place_mouth_up_cup 가 import).
HOME_JOINTS = {
    "joint_1": -0.049552422016859055,
    "joint_2": -0.26035377383232117,
    "joint_3": 1.5442062616348267,
    "joint_4": -0.05165897682309151,
    "joint_5": 1.8779057264328003,
    "joint_6": 1.5116537809371948,
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

# 그리퍼(RG2) 충돌 프록시 — planning 시 그리퍼가 차지하는 부피를 link_6 에
# AttachedCollisionObject(BOX)로 붙여 그리퍼-장애물(피라미드) 충돌까지 회피.
# link_6 좌표계 기준: +Z 가 플랜지 바깥(그리퍼 진행 방향). 박스는 (0,0,OFFZ) 중심.
# 실로봇/sim 공통으로 항상 켜는 안전 기능(그리퍼는 양쪽 다 물리적으로 존재).
GRIPPER_PROXY_SIZE  = (0.16, 0.10, 0.20)  # (x,y,z) m. x=손가락 벌어짐 폭, z=툴 길이
GRIPPER_PROXY_OFFZ  = 0.10                # 박스 중심 z (= 길이/2, 플랜지에서 시작)
GRIPPER_PROXY_TOUCH = ["link_6", "link_5", "link_4"]  # self-collision 무시 링크
GRIPPER_PROXY_ID    = "gripper_proxy"
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
HOME_RETURN_HIGH_Z = 0.30  # HOME 복귀 시 EE z 가 이 값보다 높으면 테이블 dip 위험이
                           # 없다고 보고 Pilz PTP(결정적·최단) 를 먼저 시도. retreat 후
                           # (z≈LIFT_Z) 복귀가 이 경로를 타서 OMPL 의 빙 도는 궤적을 회피.
RETREAT_LIFT_STEP_M = 0.06 # place 후 수직 상승 LIN 의 step 크기 (m). 짧을수록 특이점을
                           # 가로지를 확률↓(LIN 성공률↑) 이지만 step 수↑(느려짐).
HOME_PREREVERSE_MIN_Z = 0.25  # joint_1 pre-reverse(수평 호) 를 허용하는 최소 EE z.
                           # 이보다 낮으면 joint_1 회전이 테이블을 쓸어버리므로,
                           # 먼저 수직 상승을 시도하고 그래도 낮으면 pre-reverse 를 스킵.

# 컵 놓기 (drop / place 모드 공용)
PLACE_X       = 0.30     # 컵을 세울 위치 (base frame, m).
                         # 넘어진-컵 작업영역(+Y, base 가까움) 안에 세워서 피라미드
                         # 정면(0.45,0)을 피한다. 그리퍼는 수평이라 flange가 PLACE에서
                         # TOOL_LENGTH(0.20m) 만큼 -EE_Z 방향(여기선 ±Y)으로 offset 됨 →
                         # flange_y 가 SAFE_Y_MAX(0.30) 안에 들도록 PLACE_Y/접근측을 잡는다.
PLACE_Y       = 0.10     # +Y 작업영역. 피라미드(y≈±0.12)서 멀고 base 가까움.
                         # 0.10 검증값: +Y swing 시 flange_y=+0.281(SAFE_Y_MAX 0.30 내),
                         # -Y 대칭 swing 시 flange_y=-0.081 → 양 케이스 place+retreat 성공.
TABLE_Z       = 0.05     # 테이블 표면 z (base frame, m). 실측으로 조정.
CUP_HEIGHT    = 0.10     # 컵 높이 (m). 실측으로 조정.
DROP_HOLD_SEC = 3.0      # drop 모드에서 lift 후 release까지 대기 시간 (s)

# place 모드 (방법 2: 손목 pitch 90° 회전으로 세우기) 전용
STAND_PITCH_SIGN_OVERRIDE = None  # None=자동, +1/-1로 강제 (cup 거꾸로 서면 부호 바꿈)
STAND_CUP_MARGIN_M        = -0.05 # standing 시 컵 바닥과 테이블 사이 여유 (m).
                                  # 음수면 closing_z를 더 낮춰서 release. 컵이 튕기지
                                  # 않도록 바닥 가까이에서 놓고 싶을 때 음수 사용.
                                  # 단, TABLE_Z가 정확해야 안전. 너무 음수면 충돌.

# place 모드 standing pose의 base_Z twist 임계값.
# - twist 분리 실행(pre-twist) 트리거 (lift 위치에서 base_Z 회전을 먼저 분리)
# - "문제 케이스(컵이 강제 방향과 반대로 누움)" 자동 감지 기준
# 같은 값 사용해야 두 경로의 의미가 일관됨.
PRE_TWIST_THRESHOLD_DEG = 90.0

# place_plus_y_auto_swing 의 cup 방향 감지 임계값.
# sin(cup_yaw_base) > 이 값 → cup wide 가 +Y 영역으로 누움 → swing strategy 트리거.
#   0.50 = cup_yaw ∈ [+30°, +150°] (넉넉)
#   0.87 = cup_yaw ∈ [+60°, +120°] (엄격, 거의 ±Y 만)
# 기본 0.50: 살짝 sideways 도 +Y로 간주.
PLUS_Y_DETECT_SIN_THRESHOLD = 0.50

# ─────────────────────────────────────────────────────────
#  Pyramid obstacle geometry (MoveIt 충돌 회피용)
# ─────────────────────────────────────────────────────────
# 쌓인 피라미드를 MoveIt collision object 로 등록해 recovery 궤적이 피하게 한다.
# 기하는 vision verifier_node (2026-yarr-robotics/vision-node, cup_stacking_verify)
# 와 FastAPI RobotDomain pyramid 배치를 그대로 미러링한다:
#   PYRAMID_CUP_SPACING=0.078 (슬롯 가로 간격=박스 폭, 행 방향)
#   PYRAMID_LAYER_HEIGHT=0.093 (층 수직 피치)
#   레이아웃 [3,2,1] (L1=3, L2=2, L3=1, 총 6슬롯)
# center{x,y}+degree 는 GET /api/robot/config/pyramid 에서 런타임으로 받아온다
# (verifier 와 동일 소스). z 는 TABLE_Z 에 앵커해 바닥부터 보수적으로 세운다.
PYRAMID_CUP_SPACING = 0.078   # 슬롯 가로 간격(행 방향) = 박스 X 폭
PYRAMID_CUP_DEPTH   = 0.070   # 박스 Y 깊이 (행 수직)
PYRAMID_CUP_BOX_H   = 0.086   # 컵 한 개 박스 높이
PYRAMID_LAYER_PITCH = 0.093   # 층 수직 피치 (cup_h 0.086 + gap 0.007)
PYRAMID_LAYER_COUNTS = (3, 2, 1)
_PYRAMID_SUFFIX_BY_COUNT = {1: ("T",), 2: ("L", "R"), 3: ("L", "M", "R")}


def _build_pyramid_slot_map():
    """슬롯 키(L1_L..L3_T) → (layer_idx, pos_idx, layer_count). verifier 명명과 일치."""
    m = {}
    for layer, count in enumerate(PYRAMID_LAYER_COUNTS):
        for pos in range(count):
            suffix = _PYRAMID_SUFFIX_BY_COUNT[count][pos]
            m[f"L{layer + 1}_{suffix}"] = (layer, pos, count)
    return m


PYRAMID_SLOT_MAP = _build_pyramid_slot_map()

PYRAMID_CONFIG_URL_DEFAULT = (
    "https://yarr-api-31.simplyimg.com/api/robot/config/pyramid"
)

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
SAMPLE_COLLECT_SEC = 4.0   # grasp_pose 수집 대기 시간 (최대)
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
                     state_goal=None, params=None, clamp=True):
    arm.set_start_state_to_current_state()

    if pose_goal is not None:
        if clamp:
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
        # false(기본)면 코드에 박힌 고정 원점 HOME_JOINTS 로 시작 시 이동 + 복귀.
        #   2026-06-09: 우연한 launch 자세가 아니라 항상 고정 원점에서 시작하도록 기본 false.
        self.declare_parameter("use_current_as_home", False)
        # place 모드에서 flange가 PLACE의 어느 쪽으로 위치할지.
        # "right"  → flange가 PLACE의 -Y 쪽 (로봇이 오른쪽으로 눕음, +Y obstacle 회피)
        # "left"   → flange가 PLACE의 +Y 쪽 (왼쪽으로 눕음)
        # 컵 방향이 어떻든 강제로 한쪽 면에서 접근.
        # 컵 wide 끝이 강제 방향과 반대면 gripper가 cup axis 주변으로 180° 추가 회전됨
        # (cup은 회전축이므로 cup axis = vertical 유지, wide 끝 down 유지).
        self.declare_parameter("place_flange_side", "right")
        # place_flange_yaw_deg:
        #   NaN(기본) 이면 place_flange_side에 따른 ±90° 사용 (기존 동작).
        #   값이 있으면 standing 시 base_Z 기준 EE_Z 방향(deg)을 그 값으로 강제.
        #   예: +135 → EE_Z=(-√2/2, +√2/2, 0). flange가 +X 쪽으로 빠지면서 elbow 들림.
        # 컵이 안전하게 vertical로 서는 건 base_Z 축 twist이면 어떤 각도든 동일하지만,
        # +Y obstacle 회피 측면에서는 반드시 [+90°, +180°] (혹은 [-180°, -90°]) 범위 권장.
        # 그 외 범위는 forearm이 +Y로 다시 들어갈 수 있음 — 신중하게 사용.
        self.declare_parameter("place_flange_yaw_deg", float("nan"))
        # place_flange_yaw_auto_extra_deg:
        #   NaN(기본) 이면 자동 확장 비활성.
        #   값이 있고, provisional twist(=baseline target_angle 기준)가
        #   PRE_TWIST_THRESHOLD_DEG 를 넘는 "문제 케이스"가 감지되면
        #   target_angle 을 baseline + sign(baseline) * extra 로 자동 확장.
        #   예: side=right(baseline=+90°), extra=45 → target_angle=+135°.
        # cup wide가 강제 방향(side)과 반대로 누운 경우에만 발동 → 정상 케이스 영향 X.
        self.declare_parameter("place_flange_yaw_auto_extra_deg", float("nan"))
        # stand_cup_margin_m:
        #   standing 시 컵 바닥과 테이블 사이 여유 (m). closing_z 보정.
        #   기본값 = 모듈 상수 STAND_CUP_MARGIN_M(=-0.05) → 컵을 바닥 가까이 release.
        #   양수로 키우면 closing_z 가 그만큼 올라가 flange Z 가 같이 올라감 →
        #   팔 전체가 위로 올라가 elbow 가 테이블에서 멀어짐. 컵은 그 높이에서 drop.
        #   너무 크면 컵이 튕기므로 +0.05~+0.10 부터 시도 권장.
        self.declare_parameter("stand_cup_margin_m", STAND_CUP_MARGIN_M)
        # place_base_yaw_deg:
        #   NaN(기본) 이면 미사용. 값이 있으면 standing(+retreat) IK 시 joint_1을
        #   이 값(deg)으로 seed override → IK solver가 joint_1≈target branch로 수렴
        #   → 로봇 전체가 base 기준 yaw로 회전된 자세로 standing 수행.
        #   목적: upper arm + elbow를 workspace 밖으로 swing 시켜 elbow ↔ table 충돌 회피.
        #   주의:
        #     - 너무 큰 각(±90°+) 은 IK 실패 또는 다른 branch로 수렴할 수 있음.
        #     - +방향이면 elbow가 +Y 쪽, -방향이면 -Y 쪽. +Y obstacle 상황이면 음수 권장.
        #     - KDL IK는 seed 근사 수렴이라 항상 desired branch로 가지 않을 수 있음.
        #       로그의 final joint_1 값으로 적용 여부 확인.
        self.declare_parameter("place_base_yaw_deg", float("nan"))
        # place_cup_tilt_deg:
        #   standing 시 cup을 vertical 에서 -EE_Z 방향(그리퍼 반대 쪽)으로 α° 기울임.
        #   기본 0 = 수직. 값 있으면:
        #     - flange Z 가 sin α × TOOL_LENGTH_M ≈ α=20° 일 때 +68mm 상승
        #     - closing_z 는 cos α 보정으로 약간 하강 (CUP_HEIGHT × (1-cos α) ≈ 6mm)
        #     - 순효과: flange Z 약 +62mm (α=20°), trajectory 전체가 위로 들림 → elbow 회피.
        #   원리: cup 은 release 후 wide bottom self-righting (tip-over 한계 ≈ atan(R/h)≈37°).
        #         10~20° 는 안전 영역.
        #   주의: α 가 클수록 cup 이 더 기울어진 채로 release → 떨어지면서 한쪽 edge 가 먼저 접지.
        #         release 직후 잠시 진동 후 settle. 너무 크면 (≥30°) 완전히 넘어질 위험.
        self.declare_parameter("place_cup_tilt_deg", 0.0)
        # place_plus_y_auto_swing:
        #   true 면 cup wide 가 +Y 영역으로 누운 케이스(=sin(cup_yaw_base) > 임계값)
        #   를 자동 감지하여 아래 plus_y_* 파라미터들을 일괄 적용.
        #   - 감지 됨: side, base_yaw, tilt 가 plus_y_* 값으로 override.
        #   - 감지 안 됨: 사용자가 지정한(또는 기본) 파라미터 그대로 사용.
        #   한 명령으로 양쪽 케이스(cup wide ±Y) 모두 안전 처리하기 위함.
        self.declare_parameter("place_plus_y_auto_swing", True)
        # 아래 3개는 위 auto_swing 이 True 이고 +Y 감지된 경우에만 적용.
        # 기본값은 사용자 검증된 조합 (left / +60° / 25°).
        self.declare_parameter("place_plus_y_side", "left")
        self.declare_parameter("place_plus_y_base_yaw_deg", 60.0)
        self.declare_parameter("place_plus_y_cup_tilt_deg", 25.0)
        # place_x / place_y: single-cup PLACE 좌표 override (기본=모듈 상수 PLACE_X/Y).
        #   리빌드 없이 작업영역 안에서 세울 위치를 튜닝하기 위함 (NaN 이면 상수 사용).
        self.declare_parameter("place_x", float("nan"))
        self.declare_parameter("place_y", float("nan"))
        # multi_cup:
        #   true 면 한 프레임에 여러 넘어진 cup 이 있어도 모두 순차 처리.
        #   /fallen_cup/cups_pose2d, /fallen_cup/cups_grasp_poses (Phase 1) 토픽 사용.
        #   가까운 cup 부터 한 개씩 pick → in-place stand → HOME 복귀 → 재sense 반복.
        #   false (기본) 면 기존 single-cup 동작 (/fallen_cup/grasp_pose, /pose2d).
        self.declare_parameter("multi_cup", False)
        # 안전 / 클러스터링 파라미터
        self.declare_parameter("multi_cup_max_iterations", 10)
        self.declare_parameter("multi_cup_cluster_radius_m", 0.04)
        self.declare_parameter("multi_cup_blacklist_radius_m", 0.06)
        self.declare_parameter("multi_cup_min_samples_per_cluster", 3)
        # ── 빈 안전지점 세우기 (multi_cup 기본) ────────────────────────────
        #   multi_cup 동작을 "제자리 세우기"가 아니라 "작업영역 빈 안전지점 세우기"로.
        #   카메라가 보는 정상 컵(/hand_eye/boxes upright-cup) + 이미 세운 컵(blacklist)
        #   + 피라미드를 피해 후보 좌표 중 첫 빈 자리에 세운다. 빈 자리 없으면 그 cup skip.
        #   place_in_place:=true 면 기존 제자리 세우기로 회귀.
        self.declare_parameter("place_in_place", False)
        #   후보 PLACE 좌표 "x:y,x:y,..." (base_link, 1순위부터 검사). 기본 = 검증값
        #   (0.30,0.10) + 좌우 2개. 셋 다 +Y/-Y swing flange 가 ±SAFE_Y_MAX(0.30) 내.
        self.declare_parameter(
            "place_spot_candidates", "0.30:0.10,0.30:0.00,0.30:-0.10")
        #   후보를 "점유"로 볼 회피 반경(m) — upright-cup/blacklist/피라미드 공통.
        self.declare_parameter("place_spot_avoid_radius_m", 0.09)
        #   정상 컵 위치 토픽 (upright_cup_pose_node 발행, base_link MarkerArray).
        self.declare_parameter("upright_boxes_topic", "/hand_eye/boxes")
        # sim: 카메라/그리퍼 하드웨어 없이 MoveIt virtual에서 동작 시각화
        self.declare_parameter("sim", False)
        # gripper_backend: 그리퍼 경로를 sim 플래그와 분리 (Isaac 디지털
        # 트윈은 인식은 실제(카메라 발행)·그리퍼는 토픽이어야 한다):
        #   ''(기본)  — 레거시: sim ? none : onrobot
        #   'onrobot' — RG Modbus 직접 (실기)
        #   'topic'   — /gripper/target_width 발행 (Isaac SimRG 계약)
        #   'none'    — 그리퍼 명령 전부 no-op
        self.declare_parameter("gripper_backend", "")
        self.declare_parameter("sim_cup_x", 0.28)   # 넘어진-컵 작업영역(피라미드 정면 0.45,0 에서 옆·뒤로 빠짐)
        self.declare_parameter("sim_cup_y", 0.20)
        self.declare_parameter("sim_cup_z", 0.10)
        self.declare_parameter("sim_cup_yaw_deg", 0.0)
        # robot_namespace:
        #   bringup이 네임스페이스(예: /dsr01) 아래에서 도는 워크스페이스용.
        #   MoveItPy는 자체 rclcpp 컨텍스트를 새로 만들기 때문에 launch의
        #   namespace= / __ns remap이 전파되지 않는다. 값이 있으면 MoveItPy에
        #   name_space + __ns remap을 넘겨 내부 노드들(planning_scene_monitor,
        #   moveit_simple_controller_manager)이 모두 해당 네임스페이스 아래에서
        #   생성되게 한다. 그래야 (1) FollowJointTrajectory 액션 클라이언트가
        #   <ns>/dsr_moveit_controller 에 바인딩되고 (2) joint_states 구독이
        #   <ns>/joint_states 로 걸려 현재 관절을 읽는다. 빈 값이면 루트.
        #   기본 '': dsr_bringup2_moveit.launch.py 의 name 기본값이 ''(루트)와 정합.
        #   namespaced bringup(name:=dsr01)이면 robot_namespace:=dsr01 로 준다.
        self.declare_parameter("robot_namespace", "")

        # ── 피라미드 장애물 회피 ──────────────────────────────────────────
        # True 면 쌓인 피라미드(/stack 점유 슬롯)를 MoveIt collision object 로
        # 등록해 recovery 궤적이 회피. center/degree 는 API polling 으로 동기화.
        # vision 스택/서버가 없으면 graceful no-op (장애물 0개, 동작 변화 없음).
        self.declare_parameter("pyramid_avoid", True)
        self.declare_parameter("pyramid_config_url", PYRAMID_CONFIG_URL_DEFAULT)
        self.declare_parameter("pyramid_stack_topic", "/stack")
        self.declare_parameter("pyramid_sync_poll_period_s", 5.0)
        # 박스를 xy 로 인플레이션할 여유 (m). Pilz 는 회피 재계획을 안 하고
        # OMPL 만 피하므로 약간의 margin 으로 안전 여유 확보.
        self.declare_parameter("pyramid_obstacle_margin_m", 0.02)
        # /stack 수집 대기 (s). vision 이 떠 있으면 sticky 라 즉시 도착.
        self.declare_parameter("pyramid_stack_wait_s", 1.5)
        # 그리퍼 충돌 프록시 attach 여부 (실로봇/sim 공통, 기본 ON).
        # 끄면 planning 이 팔 링크만 보고 그리퍼 부피를 무시 → 피라미드 클립 위험.
        self.declare_parameter("attach_gripper_collision", True)

        # ── 정상(세워진) 컵 장애물 회피 ───────────────────────────────────
        # True 면 /hand_eye/boxes 의 정상 컵 위치마다 실린더 collision object 를
        # 등록해 recovery 궤적이 회피. multi_cup loop 는 매 sense 직후 갱신한다.
        # 토픽이 안 떠 있으면(정상 컵 0개) graceful no-op.
        self.declare_parameter("avoid_upright_cups", True)
        self.declare_parameter("upright_obstacle_radius_m", 0.04)  # 컵 반경+여유
        self.declare_parameter("upright_obstacle_height_m", 0.12)  # 컵 높이+여유
        # pick 전에 approach/descend IK 해가 있는지 미리 검사. 해가 없으면 그 컵을
        # 집지 않고 blacklist 처리 + /fallen_cup/unreachable 토픽으로 통보한다.
        self.declare_parameter("preflight_reach_check", True)

        self.dry_run = bool(self.get_parameter("dry_run").value)
        self.use_current_as_home = bool(
            self.get_parameter("use_current_as_home").value
        )
        self.place_flange_side = str(
            self.get_parameter("place_flange_side").value
        ).lower()
        if self.place_flange_side not in ("right", "left"):
            log.warn(
                f"unknown place_flange_side '{self.place_flange_side}' → 'right' 사용"
            )
            self.place_flange_side = "right"
        self.place_flange_yaw_deg = float(
            self.get_parameter("place_flange_yaw_deg").value
        )
        self.place_flange_yaw_auto_extra_deg = float(
            self.get_parameter("place_flange_yaw_auto_extra_deg").value
        )
        self.stand_cup_margin_m = float(
            self.get_parameter("stand_cup_margin_m").value
        )
        self.place_base_yaw_deg = float(
            self.get_parameter("place_base_yaw_deg").value
        )
        self.place_cup_tilt_deg = float(
            self.get_parameter("place_cup_tilt_deg").value
        )
        self.place_plus_y_auto_swing = bool(
            self.get_parameter("place_plus_y_auto_swing").value
        )
        self.place_plus_y_side = str(
            self.get_parameter("place_plus_y_side").value
        ).lower()
        if self.place_plus_y_side not in ("right", "left"):
            log.warn(
                f"unknown place_plus_y_side '{self.place_plus_y_side}' → 'left' 사용"
            )
            self.place_plus_y_side = "left"
        self.place_plus_y_base_yaw_deg = float(
            self.get_parameter("place_plus_y_base_yaw_deg").value
        )
        self.place_plus_y_cup_tilt_deg = float(
            self.get_parameter("place_plus_y_cup_tilt_deg").value
        )
        # multi-cup loop 에서 cup 별로 plus_y_auto_swing 판단을 새로 하려면
        # 매 cup 시작 시점에 이 세 값을 launch 시점 기본값으로 되돌려야 함.
        # 이전 cup 의 swing override 가 다음 cup 으로 누수돼 잘못된 side/yaw/tilt 가
        # 적용되는 버그 방지용 backup.
        self._default_place_flange_side = self.place_flange_side
        self._default_place_base_yaw_deg = self.place_base_yaw_deg
        self._default_place_cup_tilt_deg = self.place_cup_tilt_deg
        self.multi_cup = bool(self.get_parameter("multi_cup").value)
        self.multi_cup_max_iterations = int(
            self.get_parameter("multi_cup_max_iterations").value
        )
        self.multi_cup_cluster_radius_m = float(
            self.get_parameter("multi_cup_cluster_radius_m").value
        )
        self.multi_cup_blacklist_radius_m = float(
            self.get_parameter("multi_cup_blacklist_radius_m").value
        )
        self.multi_cup_min_samples_per_cluster = int(
            self.get_parameter("multi_cup_min_samples_per_cluster").value
        )
        self.place_in_place = bool(self.get_parameter("place_in_place").value)
        self.place_spot_avoid_radius_m = float(
            self.get_parameter("place_spot_avoid_radius_m").value
        )
        self.upright_boxes_topic = str(
            self.get_parameter("upright_boxes_topic").value
        )
        self.place_spot_candidates = self._parse_spot_candidates(
            str(self.get_parameter("place_spot_candidates").value)
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
        self.robot_namespace = str(
            self.get_parameter("robot_namespace").value or ""
        ).strip()
        self.pyramid_avoid = bool(self.get_parameter("pyramid_avoid").value)
        self.pyramid_config_url = str(
            self.get_parameter("pyramid_config_url").value or ""
        ).strip()
        self.pyramid_stack_topic = str(
            self.get_parameter("pyramid_stack_topic").value or "/stack"
        ).strip()
        self.pyramid_sync_poll_period_s = max(
            1.0, float(self.get_parameter("pyramid_sync_poll_period_s").value)
        )
        self.pyramid_obstacle_margin_m = float(
            self.get_parameter("pyramid_obstacle_margin_m").value
        )
        self.pyramid_stack_wait_s = float(
            self.get_parameter("pyramid_stack_wait_s").value
        )
        self.avoid_upright_cups = bool(
            self.get_parameter("avoid_upright_cups").value
        )
        self.upright_obstacle_radius_m = float(
            self.get_parameter("upright_obstacle_radius_m").value
        )
        self.upright_obstacle_height_m = float(
            self.get_parameter("upright_obstacle_height_m").value
        )
        self.preflight_reach_check = bool(
            self.get_parameter("preflight_reach_check").value
        )

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

        # 그리퍼 백엔드 선택 (sim 플래그와 직교 — Isaac 은 sim=false +
        # gripper_backend=topic 으로 인식은 실제, 그리퍼는 토픽 경로를 쓴다).
        # ''(기본) = 레거시 동작 유지: sim ? none : onrobot.
        # RG()는 lazy connect라 생성자에서 안 터지지만, none 이면 구성 자체를
        # 건너뛰어 _gripper_move 가 no-op 이 되게 한다.
        backend = str(self.get_parameter("gripper_backend").value).strip()
        if not backend:
            backend = "none" if self.sim else "onrobot"
        if backend == "topic":
            log.warn("[gripper] TOPIC 백엔드 — /gripper/target_width 발행 "
                     "(Isaac SimRG 계약)")
            self.gripper = TopicRG(self)
        elif backend == "none":
            log.warn("[gripper] 백엔드 none — 그리퍼 명령은 모두 skip 된다")
            self.gripper = None
        else:
            self.gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)

        # MoveIt
        # MoveItPy는 rclcpp::init(...)로 자체 컨텍스트를 만들고 launch가 넘긴
        # --ros-args(__ns 포함)를 무시한다. robot_namespace가 지정되면
        # name_space + __ns remap을 직접 넘겨 내부 노드들이 해당 네임스페이스
        # 아래에서 생성되게 한다 → FollowJointTrajectory 클라이언트가
        # <ns>/dsr_moveit_controller 에, joint_states 구독이 <ns>/joint_states 에 바인딩.
        log.info("MoveItPy 초기화 중…")
        ns = self.robot_namespace
        moveit_kwargs = {"node_name": "stand_fallen_cup_moveit_py"}
        if ns and ns != "/":
            moveit_kwargs["name_space"] = ns
            moveit_kwargs["remappings"] = {"__ns": ns}
            log.info(f"MoveItPy namespace: {ns}")
        self.robot = MoveItPy(**moveit_kwargs)
        self.arm = self.robot.get_planning_component(GROUP_NAME)
        self.robot_model = self.robot.get_robot_model()
        log.info("MoveItPy 초기화 완료")

        # Plan 파라미터
        # 속도 스케일: joint_limits.yaml 의 max 대비 비율 (0.0~1.0).
        # 너무 키우면 컵이 잡힌 상태에서 흔들리거나 RG2가 트립할 수 있음. 조심.
        self.ompl_params = PlanRequestParameters(self.robot)
        self.ompl_params.planning_pipeline = "ompl"
        self.ompl_params.planner_id = "RRTConnect"
        self.ompl_params.max_velocity_scaling_factor = 0.30      # 0.20 → 0.30
        self.ompl_params.max_acceleration_scaling_factor = 0.15  # 0.10 → 0.15
        self.ompl_params.planning_time = 2.0

        self.pilz_params = PlanRequestParameters(self.robot)
        self.pilz_params.planning_pipeline = "pilz_industrial_motion_planner"
        self.pilz_params.planner_id = "PTP"
        self.pilz_params.max_velocity_scaling_factor = 0.20      # 0.10 → 0.20 (≈2x)
        self.pilz_params.max_acceleration_scaling_factor = 0.10  # 0.05 → 0.10
        self.pilz_params.planning_time = 2.0

        # LIN: orientation 유지 + Cartesian 직선 (descend / lift 용)
        self.lin_params = PlanRequestParameters(self.robot)
        self.lin_params.planning_pipeline = "pilz_industrial_motion_planner"
        self.lin_params.planner_id = "LIN"
        self.lin_params.max_velocity_scaling_factor = 0.10       # 0.05 → 0.10
        self.lin_params.max_acceleration_scaling_factor = 0.06   # 0.03 → 0.06
        self.lin_params.planning_time = 2.0

        # 인식 결과 버퍼 (single-cup)
        self.grasp_samples = []    # list of PoseStamped
        self.last_pose2d = None    # Float32MultiArray.data (latest)
        self.pose2d_samples = []   # yaw 샘플 (circular mean 용)

        # 인식 결과 버퍼 (multi-cup, Phase 2)
        self._latest_cups_pose2d = None     # list of dict per cup_id (yaw, conf 등)
        self._cups_frame_samples = []       # list of frame dicts with cups list

        # 정상(세워진) 컵 위치 스냅샷 (base_link XY). /hand_eye/boxes 최신 프레임.
        # 빈 안전지점 선택 + 궤적 회피(실린더 collision object) 둘 다에 쓴다.
        self._upright_cups = []             # list of (x, y)
        if self.avoid_upright_cups:
            self._upright_co_pub = self.create_publisher(
                CollisionObject, "/collision_object", 10)

        # 활성 PLACE 좌표 (in-place stand 위해 multi-cup loop 가 override).
        # 기본값은 모듈 상수 — single-cup 동작 변경 없음. place_x/y 파라미터로 override 가능.
        _px = self.get_parameter("place_x").value
        _py = self.get_parameter("place_y").value
        self._active_place_x = PLACE_X if math.isnan(_px) else float(_px)
        self._active_place_y = PLACE_Y if math.isnan(_py) else float(_py)

        self.create_subscription(
            PoseStamped, "/fallen_cup/grasp_pose",
            self._grasp_cb, 10)
        self.create_subscription(
            Float32MultiArray, "/fallen_cup/pose2d",
            self._pose2d_cb, 10)
        # Multi-cup 토픽 (Phase 1 에서 pose node 가 publish)
        self.create_subscription(
            Float32MultiArray, "/fallen_cup/cups_pose2d",
            self._cups_pose2d_cb, 10)
        self.create_subscription(
            PoseArray, "/fallen_cup/cups_grasp_poses",
            self._cups_grasp_poses_cb, 10)
        # 정상 컵 위치 (upright_cup_pose_node). 빈 안전지점 선택에 사용.
        self.create_subscription(
            MarkerArray, self.upright_boxes_topic,
            self._upright_boxes_cb, 10)

        # ── 피라미드 장애물 상태 + API polling ────────────────────────────
        # center/degree 는 백그라운드 스레드가 API 를 polling 해 갱신(순수 데이터만
        # 갱신, planning scene 변경은 메인 스레드에서만). /stack 점유는 구독으로 수신.
        self._pyramid_center = None         # (x, y) | None  (API 동기화값)
        self._pyramid_degree = None         # float  | None
        self._pyramid_occupied = set()      # 점유 슬롯 키 집합 (/stack 최신)
        self._pyramid_stack_seen = False    # /stack 메시지 수신 여부
        self._pyramid_obstacle_ids = []     # 등록된 collision object id (정리용)
        self._upright_obstacle_ids = []     # 정상 컵 실린더 collision object id
        self._upright_co_pub = None         # RViz 표준 scene 용 (아래서 생성)
        # move_group 의 표준 scene(/collision_object → /monitored_planning_scene)
        # 에도 박스를 publish 해서 RViz MotionPlanning 에 보이게 한다. (노드 자체
        # MoveItPy scene 은 moveit_py.yaml 의 별도 토픽이라 RViz 가 안 봄.)
        self._pyramid_co_pub = None
        if self.pyramid_avoid:
            self._pyramid_co_pub = self.create_publisher(
                CollisionObject, "/collision_object", 10)
            self.create_subscription(
                String, self.pyramid_stack_topic, self._stack_cb, 10)
            if self.pyramid_config_url:
                threading.Thread(
                    target=self._pyramid_poll_loop, daemon=True).start()
                log.info(
                    f"[pyramid] avoid ON — /stack='{self.pyramid_stack_topic}', "
                    f"config polling {self.pyramid_config_url} "
                    f"every {self.pyramid_sync_poll_period_s:.0f}s")
            else:
                log.warn("[pyramid] config_url 비어있음 — center/degree 기본값 사용")
        else:
            log.info("[pyramid] avoid OFF — 장애물 등록 안 함")

        # 그리퍼 충돌 프록시: 노드 MoveItPy scene 에 직접 attach + move_group
        # 표준 scene 에도 /attached_collision_object 로 publish(RViz 수동 plan 용).
        # move_group 은 attached_collision_object 토픽을 구독하지 않으므로(world
        # geometry monitor 만), /apply_planning_scene 서비스로 diff 를 적용한다.
        # 실로봇(move_group 없음)에선 서비스가 없어 graceful skip.
        self.attach_gripper_collision = bool(
            self.get_parameter("attach_gripper_collision").value)
        self._apply_scene_cli = None
        if self.attach_gripper_collision:
            self._apply_scene_cli = self.create_client(
                ApplyPlanningScene, "/apply_planning_scene")

        # 도달불가(IK 해 없음) 컵 통보용 flag 토픽. pick 전 사전검증에서 approach/
        # descend IK 해가 없으면 그 컵 정보를 JSON String 으로 발행한다(집지 않음).
        self.unreachable_pub = self.create_publisher(
            String, "/fallen_cup/unreachable", 10
        )

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

    # ── Multi-cup 콜백 (Phase 2) ─────────────────
    def _cups_pose2d_cb(self, msg: Float32MultiArray):
        """/fallen_cup/cups_pose2d 콜백. layout = [{cup,N,N*13}, {field,13,13}].
        N 개 cup 의 pose2d (cup_id, yaw, conf 등) 를 dict 리스트로 저장.
        """
        if len(msg.layout.dim) < 2 or msg.layout.dim[1].size != 13:
            return
        n = msg.layout.dim[0].size
        if len(msg.data) != n * 13:
            return
        cups = []
        for i in range(n):
            row = msg.data[i * 13:(i + 1) * 13]
            cups.append({
                "cup_id": int(row[0]),
                "yaw": float(row[7]),
                "grip_px": (float(row[8]), float(row[9])),
                "conf": float(row[10]),
            })
        self._latest_cups_pose2d = cups

    def _cups_grasp_poses_cb(self, msg: PoseArray):
        """/fallen_cup/cups_grasp_poses 콜백. 같은 프레임의 cups_pose2d 와 결합해
        프레임 sample 생성. depth NaN cup 은 스킵.
        """
        if self._latest_cups_pose2d is None:
            return
        if len(msg.poses) != len(self._latest_cups_pose2d):
            return  # 프레임 mismatch — 다음 callback 기다림
        frame_cups = []
        for i, pose in enumerate(msg.poses):
            x = pose.position.x
            if math.isnan(x) or math.isnan(pose.position.y) or math.isnan(pose.position.z):
                continue
            p_cam = np.array([
                float(x), float(pose.position.y), float(pose.position.z)
            ])
            meta = self._latest_cups_pose2d[i]
            frame_cups.append({
                "cup_id": meta["cup_id"],
                "p_cam": p_cam,
                "yaw": meta["yaw"],
                "conf": meta["conf"],
            })
        if frame_cups:
            self._cups_frame_samples.append({
                "stamp_sec": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                "cups": frame_cups,
            })
            # 오래된 frame 정리 (메모리 무한 증가 방지)
            if len(self._cups_frame_samples) > 200:
                self._cups_frame_samples.pop(0)

    # ── Multi-cup 헬퍼 ───────────────────────────
    def _cluster_frame_samples(self, frames, radius_m):
        """frame 별 cup 들을 카메라 frame 공간 근접도로 클러스터링.

        각 클러스터 = 같은 물리적 cup 으로 추정되는 sample 모음.
        반환: list of {"center_cam": np.array(3), "samples": list of cup sample}
        """
        clusters = []
        for frame in frames:
            for cup in frame["cups"]:
                p = cup["p_cam"]
                best_idx = -1
                best_dist = float("inf")
                for ci, cl in enumerate(clusters):
                    d = float(np.linalg.norm(p - cl["center_cam"]))
                    if d < best_dist:
                        best_dist = d
                        best_idx = ci
                if best_idx >= 0 and best_dist < radius_m:
                    cl = clusters[best_idx]
                    cl["samples"].append(cup)
                    n = len(cl["samples"])
                    cl["center_cam"] = cl["center_cam"] * ((n - 1) / n) + p / n
                else:
                    clusters.append({
                        "center_cam": p.copy(),
                        "samples": [cup],
                    })
        return clusters

    def _compute_cluster_target(self, cluster, T_base_cam):
        """단일 클러스터에서 (p_base, cup_yaw_base, n_samples, R) 산출. 실패 시 None.
        compute_target 의 cluster 버전 — buffer 대신 cluster["samples"] 사용.
        """
        log = self.get_logger()
        samples = cluster["samples"]
        if len(samples) < self.multi_cup_min_samples_per_cluster:
            return None

        ps = np.array([s["p_cam"] for s in samples])
        p_cam = ps.mean(axis=0)

        p_base = (T_base_cam @ np.append(p_cam, 1.0))[:3]
        p_base[0] -= BASE_OFFSET_X
        p_base[1] -= BASE_OFFSET_Y
        p_base[2] -= BASE_OFFSET_Z

        yaws = [s["yaw"] for s in samples]
        m1 = circular_mean(yaws)
        thresh = math.radians(30.0)
        yaws_in = [y for y in yaws if abs(angular_diff(y, m1)) <= thresh]
        if not yaws_in:
            return None
        cam_yaw = circular_mean(yaws_in)
        R_val = circular_R(yaws_in)

        v_cam = np.array([math.cos(cam_yaw), math.sin(cam_yaw), 0.0])
        v_base = T_base_cam[:3, :3] @ v_cam
        cup_yaw_base = math.atan2(v_base[1], v_base[0])

        return (p_base, cup_yaw_base, len(samples), R_val)

    @staticmethod
    def _parse_spot_candidates(spec):
        """'x:y,x:y' → [(x, y), ...]. 파싱 실패 항목은 무시."""
        spots = []
        for tok in spec.split(","):
            tok = tok.strip()
            if not tok:
                continue
            try:
                xs, ys = tok.split(":")
                spots.append((float(xs), float(ys)))
            except (ValueError, TypeError):
                continue
        return spots

    def _upright_boxes_cb(self, msg):
        """/hand_eye/boxes (base_link MarkerArray) → 정상 컵 XY 스냅샷 갱신.
        매 발행이 DELETEALL + box_top/box_labels 전체 스냅샷이라 ns=='box_top'
        ADD 마커만 모아 통째로 교체한다."""
        cups = []
        for m in msg.markers:
            if m.action == Marker.ADD and m.ns == "box_top":
                cups.append(
                    (float(m.pose.position.x), float(m.pose.position.y)))
        self._upright_cups = cups

    def _select_safe_place_spot(self, blacklist):
        """후보 PLACE 좌표 중 정상 컵/이미 세운 컵/피라미드를 모두 피한 첫 빈 자리.
        없으면 None. 회피 반경 = place_spot_avoid_radius_m."""
        log = self.get_logger()
        r = self.place_spot_avoid_radius_m
        uprights = list(self._upright_cups)
        for (sx, sy) in self.place_spot_candidates:
            hit_up = next(
                ((ux, uy) for (ux, uy) in uprights
                 if math.hypot(sx - ux, sy - uy) < r), None)
            if hit_up is not None:
                log.info(
                    f"[spot] 후보 ({sx:.3f},{sy:.3f}) 스킵 — 정상 컵 "
                    f"({hit_up[0]:.3f},{hit_up[1]:.3f}) 근접")
                continue
            hit_bl = next(
                ((bx, by) for (bx, by) in blacklist
                 if math.hypot(sx - bx, sy - by) < r), None)
            if hit_bl is not None:
                log.info(
                    f"[spot] 후보 ({sx:.3f},{sy:.3f}) 스킵 — 기점유 "
                    f"({hit_bl[0]:.3f},{hit_bl[1]:.3f}) 근접")
                continue
            if self._pyramid_center is not None:
                pcx, pcy = self._pyramid_center
                if math.hypot(sx - pcx, sy - pcy) < r:
                    log.info(
                        f"[spot] 후보 ({sx:.3f},{sy:.3f}) 스킵 — 피라미드 근접")
                    continue
            log.info(
                f"[spot] 빈 안전지점 선택 → ({sx:.3f},{sy:.3f}) "
                f"(정상 컵 {len(uprights)}개·기점유 {len(blacklist)}개 회피)")
            return (sx, sy)
        log.warn(
            f"[spot] 빈 안전지점 없음 — 후보 {len(self.place_spot_candidates)}개 "
            f"모두 점유/근접")
        return None

    def _sense_multi_targets(self, blacklist):
        """SAMPLE_COLLECT_SEC 동안 cup frame sample 수집 → 클러스터링 → base 변환
        → blacklist 필터 → base 거리 가까운 순으로 정렬한 후보 리스트 반환.

        반환: list of dict {p_base, cup_yaw, n_samples, R}, 비어 있으면 [].
        """
        log = self.get_logger()
        self._cups_frame_samples = []
        self._latest_cups_pose2d = None

        log.info(
            f"[multi-cup] sense 시작 (max {SAMPLE_COLLECT_SEC}s, "
            f"cluster_radius={self.multi_cup_cluster_radius_m * 1000:.0f}mm)"
        )
        t0 = time.time()
        last_status = 0.0
        while rclpy.ok() and time.time() - t0 < SAMPLE_COLLECT_SEC:
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.time() - last_status > 1.0:
                last_status = time.time()
                log.info(
                    f"[multi-cup] frames={len(self._cups_frame_samples)}"
                )

        if not self._cups_frame_samples:
            log.warn("[multi-cup] frame sample 한 개도 못 받음")
            return []

        clusters = self._cluster_frame_samples(
            self._cups_frame_samples, self.multi_cup_cluster_radius_m
        )
        log.info(
            f"[multi-cup] frames={len(self._cups_frame_samples)}, "
            f"clusters={len(clusters)} "
            f"(min_samples={self.multi_cup_min_samples_per_cluster})"
        )

        T_base_ee = get_ee_matrix(self.robot)
        T_ee_cam = self.gripper2cam
        T_base_cam = T_base_ee @ T_ee_cam

        candidates = []
        for cl_idx, cluster in enumerate(clusters):
            result = self._compute_cluster_target(cluster, T_base_cam)
            if result is None:
                log.info(
                    f"[multi-cup] cluster {cl_idx} 스킵 "
                    f"(samples={len(cluster['samples'])} < min)"
                )
                continue
            p_base, cup_yaw, n_samples, R_val = result

            # blacklist (이미 세운 cup) 체크
            in_bl = False
            for bx, by in blacklist:
                if math.hypot(p_base[0] - bx, p_base[1] - by) < self.multi_cup_blacklist_radius_m:
                    in_bl = True
                    break
            if in_bl:
                log.info(
                    f"[multi-cup] cluster {cl_idx} 스킵 "
                    f"(blacklist hit @ ({p_base[0]:.3f},{p_base[1]:.3f}))"
                )
                continue

            dist_from_base = math.hypot(p_base[0], p_base[1])
            log.info(
                f"[multi-cup] cluster {cl_idx}: "
                f"p_base=({p_base[0]:.3f},{p_base[1]:.3f},{p_base[2]:.3f}) "
                f"cup_yaw={math.degrees(cup_yaw):+.1f}° "
                f"n={n_samples} R={R_val:.2f} "
                f"dist_from_base={dist_from_base:.3f}m"
            )
            candidates.append({
                "p_base": p_base,
                "cup_yaw": cup_yaw,
                "n_samples": n_samples,
                "R": R_val,
                "dist": dist_from_base,
            })

        candidates.sort(key=lambda c: c["dist"])
        return candidates

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
    def ik_state_with_current_seed(self, pose_stamped, timeout=1.0,
                                   seed_overrides=None, retries=12,
                                   retry_timeout=0.2,
                                   max_seed_jump=math.radians(90)):
        """
        현재 관절 상태를 seed로 IK를 풀어 RobotState를 만든다.
        목적: descend / lift에서 IK가 다른 branch를 골라 wrist가 회전하는 것 방지.
        seed_overrides (dict[str, float] | None): 주어지면 현재 joint dict에
          update해서 seed로 사용. 예: {"joint_1": 1.0472} → joint_1을 60°로 강제.
          IK solver(KDL)는 seed 근처로 수렴하므로 desired branch 유도 효과.

        1차(원래 seed = current + overrides, full timeout) 실패 시 fallback:
          KDL 수치 IK 는 seed 의존성이 커서 해가 존재해도 단일 seed 로는 수렴 못하는
          경우가 흔하다(예: 베이스 안쪽 + 낮은 z 의 top-down 자세). 그래서 base_seed
          주변을 점점 넓혀가며 랜덤 seed 로 재시도한다. base_seed 와의 최대 관절 편차가
          max_seed_jump 이내인 해를 우선 채택(원래 branch 유지 → wrist-flip/큰 swing
          방지). 그런 해가 끝내 없으면 편차가 가장 작은 해라도 반환(완전 실패보다 나음).
        반환: 성공 시 RobotState, 실패 시 None.
        """
        log = self.get_logger()
        psm = self.robot.get_planning_scene_monitor()
        with psm.read_only() as scene:
            current_joints = dict(scene.current_state.joint_positions)

        base_seed = dict(current_joints)
        if seed_overrides:
            for jn, jv in seed_overrides.items():
                if jn in base_seed:
                    base_seed[jn] = float(jv)
                else:
                    log.warn(f"[ik-seed] unknown joint '{jn}' in overrides — 무시")

        joint_names = list(base_seed.keys())

        def _solve(seed, to):
            st = RobotState(self.robot_model)
            st.joint_positions = seed
            st.update()
            if not st.set_from_ik(GROUP_NAME, pose_stamped.pose, EE_LINK, to):
                return None
            st.update()
            return st

        # 1차: 원래 seed (current + overrides) — branch 유지 우선, full timeout.
        st = _solve(base_seed, timeout)
        if st is not None:
            return st

        # fallback: base_seed 주변 랜덤 seed 로 KDL 재수렴 유도.
        best = None  # (jump, state) — max_seed_jump 초과지만 가장 가까운 해 보관.
        for k in range(retries):
            scale = 0.2 + 0.7 * (k / max(1, retries - 1))  # 0.2 → 0.9 rad 로 점증
            seed = {jn: base_seed[jn] + float(np.random.uniform(-scale, scale))
                    for jn in joint_names}
            cand = _solve(seed, retry_timeout)
            if cand is None:
                continue
            sol = cand.joint_positions
            jump = max(abs(sol[jn] - base_seed[jn]) for jn in joint_names)
            if jump <= max_seed_jump:
                log.warn(
                    f"[ik-retry] seed#{k + 1}/{retries} 수렴 (1차 current-seed 실패, "
                    f"max Δjoint={math.degrees(jump):.0f}°)"
                )
                return cand
            if best is None or jump < best[0]:
                best = (jump, cand)

        if best is not None:
            log.warn(
                f"[ik-retry] branch 유지 해 없음 → 최소 편차 해 채택 "
                f"(max Δjoint={math.degrees(best[0]):.0f}°). wrist-flip/swing 주의."
            )
            return best[1]

        log.error(
            f"IK 실패: pose=({pose_stamped.pose.position.x:.3f},"
            f"{pose_stamped.pose.position.y:.3f},"
            f"{pose_stamped.pose.position.z:.3f}) "
            f"(current-seed + 랜덤 seed {retries}회 모두 실패)"
        )
        return None

    # ── 수직 상승 헬퍼 ─────────────────────────────
    def _lift_straight_up(self, target_z):
        """현재 EE 자세(XY·orientation) 를 그대로 유지하며 z 만 target_z 까지
        짧은 LIN step 으로 수직 상승. clamp=False 로 현재 XY 를 고정해 대각선화 방지.
        한 step 이 실패하면 그때까지 도달한 높이에서 멈춘다(부분 상승도 컵 회피에 유효).
        반환: 도달한 z (m).
        """
        log = self.get_logger()
        cur_T = get_ee_matrix(self.robot)
        cur_x = float(cur_T[0, 3])
        cur_y = float(cur_T[1, 3])
        start_z = float(cur_T[2, 3])
        if start_z >= target_z - 1e-3:
            return start_z
        cqx, cqy, cqz, cqw = rotmat_to_quat_xyzw(cur_T[:3, :3])
        cur_ori = {"x": cqx, "y": cqy, "z": cqz, "w": cqw}
        log.info(
            f"[lift] 자세 유지 수직 상승 (LIN, XY 고정·no-clamp) "
            f"z {start_z:.3f}→{target_z:.3f} @ XY=({cur_x:+.3f},{cur_y:+.3f})"
        )
        # 1) 먼저 한 번의 연속 LIN 으로 target_z 까지 — 끊김 없이 매끄럽게 상승.
        full_pose = make_pose(cur_x, cur_y, target_z, cur_ori)
        if plan_and_execute(self.robot, self.arm, log,
                            pose_goal=full_pose,
                            params=self.lin_params, clamp=False):
            log.info(f"[lift] 단일 LIN 성공 → z={target_z:.3f}")
            return target_z
        # 2) 단일 LIN 실패(특이점 가로지름 등) 시에만 짧은 step 으로 fallback.
        #    계단식이라 끊기지만, 부분 상승이라도 컵/테이블 회피에 유효.
        log.warn("[lift] 단일 LIN 실패 — 짧은 step fallback (계단식)")
        step = RETREAT_LIFT_STEP_M
        reached = start_z
        n_steps = max(1, int(math.ceil((target_z - start_z) / step)))
        for i in range(n_steps):
            z_goal = min(target_z, start_z + step * (i + 1))
            step_pose = make_pose(cur_x, cur_y, z_goal, cur_ori)
            if plan_and_execute(self.robot, self.arm, log,
                                pose_goal=step_pose,
                                params=self.lin_params, clamp=False):
                reached = z_goal
            else:
                log.warn(
                    f"[lift] LIN step z→{z_goal:.3f} 실패 "
                    f"(확보 높이 {reached:.3f}m, Δ={reached - start_z:.3f}m)"
                )
                break
        return reached

    # ── 메인 ──────────────────────────────
    # ── HOME 복귀 헬퍼 (Phase 2) ─────────────────
    def _return_to_session_home(self, final=False):
        """session HOME 으로 복귀.
        1단계 (pre-reverse): joint_1 이 HOME 과 크게 다르면 (예: plus_y_auto_swing
                            에 의한 +60°) joint_1 만 먼저 HOME 값으로 PTP. 다른 joint
                            유지 → EE z 변동 없음 → 테이블 dip 위험 없음.
        2단계 (HOME 복귀): 전체 joint 를 session HOME 으로. OMPL 먼저
                          (collision-aware sampling), Pilz PTP fallback. PTP first 로
                          했더니 joint-space 직선이 Cartesian dip 을 만들어 테이블 충돌
                          궤적이 나옴 — OMPL 로 다시 두는 게 안전.
        final=True 면 그리퍼 open + 시작 자세 대비 검증 로그까지 수행.
        반환: True (성공), False (모든 시도 실패).
        """
        log = self.get_logger()

        # 0단계: 안전 높이 확보 — pre-reverse(joint_1 수평 호) 전에 반드시 EE 를 띄운다.
        # retreat 가 실패해 EE 가 낮은 채로 들어오면(예: -Y standing IK 실패), joint_1
        # 만 회전시키는 pre-reverse 가 그리퍼를 테이블과 평행하게 쓸어버린다(작업영역
        # 컵 타격). 낮으면 먼저 수직 상승시켜 그 sweep 을 원천 차단한다.
        ee_z_now = float(get_ee_matrix(self.robot)[2, 3])
        if ee_z_now < HOME_PREREVERSE_MIN_Z:
            log.info(
                f"[Home] EE z={ee_z_now:.3f}m < {HOME_PREREVERSE_MIN_Z:.2f}m "
                "→ pre-reverse 전 수직 상승 (수평 sweep 방지)"
            )
            ee_z_now = self._lift_straight_up(LIFT_Z)
            log.info(f"[Home] 수직 상승 결과 z={ee_z_now:.3f}m")

        # 1단계: pre-reverse joint_1 (필요시, 그리고 EE 가 충분히 높을 때만)
        # plus_y_auto_swing 등으로 joint_1 이 +60° 같은 큰 값에 가 있는 상태에서
        # 곧장 HOME 으로 가면 6-joint 동시 보간이 Cartesian dip 을 만들어 테이블
        # 충돌 가능. joint_1 만 먼저 HOME 값으로 돌려 두면 EE z 가 그대로 유지되고
        # 이후 HOME 복귀는 joint 변화량이 작아져 dip risk 도 작아짐.
        # ※ 단 EE 가 낮으면(수직 상승 실패) joint_1 회전이 곧 수평 sweep 이므로 스킵.
        home_j1 = self._session_home_joints.get("joint_1", 0.0)
        try:
            current_joints = self._read_current_joints()
            current_j1 = current_joints.get("joint_1", home_j1)
        except Exception as exc:
            log.warn(f"[Home] 현재 joint 읽기 실패 ({exc}) — pre-reverse 스킵")
            current_joints = None

        if current_joints is not None and ee_z_now < HOME_PREREVERSE_MIN_Z:
            log.warn(
                f"[Home] EE z={ee_z_now:.3f}m 여전히 낮음 → joint_1 pre-reverse 스킵 "
                "(수평 sweep 방지). OMPL collision-aware 복귀에 위임."
            )
        elif current_joints is not None:
            j1_diff_deg = math.degrees(abs(current_j1 - home_j1))
            if j1_diff_deg > 20.0:
                log.info(
                    f"[Home] pre-reverse: joint_1 "
                    f"{math.degrees(current_j1):+.1f}° → "
                    f"{math.degrees(home_j1):+.1f}° (다른 joint 유지, EE z={ee_z_now:.3f}m)"
                )
                intermediate = dict(current_joints)
                intermediate["joint_1"] = home_j1
                inter_state = RobotState(self.robot_model)
                inter_state.joint_positions = intermediate
                inter_state.update()
                inter_ok = plan_and_execute(self.robot, self.arm, log,
                                            state_goal=inter_state,
                                            params=self.pilz_params)
                if not inter_ok:
                    log.warn(
                        "[Home] pre-reverse 실패 — 그대로 HOME 시도 (Cartesian dip "
                        "가능성 ↑)"
                    )

        # 2단계: 전체 HOME 복귀
        home_back = RobotState(self.robot_model)
        home_back.joint_positions = self._session_home_joints
        home_back.update()

        # 플래너 순서 결정 (EE 높이 기반):
        #   - 이미 충분히 높으면(>HOME_RETURN_HIGH_Z): Pilz PTP 먼저.
        #     OMPL(RRTConnect)은 샘플링 기반이라 가끔 빙 도는 과한 궤적을 내놓는데,
        #     retreat(z=LIFT_Z) + pre-reverse 로 이미 떠 있으면 joint 직선보간(PTP)이
        #     테이블 dip 없이 결정적·최단 경로로 HOME 에 간다.
        #   - 낮은 자세에서 호출된 경우(예외): 기존대로 OMPL 먼저 (dip 회피).
        try:
            ee_z = float(get_ee_matrix(self.robot)[2, 3])
        except Exception:
            ee_z = 0.0
        if ee_z > HOME_RETURN_HIGH_Z:
            log.info(
                f"[Home] EE z={ee_z:.3f}m > {HOME_RETURN_HIGH_Z:.2f}m → "
                "Pilz PTP 우선 (결정적·최단), OMPL fallback"
            )
            primary, secondary = self.pilz_params, self.ompl_params
            primary_name, secondary_name = "Pilz PTP", "OMPL"
        else:
            log.info(
                f"[Home] EE z={ee_z:.3f}m ≤ {HOME_RETURN_HIGH_Z:.2f}m → "
                "OMPL 우선 (낮은 자세 dip 회피), Pilz PTP fallback"
            )
            primary, secondary = self.ompl_params, self.pilz_params
            primary_name, secondary_name = "OMPL", "Pilz PTP"

        home_ok = plan_and_execute(self.robot, self.arm, log,
                                   state_goal=home_back,
                                   params=primary)
        if not home_ok:
            log.warn(f"[Home] {primary_name} HOME 복귀 실패 — {secondary_name} 로 재시도")
            home_ok = plan_and_execute(self.robot, self.arm, log,
                                       state_goal=home_back,
                                       params=secondary)
        if not final:
            return home_ok

        # final: gripper open + 시작/종료 자세 비교
        self._gripper_move(GRIP_OPEN_WIDTH, GRIP_FORCE)
        time.sleep(0.5)
        if hasattr(self, "_start_T_base_ee"):
            end_T_base_ee = get_ee_matrix(self.robot)
            dp = end_T_base_ee[:3, 3] - self._start_T_base_ee[:3, 3]
            dp_norm = float(np.linalg.norm(dp))
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
        return home_ok

    # ── Multi-cup loop (Phase 2) ─────────────────
    def _run_multi_cup_loop(self):
        """한 프레임에 여러 cup 이 있을 때 가까운 순서대로 in-place stand.

        각 iteration:
          1. (it>0) HOME 복귀 (재sense 위해 camera 시야 확보)
          2. settle 대기
          3. _sense_multi_targets() — cups 토픽 sample 수집 + 클러스터링 +
             blacklist 필터 + base 거리 정렬
          4. 후보 없으면 종료
          5. 가장 가까운 cup 선택, _active_place_x/y = cup 원위치
          6. _pick_and_handle_cup() 호출
          7. 성공 시 blacklist 에 PLACE 추가, 실패 시 skip
        루프 종료 후 final HOME 복귀.

        반환: 실제로 세운 cup 이 하나라도 있으면 True, 0개면 False
        (감지 0개 / 전부 도달불가 / 전부 pick 실패 → 실패로 통보).
        """
        log = self.get_logger()
        placed = []  # blacklist: list of (x, y) base frame — 이미 세운 cup 위치
        stood = 0    # 실제로 세운 cup 수(도달불가/skip 제외) — 성공 판정용

        for it in range(self.multi_cup_max_iterations):
            log.info(
                f"[multi-cup] === Iteration {it + 1}/"
                f"{self.multi_cup_max_iterations} ==="
            )

            # 첫 iter 는 직전에 HOME init 끝났으니 스킵.
            # 이후 iter 는 HOME 복귀해서 camera 시야 확보.
            if it > 0:
                log.info("[multi-cup] HOME 복귀 (재sense 전)")
                if not self._return_to_session_home(final=False):
                    log.error("[multi-cup] HOME 복귀 실패 — 루프 중단")
                    break

            log.info("[multi-cup] sense settle 대기 0.5s")
            time.sleep(0.5)

            candidates = self._sense_multi_targets(placed)
            # sense 동안 갱신된 정상 컵 스냅샷을 collision object 로 등록/갱신
            # (이후 approach/lift/place 궤적이 정상 컵을 회피).
            self._register_upright_obstacles()
            if not candidates:
                log.info("[multi-cup] 처리할 cup 없음 → 루프 종료")
                break

            target = candidates[0]  # 가장 가까운 cup
            p_base = target["p_base"]
            cup_yaw = target["cup_yaw"]
            log.info(
                f"[multi-cup] 선택: p_base=("
                f"{p_base[0]:.3f},{p_base[1]:.3f},{p_base[2]:.3f}) "
                f"cup_yaw={math.degrees(cup_yaw):+.1f}° "
                f"dist={target['dist']:.3f}m (남은 후보 {len(candidates) - 1}개)"
            )

            # 사전 도달성 검사: approach/descend IK 해가 없으면 집지 않고
            # blacklist 처리 + /fallen_cup/unreachable 통보(다음 sense 에서도 제외).
            if self.preflight_reach_check:
                reachable, reason = self._preflight_grasp_reachable(p_base, cup_yaw)
                if not reachable:
                    log.warn(
                        f"[preflight] cup ({p_base[0]:.3f},{p_base[1]:.3f}) "
                        f"도달불가 — {reason}. pick 생략 + blacklist 추가, 통보."
                    )
                    self._publish_unreachable(p_base, cup_yaw, reason)
                    placed.append((float(p_base[0]), float(p_base[1])))
                    continue
                log.info("[preflight] approach/descend IK OK → pick 진행")

            if self.place_in_place:
                # legacy: 제자리 세우기 — PLACE = cup 원위치
                self._active_place_x = float(p_base[0])
                self._active_place_y = float(p_base[1])
                log.info(
                    f"[multi-cup] PLACE = ({self._active_place_x:.3f}, "
                    f"{self._active_place_y:.3f}) (in-place stand)"
                )
            else:
                # 기본: 작업영역 빈 안전지점에 세우기 (정상 컵/기점유/피라미드 회피)
                spot = self._select_safe_place_spot(placed)
                if spot is None:
                    log.warn(
                        "[multi-cup] 빈 안전지점 없음 — 이 cup skip "
                        "(pick 안 함, blacklist 추가 안 함)"
                    )
                    continue
                self._active_place_x, self._active_place_y = spot
                log.info(
                    f"[multi-cup] PLACE = ({self._active_place_x:.3f}, "
                    f"{self._active_place_y:.3f}) (safe-spot stand)"
                )

            # cup_yaw override 가 set 되어 있으면 그대로 적용 (테스트 용도).
            # 정상 mode 에서는 NaN 이라 무시.
            ok = self._pick_and_handle_cup(p_base, cup_yaw)
            if not ok:
                log.warn(
                    f"[multi-cup] cup 처리 실패 — 다음 iteration 시도 "
                    "(blacklist 추가 안 함, 다음 sense 에서 재시도 가능)"
                )
                continue

            placed.append((self._active_place_x, self._active_place_y))
            stood += 1
            log.info(
                f"[multi-cup] cup 처리 완료. 세운 cup {stood}개 "
                f"(blacklist 누적 {len(placed)}개)"
            )

        log.info(
            f"[multi-cup] 루프 종료. 세운 cup: {stood}개 / "
            f"max_iter={self.multi_cup_max_iterations}"
        )
        # final HOME 복귀 + 검증
        log.info("[Final] HOME 복귀 (시작 자세로)")
        self._return_to_session_home(final=True)
        return stood > 0

    def _preflight_grasp_reachable(self, p_base, cup_yaw):
        """집기 전에 approach/descend IK 해가 있는지만 검사한다(실행·이동 없음).

        descend 는 approach 해의 관절을 seed 로 풀어 실제 실행 시퀀스(approach 도달
        후 그 자세에서 descend) 를 모사한다. ik_state_with_current_seed 의 랜덤 seed
        fallback 까지 거치므로, '집어보면 IK 가 풀릴 컵' 을 도달불가로 오판하지 않는다.
        반환: (ok: bool, reason: str).
        """
        bx, by, bz = float(p_base[0]), float(p_base[1]), float(p_base[2])
        ori = self.grip_orientation(cup_yaw)
        flange_at_grip = bz - CUP_R_AT_GRIP + TOOL_LENGTH_M
        approach_z = flange_at_grip + APPROACH_OFFSET
        descend_z = flange_at_grip + GRIP_Z_MARGIN
        if descend_z < SAFE_Z_MIN:
            descend_z = SAFE_Z_MIN

        approach_state = self.ik_state_with_current_seed(
            make_pose(bx, by, approach_z, ori)
        )
        if approach_state is None:
            return False, f"approach IK 해 없음 @ z={approach_z:.3f}"

        seed_from_approach = dict(approach_state.joint_positions)
        descend_state = self.ik_state_with_current_seed(
            make_pose(bx, by, descend_z, ori),
            seed_overrides=seed_from_approach,
        )
        if descend_state is None:
            return False, f"descend IK 해 없음 @ z={descend_z:.3f}"
        return True, "ok"

    def _publish_unreachable(self, p_base, cup_yaw, reason):
        """도달불가 컵을 /fallen_cup/unreachable 토픽(JSON String)으로 통보."""
        msg = String()
        msg.data = json.dumps({
            "x": round(float(p_base[0]), 4),
            "y": round(float(p_base[1]), 4),
            "z": round(float(p_base[2]), 4),
            "cup_yaw_deg": round(math.degrees(cup_yaw), 1),
            "reason": reason,
            "stamp": self.get_clock().now().nanoseconds,
        })
        self.unreachable_pub.publish(msg)

    def _pick_and_handle_cup(self, p_base, cup_yaw):
        """단일 cup 의 cup_yaw override → approach → descend → close → lift →
        drop/place (in-place stand) 전체 시퀀스. self._active_place_x/y 를 PLACE 좌표로 사용.
        반환: True (성공 또는 dry-run 정상 종료), False (모션 실패).
        """
        log = self.get_logger()
        # multi-cup loop 누수 방지: place_plus_y_auto_swing 블록이 이 세 인스턴스
        # 변수를 mutate 한 채로 함수가 끝나기 때문에, cup 마다 launch 시점 기본값
        # 으로 reset 하지 않으면 직전 cup 의 swing override 가 다음 cup 에 그대로
        # 적용됨 (잘못된 base yaw/tilt → IK 실패 또는 엉뚱한 trajectory).
        self.place_flange_side = self._default_place_flange_side
        self.place_base_yaw_deg = self._default_place_base_yaw_deg
        self.place_cup_tilt_deg = self._default_place_cup_tilt_deg
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
            return False

        if self.dry_run:
            log.warn("=== DRY-RUN: approach 자세 도달. 30초 정지 ===")
            log.warn("  → 그리퍼 손가락 평면이 컵 긴축에 [수직]이면 OK")
            log.warn("  → [평행]이면 YAW_OFFSET_DEG 부호 반대로 (90→-90)")
            t0 = time.time()
            while rclpy.ok() and time.time() - t0 < 30.0:
                rclpy.spin_once(self, timeout_sec=0.1)
            log.info("=== DRY-RUN 종료 ===")
            return True

        # 5) DESCEND — IK를 현재 관절 seed로 잠가서 wrist 회전 차단
        descend_z = flange_at_grip + GRIP_Z_MARGIN
        if descend_z < SAFE_Z_MIN:
            log.warn(f"descend_z={descend_z:.3f} → {SAFE_Z_MIN} (clamp)")
            descend_z = SAFE_Z_MIN
        descend_pose = make_pose(bx, by, descend_z, ori)
        descend_state = self.ik_state_with_current_seed(descend_pose)
        if descend_state is None:
            log.error("descend IK 실패 — 종료")
            return False
        log.info(
            f"[2] Descend @ z={descend_z:.3f} "
            "(joint goal, orientation 잠금)"
        )
        if not plan_and_execute(
                self.robot, self.arm, log,
                state_goal=descend_state,
                params=self.pilz_params):
            return False

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

        # 7) LIFT — Cartesian 수직 상승(XY 고정·orientation 잠금).
        #    PTP joint-space 보간은 EE 가 곡선을 그리며 피라미드(L1_M/L2_R)로
        #    swing → gripper_proxy 충돌로 plan 실패했음. LIN 수직은 XY 를 픽
        #    위치에 고정해 그리퍼가 피라미드로 휘둘리지 않는다.
        log.info(f"[4] Lift @ z={LIFT_Z:.3f} (Cartesian 수직, XY 고정·orientation 잠금)")
        reached_z = self._lift_straight_up(LIFT_Z)
        if reached_z < LIFT_Z - 0.03:
            # LIN 으로 충분히 못 올라옴(특이점 등) → collision-aware OMPL 로 재시도.
            log.warn(
                f"[4] 수직 LIN 확보 높이 {reached_z:.3f}m < 목표 {LIFT_Z:.3f}m "
                "→ OMPL(충돌회피)로 lift 재시도"
            )
            lift_pose = make_pose(bx, by, LIFT_Z, ori)
            lift_state = self.ik_state_with_current_seed(lift_pose)
            if lift_state is None or not plan_and_execute(
                    self.robot, self.arm, log,
                    state_goal=lift_state,
                    params=self.ompl_params):
                log.error("lift 실패 — 종료")
                return False

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

            # (pre-a) place_plus_y_auto_swing — cup wide 가 +Y 영역으로 누운 경우
            #   자동으로 swing strategy 발동 (side / base_yaw / cup_tilt 일괄 override).
            #   감지: sin(cup_yaw) > PLUS_Y_DETECT_SIN_THRESHOLD.
            #   미감지 시 사용자가 명시한(또는 기본) 파라미터 그대로 사용.
            if self.place_plus_y_auto_swing:
                sin_cup = math.sin(cup_yaw)
                if sin_cup > PLUS_Y_DETECT_SIN_THRESHOLD:
                    log.info(
                        f"[plus-y-auto] cup_yaw={math.degrees(cup_yaw):+.1f}° "
                        f"(sin={sin_cup:.3f}) > thr={PLUS_Y_DETECT_SIN_THRESHOLD} → "
                        "swing strategy 발동: "
                        f"side={self.place_plus_y_side}, "
                        f"base_yaw={self.place_plus_y_base_yaw_deg:+.1f}°, "
                        f"tilt={self.place_plus_y_cup_tilt_deg:+.1f}°"
                    )
                    self.place_flange_side = self.place_plus_y_side
                    self.place_base_yaw_deg = self.place_plus_y_base_yaw_deg
                    self.place_cup_tilt_deg = self.place_plus_y_cup_tilt_deg
                elif sin_cup < -PLUS_Y_DETECT_SIN_THRESHOLD:
                    # cup wide 가 -Y 로 누운 대칭 케이스: 위 +Y swing 을 좌우 반전.
                    #   side left↔right, base_yaw 부호 반전(+60→-60), tilt 동일.
                    # joint_1 을 -Y 쪽으로 swing 시켜 elbow 가 피라미드를 쓸지 않게 한다
                    #   (반전 없으면 default config 로 접근해 link_4/5 가 피라미드 통과).
                    mirror_side = "right" if self.place_plus_y_side == "left" else "left"
                    mirror_base_yaw = -self.place_plus_y_base_yaw_deg
                    log.info(
                        f"[plus-y-auto] cup_yaw={math.degrees(cup_yaw):+.1f}° "
                        f"(sin={sin_cup:.3f}) < -thr={-PLUS_Y_DETECT_SIN_THRESHOLD} → "
                        "-Y 대칭 swing 발동: "
                        f"side={mirror_side}, base_yaw={mirror_base_yaw:+.1f}°, "
                        f"tilt={self.place_plus_y_cup_tilt_deg:+.1f}°"
                    )
                    self.place_flange_side = mirror_side
                    self.place_base_yaw_deg = mirror_base_yaw
                    self.place_cup_tilt_deg = self.place_plus_y_cup_tilt_deg
                else:
                    log.info(
                        f"[plus-y-auto] cup_yaw={math.degrees(cup_yaw):+.1f}° "
                        f"(sin={sin_cup:.3f}) |·| ≤ thr → swing 미발동, "
                        "기본 파라미터 유지"
                    )

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

            # (b.5) gripper EE_Z를 항상 한쪽 ±base_Y로 강제 → obstacle 회피.
            #   place_flange_side="right": EE_Z=+base_Y → flange가 PLACE의 -Y 쪽
            #     (로봇이 오른쪽으로 눕음. +Y 측 obstacle 피함.)
            #   place_flange_side="left":  EE_Z=-base_Y → flange가 PLACE의 +Y 쪽
            #
            # 컵이 강제 방향과 반대로 누워 있으면 R_pre_twist의 EE_Z가 반대 방향이라
            # twist가 180° 가까이 나옴. 그래도 base_Z 축 회전이므로 컵 axis(수직) 유지,
            # wide 끝 down 유지. gripper 만 cup 주변으로 180° 회전된 자세가 됨.
            EE_Z_after = R_pre_twist[:, 2]
            ez_xy_norm = float(np.linalg.norm(EE_Z_after[:2]))
            if ez_xy_norm > 1e-6:
                cur_angle = math.atan2(EE_Z_after[1], EE_Z_after[0])
                # baseline: side에 따른 ±90° (기존 동작)
                if self.place_flange_side == "right":
                    baseline_target = math.pi / 2      # EE_Z = +Y
                else:  # "left"
                    baseline_target = -math.pi / 2     # EE_Z = -Y

                # target_angle 결정:
                #   1) place_flange_yaw_deg 가 set → 그 값을 직접 사용 (명시 override)
                #   2) place_flange_yaw_auto_extra_deg 가 set 이고
                #      provisional twist 가 PRE_TWIST_THRESHOLD_DEG 를 넘는 "문제 케이스"
                #      (= 컵 wide가 강제 방향과 반대로 누움) → baseline + sign*extra
                #   3) 그 외 → baseline 그대로
                if not math.isnan(self.place_flange_yaw_deg):
                    target_angle = math.radians(self.place_flange_yaw_deg)
                    log.info(
                        f"[place-yaw] override → target_angle="
                        f"{math.degrees(target_angle):+.1f}°"
                    )
                else:
                    prov_twist = baseline_target - cur_angle
                    prov_twist = math.atan2(math.sin(prov_twist),
                                            math.cos(prov_twist))
                    prov_twist_deg = math.degrees(prov_twist)
                    if (not math.isnan(self.place_flange_yaw_auto_extra_deg)
                            and abs(prov_twist_deg) > PRE_TWIST_THRESHOLD_DEG):
                        sign = 1.0 if baseline_target >= 0.0 else -1.0
                        extra = sign * math.radians(
                            self.place_flange_yaw_auto_extra_deg
                        )
                        target_angle = baseline_target + extra
                        log.info(
                            f"[place-yaw] auto-extra 발동 "
                            f"(|prov_twist|={abs(prov_twist_deg):.1f}° > "
                            f"{PRE_TWIST_THRESHOLD_DEG:.0f}°): target_angle "
                            f"{math.degrees(baseline_target):+.1f}° → "
                            f"{math.degrees(target_angle):+.1f}°"
                        )
                    else:
                        target_angle = baseline_target

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
                stand_target_angle = target_angle  # tilt 축 계산용 보관
            else:
                R_stand = R_pre_twist
                twist_deg = 0.0
                stand_target_angle = None

            # (b.6) place_cup_tilt_deg — cup 을 vertical 에서 -EE_Z 방향으로 α° 기울임.
            # 회전축 = (-sin target, cos target, 0)  (base XY 수평면에서 EE_Z를 +90° 회전)
            # 회전 적용: R_stand ← R_tilt @ R_stand
            # 효과: EE_Z 가 (cos·target·cos α, sin·target·cos α, -sin α) 로 변화 →
            #       flange Z 가 +TOOL_LENGTH·sin α 만큼 상승 (closing_z 고정 시).
            # closing_z 도 cos α 보정 (cup 의 vertical projection 이 짧아지므로 더 낮은 위치
            # 에서 release).
            tilt_alpha_rad = math.radians(self.place_cup_tilt_deg)
            if abs(tilt_alpha_rad) > 1e-6 and stand_target_angle is not None:
                tilt_axis = np.array([
                    -math.sin(stand_target_angle),
                    math.cos(stand_target_angle),
                    0.0,
                ])
                K = np.array([
                    [0.0, -tilt_axis[2], tilt_axis[1]],
                    [tilt_axis[2], 0.0, -tilt_axis[0]],
                    [-tilt_axis[1], tilt_axis[0], 0.0],
                ])
                R_tilt = (np.eye(3)
                          + math.sin(tilt_alpha_rad) * K
                          + (1.0 - math.cos(tilt_alpha_rad)) * (K @ K))
                R_stand = R_tilt @ R_stand
                log.info(
                    f"[cup-tilt] α={self.place_cup_tilt_deg:+.1f}° 적용 → "
                    f"예상 flange Z 상승 ≈ "
                    f"{TOOL_LENGTH_M * math.sin(tilt_alpha_rad) * 1000:+.0f}mm"
                )

            sqx, sqy, sqz, sqw = rotmat_to_quat_xyzw(R_stand)
            stand_ori = {"x": sqx, "y": sqy, "z": sqz, "w": sqw}
            log.info(
                f"[place] cup_dir_base=({cup_dir_base[0]:+.2f},"
                f"{cup_dir_base[1]:+.2f},{cup_dir_base[2]:+.2f}) → "
                f"align={align_angle_deg:.1f}° + "
                f"Z-twist={twist_deg:+.1f}° (side={self.place_flange_side})"
            )

            # (d) standing flange 위치
            # closing plane을 (PLACE_X, PLACE_Y, TABLE_Z+CUP_HEIGHT+margin) 로
            # → flange = closing_plane - TOOL_LENGTH_M * EE_Z_in_base
            EE_Z_stand = R_stand[:, 2]
            # cup이 α° 기울어진 경우 cup axis 의 vertical projection 이 cos α 배가 됨.
            # 따라서 cup top(=closing_plane) 도 그만큼 낮춰서 cup 바닥이 TABLE_Z+margin 에
            # 안착하도록.
            closing_z  = (TABLE_Z
                          + math.cos(tilt_alpha_rad) * CUP_HEIGHT
                          + self.stand_cup_margin_m)
            stand_fx   = self._active_place_x - TOOL_LENGTH_M * EE_Z_stand[0]
            stand_fy   = self._active_place_y - TOOL_LENGTH_M * EE_Z_stand[1]
            stand_fz   = closing_z - TOOL_LENGTH_M * EE_Z_stand[2]

            log.info(
                f"[5] place + pitch tilt: "
                f"closing=({self._active_place_x:+.3f},{self._active_place_y:+.3f},{closing_z:+.3f}) "
                f"[margin={self.stand_cup_margin_m:+.3f}m], "
                f"flange=({stand_fx:+.3f},{stand_fy:+.3f},{stand_fz:+.3f}), "
                f"EE_Z_base=({EE_Z_stand[0]:+.2f},{EE_Z_stand[1]:+.2f},{EE_Z_stand[2]:+.2f})"
            )

            # (d.4) place_base_yaw_deg 가 set이면 standing/retreat IK seed에 joint_1 강제.
            # 동일 override를 pre-twist IK에도 전달해서 pre-twist가 joint_1을 "자연스러운"
            # branch로 되돌리는 것을 막음.
            place_seed_overrides = None
            if not math.isnan(self.place_base_yaw_deg):
                place_seed_overrides = {
                    "joint_1": math.radians(self.place_base_yaw_deg)
                }
                log.info(
                    f"[place-base-yaw] IK seed override: "
                    f"joint_1 = {self.place_base_yaw_deg:+.1f}° "
                    "(pre-twist / standing / retreat 공통 적용)"
                )

            # (d.45) pre-base-yaw — joint_1을 lift 높이에서 사전 회전
            #   목적: standing motion 의 joint_1 변화량을 줄여서 trajectory 안정화 +
            #         elbow 가 target joint_1 방향(workspace 밖)으로 확실히 swing.
            #   동작: 현재 joint dict 의 joint_1만 target 으로 교체, joints 2~6 유지.
            #         joint-space PTP로 이동 → EE는 lift 높이(z=0.45m) 에서 호 그리며 swing.
            #         테이블에서 안전한 높이라 swing 도중 충돌 가능성 낮음.
            #   주의: 큰 각(예 ±90°+) 은 EE가 base 뒤쪽으로 가서 SAFE workspace 위반 가능.
            #         처음엔 ±45° 정도로 시작.
            if place_seed_overrides is not None:
                cur_joints = self._read_current_joints()
                cur_j1 = float(cur_joints.get("joint_1", 0.0))
                target_j1 = float(place_seed_overrides["joint_1"])
                log.info(
                    f"[pre-base-yaw] joint_1 {math.degrees(cur_j1):+.1f}° → "
                    f"{math.degrees(target_j1):+.1f}° @ lift z={LIFT_Z:.2f}m "
                    f"(Δ={math.degrees(target_j1 - cur_j1):+.1f}°)"
                )
                pre_base_joints = dict(cur_joints)
                pre_base_joints["joint_1"] = target_j1
                pre_base_state = RobotState(self.robot_model)
                pre_base_state.joint_positions = pre_base_joints
                pre_base_state.update()
                ok_pby = plan_and_execute(
                    self.robot, self.arm, log,
                    state_goal=pre_base_state,
                    params=self.pilz_params,
                )
                if not ok_pby:
                    log.warn(
                        "[pre-base-yaw] 사전 회전 실패 — standing seed override 만 가지고 fallback"
                    )

            # (d.5) 사전 base_Z 회전 — twist가 크면(cup wide가 +Y인 경우 ≈180°),
            # standing motion에서 (twist + tilt + translate)가 한 plan에 묶여 elbow가
            # 위로 들리는 high-arc 궤적이 나옴. twist만 먼저 lift 위치에서 적용해
            # cup을 수평인 채로 cup_dir만 뒤집어 두면, 이후 standing motion은 cup wide
            # 가 -Y인 경우와 기하학적으로 동일해져 깔끔한 tilt+translate만 남음.
            if abs(twist_deg) > PRE_TWIST_THRESHOLD_DEG:
                R_pre_only = R_twist @ R_lift_mat  # tilt 없이 base_Z twist만
                pqx, pqy, pqz, pqw = rotmat_to_quat_xyzw(R_pre_only)
                pre_ori = {"x": pqx, "y": pqy, "z": pqz, "w": pqw}
                ee_T_now = get_ee_matrix(self.robot)
                pre_x = float(ee_T_now[0, 3])
                pre_y = float(ee_T_now[1, 3])
                pre_z = float(ee_T_now[2, 3])
                log.info(
                    f"[pre-twist] |twist|={abs(twist_deg):.1f}° > "
                    f"{PRE_TWIST_THRESHOLD_DEG}° → base_Z 회전 분리 실행 "
                    f"@ ({pre_x:+.3f},{pre_y:+.3f},{pre_z:+.3f})"
                )
                pre_pose = make_pose(pre_x, pre_y, pre_z, pre_ori)
                pre_state = self.ik_state_with_current_seed(
                    pre_pose, timeout=2.0, seed_overrides=place_seed_overrides
                )
                if pre_state is not None:
                    ok_pre = plan_and_execute(
                        self.robot, self.arm, log,
                        state_goal=pre_state,
                        params=self.pilz_params,
                    )
                    if not ok_pre:
                        log.warn(
                            "[pre-twist] 사전 회전 실패 — 통합 standing motion 으로 fallback"
                        )
                else:
                    log.warn(
                        "[pre-twist] IK 실패 — 통합 standing motion 으로 fallback"
                    )

            # (e) lift → standing 한 번에 plan (IK seed 우선, 실패 시 pose goal)
            stand_pose  = make_pose(stand_fx, stand_fy, stand_fz, stand_ori)
            stand_state = self.ik_state_with_current_seed(
                stand_pose, timeout=2.0, seed_overrides=place_seed_overrides
            )
            if stand_state is not None:
                if place_seed_overrides is not None:
                    final_j1 = float(stand_state.joint_positions.get(
                        "joint_1", float("nan")
                    ))
                    log.info(
                        f"[place-base-yaw] IK 수렴 joint_1 = "
                        f"{math.degrees(final_j1):+.1f}° "
                        f"(target {self.place_base_yaw_deg:+.1f}°)"
                    )
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
                return False

            # (f) release
            log.info("[6] 그리퍼 open (cup release at standing)")
            self._gripper_move(GRIP_OPEN_WIDTH, GRIP_FORCE)
            if self.sim:
                self._cup_place_xy = (self._active_place_x, self._active_place_y)
                self._cup_state = "placed"
                log.info("[sim] cup placed (서있음)")
            time.sleep(1.0)

            # (g) retreat — 자세 그대로 z 만 수직 상승 (incremental Cartesian LIN).
            # 한 번의 큰 LIN 은 -Y standing 의 기운 wrist 자세에서 특이점을 가로질러
            # 실패하므로 짧은 step 으로 나눠 올린다. 부분 상승이라도 컵을 벗어나 안전.
            log.info(f"[7] 후퇴: 수직 상승 → z={LIFT_Z:.3f}")
            start_z = float(get_ee_matrix(self.robot)[2, 3])
            reached = self._lift_straight_up(LIFT_Z)
            if reached - start_z < 0.03:
                # LIN 이 거의 못 올라감 — standing 자세가 특이점/한계에 가깝다는 신호.
                # PTP IK 로 한 번 더 시도하되, 실패하면 HOME 복귀 단계가 (낮은 z 를
                # 감지해) 다시 수직 상승 후 안전하게 복귀하므로 여기서 sweep 위험 없음.
                log.warn(
                    f"[7] 수직 상승 거의 실패 (Δz={reached - start_z:.3f}m) — "
                    "PTP IK 로 재시도 (HOME 복귀에서 재상승 보장)"
                )
                retreat_pose  = make_pose(stand_fx, stand_fy, LIFT_Z, stand_ori)
                retreat_state = self.ik_state_with_current_seed(
                    retreat_pose, timeout=2.0, seed_overrides=place_seed_overrides
                )
                if retreat_state is not None:
                    plan_and_execute(self.robot, self.arm, log,
                                     state_goal=retreat_state,
                                     params=self.pilz_params)
            log.info("=== place (방법2: pitch tilt) 완료 ===")

        return True

    # ──────────────────────────────────────────
    #  Pyramid 장애물 회피
    # ──────────────────────────────────────────
    def _stack_cb(self, msg: String):
        """/stack (JSON {slot: color|null}) 수신 → 점유 슬롯 집합 갱신."""
        try:
            data = json.loads(msg.data)
        except (ValueError, TypeError) as e:
            self.get_logger().warn(
                f"[pyramid] /stack JSON 파싱 실패: {e}", throttle_duration_sec=10.0)
            return
        if not isinstance(data, dict):
            return
        occupied = {
            slot for slot, color in data.items()
            if color is not None and slot in PYRAMID_SLOT_MAP
        }
        self._pyramid_occupied = occupied
        self._pyramid_stack_seen = True

    def _fetch_pyramid_config(self, timeout=3.0):
        """GET /api/robot/config/pyramid → (center_x, center_y, degree) | None.
        Cloudflare 가 기본 urllib UA 를 403 하므로 curl 류 UA 로 위장(verifier 동일)."""
        url = self.pyramid_config_url
        if not url:
            return None
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "curl/7.81.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read())
            return (float(data["center"]["x"]),
                    float(data["center"]["y"]),
                    float(data["degree"]))
        except Exception as e:  # noqa: BLE001 — 도달불가/파싱오류 → None
            self.get_logger().warn(
                f"[pyramid] config fetch 실패: {e}", throttle_duration_sec=15.0)
            return None

    def _pyramid_poll_loop(self):
        """백그라운드: API 를 polling 해 center/degree 를 최신으로 유지.
        planning scene 은 건드리지 않는다(메인 스레드 전용). 값 변경 시에만 로그."""
        last = None
        while rclpy.ok():
            cfg = self._fetch_pyramid_config()
            if cfg is not None:
                cx, cy, deg = cfg
                self._pyramid_center = (cx, cy)
                self._pyramid_degree = deg
                key = (round(cx, 4), round(cy, 4), round(deg, 2))
                if key != last:
                    self.get_logger().info(
                        f"[pyramid] config 동기화 center=({cx:.3f},{cy:.3f}) "
                        f"degree={deg:.1f}")
                    last = key
            time.sleep(self.pyramid_sync_poll_period_s)

    def _pyramid_slot_box(self, slot_key, center_xy, degree_deg, margin):
        """점유 슬롯 → (id, [dx,dy,dz], (cx,cy,cz)) base_link AABB. verifier
        get_virtual_box 와 동일 기하. z 는 TABLE_Z 에 앵커(바닥부터 보수적)."""
        layer, pos, count = PYRAMID_SLOT_MAP[slot_key]
        theta = math.radians(degree_deg)
        ux, uy = math.cos(theta), math.sin(theta)
        offset = (pos - (count - 1) / 2.0) * PYRAMID_CUP_SPACING
        cx = center_xy[0] + offset * ux
        cy = center_xy[1] + offset * uy
        z_bottom = TABLE_Z + layer * PYRAMID_LAYER_PITCH
        cz = z_bottom + PYRAMID_CUP_BOX_H / 2.0
        dx = PYRAMID_CUP_SPACING + 2.0 * margin
        dy = PYRAMID_CUP_DEPTH + 2.0 * margin
        dz = PYRAMID_CUP_BOX_H
        # 박스를 행 방향(degree)으로 회전 → 긴 축(dx=spacing)이 행을 따라간다.
        return (f"pyramid_{slot_key}", [dx, dy, dz], (cx, cy, cz), theta)

    def _build_gripper_aco(self):
        """RG2 envelope BOX 를 link_6 에 붙이는 AttachedCollisionObject 생성."""
        co = CollisionObject()
        co.header.frame_id = EE_LINK          # link_6 좌표계로 부착
        co.id = GRIPPER_PROXY_ID
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = list(GRIPPER_PROXY_SIZE)
        pose = Pose()
        pose.position.z = GRIPPER_PROXY_OFFZ  # 플랜지 +Z 로 박스 중심 이동
        pose.orientation.w = 1.0
        co.primitives.append(prim)
        co.primitive_poses.append(pose)
        co.operation = CollisionObject.ADD
        aco = AttachedCollisionObject()
        aco.link_name = EE_LINK
        aco.object = co
        aco.touch_links = list(GRIPPER_PROXY_TOUCH)
        return aco

    def _attach_gripper_proxy(self):
        """그리퍼 부피를 link_6 에 attach → 이후 모든 plan 이 그리퍼-장애물 충돌
        까지 회피. 노드 MoveItPy scene(실제 plan) + move_group scene(RViz) 양쪽.
        실로봇/sim 공통, 그리퍼는 항상 물리적으로 달려있으므로 기본 ON."""
        log = self.get_logger()
        if not self.attach_gripper_collision:
            log.info("[gripper-co] attach OFF — planning 이 그리퍼 부피 무시")
            return
        aco = self._build_gripper_aco()
        # 1) 노드 자체 MoveItPy scene (실제 recovery plan 이 보는 scene)
        psm = self.robot.get_planning_scene_monitor()
        with psm.read_write() as scene:
            scene.process_attached_collision_object(aco)
            scene.current_state.update()
        # 2) move_group 표준 scene (RViz MotionPlanning 수동 plan 용) — 서비스 diff
        self._apply_aco_to_move_group(aco)
        s = GRIPPER_PROXY_SIZE
        log.info(
            f"[gripper-co] RG2 프록시 attach @ {EE_LINK} "
            f"box=[{s[0]:.2f},{s[1]:.2f},{s[2]:.2f}]m offz={GRIPPER_PROXY_OFFZ:.2f} "
            f"touch={GRIPPER_PROXY_TOUCH}")

    def _apply_aco_to_move_group(self, aco):
        """AttachedCollisionObject 를 외부 move_group scene 에 적용(RViz 시각화용).
        move_group 미가동(실로봇)이면 서비스가 없어 조용히 skip."""
        log = self.get_logger()
        cli = self._apply_scene_cli
        if cli is None:
            return
        if not cli.wait_for_service(timeout_sec=2.0):
            log.warn("[gripper-co] /apply_planning_scene 없음 — move_group "
                     "scene 갱신 skip (실로봇이면 정상, RViz 수동 plan 에만 영향)")
            return
        scene = PlanningScene()
        scene.is_diff = True
        scene.robot_state.is_diff = True
        scene.robot_state.attached_collision_objects.append(aco)
        req = ApplyPlanningScene.Request()
        req.scene = scene
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        ok = fut.result().success if fut.result() else False
        log.info(f"[gripper-co] move_group scene apply: success={ok}")

    def _register_pyramid_obstacles(self):
        """쌓인 피라미드를 MoveIt collision object 로 등록. 메인 스레드에서 호출.
        서버/토픽 미가용이면 graceful no-op (장애물 0개)."""
        if not self.pyramid_avoid:
            return
        log = self.get_logger()

        # 1) center/degree 확보: poll 스레드 값 우선, 없으면 1회 동기 fetch.
        center = self._pyramid_center
        degree = self._pyramid_degree
        if center is None or degree is None:
            cfg = self._fetch_pyramid_config()
            if cfg is not None:
                center = (cfg[0], cfg[1])
                degree = cfg[2]
        if center is None or degree is None:
            log.warn(
                "[pyramid] config 미확보(서버 미응답) — 장애물 등록 스킵 "
                f"(기본값 무시, recovery 정상 진행)")
            return

        # 2) /stack 점유 수집: sticky 라 잠깐 spin 하면 즉시 도착.
        t0 = time.time()
        while (rclpy.ok() and not self._pyramid_stack_seen
               and time.time() - t0 < self.pyramid_stack_wait_s):
            rclpy.spin_once(self, timeout_sec=0.05)
        if not self._pyramid_stack_seen:
            log.warn(
                f"[pyramid] '{self.pyramid_stack_topic}' 미수신 "
                f"(vision 스택 미가동?) — 장애물 등록 스킵, recovery 정상 진행")
            return

        occupied = sorted(self._pyramid_occupied)
        if not occupied:
            log.info("[pyramid] 점유 슬롯 0개 — 빈 피라미드, 장애물 없음")
            return

        # 3) 점유 슬롯마다 collision object(BOX) 등록.
        margin = self.pyramid_obstacle_margin_m
        boxes = [self._pyramid_slot_box(s, center, degree, margin)
                 for s in occupied]
        psm = self.robot.get_planning_scene_monitor()
        registered = []
        with psm.read_write() as scene:
            for obj_id, dims, (cx, cy, cz), yaw in boxes:
                co = CollisionObject()
                co.header.frame_id = BASE_FRAME
                co.id = obj_id
                prim = SolidPrimitive()
                prim.type = SolidPrimitive.BOX
                prim.dimensions = dims
                pose = Pose()
                pose.position.x = cx
                pose.position.y = cy
                pose.position.z = cz
                pose.orientation.z = math.sin(yaw / 2.0)
                pose.orientation.w = math.cos(yaw / 2.0)
                co.primitives.append(prim)
                co.primitive_poses.append(pose)
                co.operation = CollisionObject.ADD
                scene.apply_collision_object(co)
                # RViz(move_group 표준 scene)에도 동일 박스 publish.
                if self._pyramid_co_pub is not None:
                    co.header.stamp = self.get_clock().now().to_msg()
                    self._pyramid_co_pub.publish(co)
                registered.append(obj_id)
            scene.current_state.update()
        self._pyramid_obstacle_ids = registered
        log.info(
            f"[pyramid] 장애물 {len(registered)}개 등록 "
            f"(center=({center[0]:.3f},{center[1]:.3f}), degree={degree:.1f}, "
            f"margin={margin*1000:.0f}mm): {', '.join(occupied)}")
        for obj_id, dims, (cx, cy, cz), yaw in boxes:
            log.info(
                f"    {obj_id}: c=({cx:.3f},{cy:.3f},{cz:.3f}) "
                f"box=[{dims[0]:.3f},{dims[1]:.3f},{dims[2]:.3f}] "
                f"yaw={math.degrees(yaw):.0f}°")

    def _register_upright_obstacles(self):
        """정상(세워진) 컵(self._upright_cups)을 MoveIt 실린더 collision object 로
        등록/갱신. 메인 스레드에서 호출. 매번 이전 등록을 REMOVE 한 뒤 최신 스냅샷으로
        다시 ADD 한다(컵이 사라지거나 옮겨져도 정합). 정상 컵 0개면 전부 REMOVE 만."""
        if not self.avoid_upright_cups:
            return
        log = self.get_logger()
        cups = list(self._upright_cups)
        radius = self.upright_obstacle_radius_m
        height = self.upright_obstacle_height_m
        cz = TABLE_Z + height / 2.0

        psm = self.robot.get_planning_scene_monitor()
        with psm.read_write() as scene:
            # 1) 이전 프레임 실린더 제거 (REMOVE) — 옮겨졌거나 사라진 컵 정리.
            for old_id in self._upright_obstacle_ids:
                rm = CollisionObject()
                rm.header.frame_id = BASE_FRAME
                rm.id = old_id
                rm.operation = CollisionObject.REMOVE
                scene.apply_collision_object(rm)
                if self._upright_co_pub is not None:
                    rm.header.stamp = self.get_clock().now().to_msg()
                    self._upright_co_pub.publish(rm)
            # 2) 최신 정상 컵마다 실린더 ADD.
            registered = []
            for i, (ux, uy) in enumerate(cups):
                obj_id = f"upright_cup_{i}"
                co = CollisionObject()
                co.header.frame_id = BASE_FRAME
                co.id = obj_id
                prim = SolidPrimitive()
                prim.type = SolidPrimitive.CYLINDER
                # CYLINDER dims = [height, radius]
                prim.dimensions = [height, radius]
                pose = Pose()
                pose.position.x = float(ux)
                pose.position.y = float(uy)
                pose.position.z = cz
                pose.orientation.w = 1.0
                co.primitives.append(prim)
                co.primitive_poses.append(pose)
                co.operation = CollisionObject.ADD
                scene.apply_collision_object(co)
                if self._upright_co_pub is not None:
                    co.header.stamp = self.get_clock().now().to_msg()
                    self._upright_co_pub.publish(co)
                registered.append(obj_id)
            scene.current_state.update()
        self._upright_obstacle_ids = registered
        if registered:
            log.info(
                f"[upright] 정상 컵 장애물 {len(registered)}개 등록 "
                f"(r={radius*1000:.0f}mm, h={height*1000:.0f}mm): "
                + ", ".join(f"({x:.3f},{y:.3f})" for x, y in cups))
        else:
            log.info("[upright] 정상 컵 0개 — 장애물 없음(이전 등록 정리)")

    def run(self):
        """recovery 1회 실행. 성공 시 True, 실패 시 False 를 돌려준다.

        main() 이 이 bool 을 프로세스 종료코드(0/1)로 변환하고, 그 코드가
        cup_stack wrapper launch → 서버 LaunchManager 까지 그대로 전파돼
        /api/robot/status 의 task 상태(idle=성공 / failed=실패)가 된다.
        """
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
            log.info("[Init] HOME 이동 (코드 HOME_JOINTS, Pilz PTP)")
            home_state = RobotState(self.robot_model)
            home_state.joint_positions = HOME_JOINTS
            home_state.update()
            # PTP(관절 단조 보간) 우선. OMPL(RRTConnect)은 무작위라 시작 자세가
            # HOME 과 멀면 팔을 천장으로 쳐들었다 내려오는 경로를 만들 수 있다.
            if not plan_and_execute(self.robot, self.arm, log,
                                    state_goal=home_state,
                                    params=self.pilz_params):
                log.warn("[Init] Pilz PTP HOME 실패 — OMPL 재시도")
                if not plan_and_execute(self.robot, self.arm, log,
                                        state_goal=home_state,
                                        params=self.ompl_params):
                    log.error("HOME 이동 실패 — 종료")
                    return False
            self._session_home_joints = dict(HOME_JOINTS)

        # 시작 시점의 link_6 pose를 저장 (종료 시 동일한 자세로 복귀했는지 검증용)
        self._start_T_base_ee = get_ee_matrix(self.robot)
        sp = self._start_T_base_ee[:3, 3]
        log.info(
            f"[Init] HOME link_6 pose: "
            f"pos=({sp[0]:.3f},{sp[1]:.3f},{sp[2]:.3f})"
        )

        # 1.4) 그리퍼 부피를 link_6 에 attach (이후 모든 plan 이 그리퍼-장애물 회피).
        self._attach_gripper_proxy()

        # 1.5) 쌓인 피라미드를 MoveIt 장애물로 등록 (이후 모든 plan 이 회피).
        self._register_pyramid_obstacles()

        # 2) 그리퍼 열기
        self._gripper_move(GRIP_OPEN_WIDTH, GRIP_FORCE)
        time.sleep(1.0)

        # multi-cup mode 분기: 별도 loop 가 sense + handle 을 반복.
        # loop 자체가 final HOME 복귀까지 수행하므로 여기서 return.
        if self.multi_cup and not self.sim:
            return self._run_multi_cup_loop()

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
                return False

            target = self.compute_target()
            if target is None:
                log.error("target 계산 실패 — 종료")
                return False
            p_base, cup_yaw = target

        # sense 동안 갱신된 정상 컵을 collision object 로 등록 (궤적 회피).
        self._register_upright_obstacles()

        # single-cup mode: 기본 PLACE 좌표(또는 place_x/y override) 사용
        _px = self.get_parameter("place_x").value
        _py = self.get_parameter("place_y").value
        self._active_place_x = PLACE_X if math.isnan(_px) else float(_px)
        self._active_place_y = PLACE_Y if math.isnan(_py) else float(_py)
        if not self._pick_and_handle_cup(p_base, cup_yaw):
            return False

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

        # 컵을 세웠으면 recovery 성공. HOME 복귀 실패는 경고만(컵은 이미 섰음).
        return True


def main(args=None):
    rclpy.init(args=args)
    node = StandFallenCupNode()
    ok = False
    try:
        ok = bool(node.run())
    finally:
        node.destroy_node()
        rclpy.shutdown()
    # 종료코드로 성공/실패를 알린다(0=성공, 1=실패). cup_stack wrapper launch 가
    # 이 코드를 전파하고 서버 LaunchManager 가 task 상태(idle/failed)로 반영한다.
    # (rclpy 는 이미 shutdown 됐으므로 logger 대신 stderr 로 출력.)
    if not ok:
        print(
            "[stand_fallen_cup] recovery 실패 — 종료코드 1 로 종료(서버에 failed 통보)",
            file=sys.stderr,
        )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
