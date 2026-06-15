#!/usr/bin/env python3
"""
place_mouth_up_cup.py

mouth-up-cup(넓은 입구가 천장/카메라를 향해 똑바로 선 컵)을 옆에서 잡아 들어올린 뒤
그리퍼를 ee_z(=joint_6) 둘레 180° 롤로 뒤집어(mouth-down) PLACE_X/Y 에 엎어 내려놓는 노드
(speed-stack 형태). 단일 task — 별도 recovery 불필요.

  왜 joint_6 180° 로 mouth-down 이 되나: 컵을 옆에서 잡으면 컵 축(월드 +Z)이 flange 롤축
  (ee_z)과 거의 직교라, ee_z 둘레 180° 롤(=joint_6)이면 컵 축이 +Z→-Z 로 뒤집혀 mouth-down
  이 된다. 과거엔 같은 180° 를 손가락축(월드 X, rot_x) 둘레로 줬는데, 그러면 mouth-down 은
  되지만 ee_z(접근축)가 반전돼 flange 가 반대쪽으로 점프 → joint_5 ~237° 대휘둘림(위험 궤적)
  이었다. 회전축을 ee_z(joint_6)로 바꾸면 ee_z 가 유지돼 손목 재배치 없이 싸고 안전하다.

  tilt 와의 trade-off: 잡을 때 grip_tilt_deg(카메라-바닥 클리어용)만큼 ee_z 가 수평에서
  기울어 있으면, 순수 joint_6 180° 롤로는 컵이 2·tilt 만큼 기운 mouth-down 이 된다
  (tilt=0 이면 정확히 수직). 평평하게 엎으려면 floor 클리어가 허용하는 선에서 grip_tilt_deg
  를 줄여야 한다. (예전엔 'ee_z 의 수평투영 h' 둘레 180° 로 정확한 수직을 만들었으나, 그건
  ee_z 의 상하 성분을 반전시켜 손목이 통째로 뒤집히는 고비용·위험 궤적 → joint_6 가드에
  막혀 OMPL 랜덤폴백으로 컵을 크게 휘둘렀다. 그래서 순수 joint_6 롤로 되돌렸다.)

비전 입력:
  /mouth_up_cup/grasp_pose (PoseStamped, camera optical frame, 컵 윗면 중심)
  → mouth_up_cup_pose_node 가 발행. 이 노드가 자기 EE FK 로 base_link 변환.

동작 시퀀스:
  1) 컵 윗면 중심 → base_link (cx, cy, cz)
  2) 접근 방향 결정 후 작업영역 바깥에서 수평(사선) 측면 접근.
  3) 컵 옆면 중간높이에서 측면 그립 → CLOSE → 수직 상승(LIFT).
  4) PLACE_X/Y 위로 이동 → joint_6 180° mouth-down flip(단계 분할) → 하강 안착 → release.
  5) HOME 복귀.

좌표 변환은 stand_fallen_cup.py 검증식 재사용:
  T_base_cam = get_ee_matrix(robot) @ gripper2cam ;  p_base = T_base_cam·p_cam - BASE_OFFSET

전제: stand_fallen_cup.py 와 동일 환경 (MoveItPy, RG2, hand-eye calib, bringup).
"""

import math
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory

from geometry_msgs.msg import PoseStamped

from moveit.core.robot_state import RobotState
from moveit.planning import MoveItPy, PlanRequestParameters

from .onrobot import RG
from .common import (
    make_pose,
    plan_and_execute,
    get_ee_matrix,
    rotmat_to_quat_xyzw,
    GROUP_NAME,
    EE_LINK,
    HOME_JOINTS,
)


# ─────────────────────────────────────────────────────────
#  설정
# ─────────────────────────────────────────────────────────
GRIPPER_NAME     = "rg2"
TOOLCHARGER_IP   = "192.168.1.1"
TOOLCHARGER_PORT = 502

# 컵 치수 (사용자 측정)
CUP_HEIGHT       = 0.095   # 컵 높이 (m)
CUP_MID_DIAMETER = 0.065   # 옆면 중간 지름 (m)
GRIP_HEIGHT      = CUP_HEIGHT / 2.0   # 옆면 중간 높이(바닥 기준) ≈ 4.75cm
# 컵을 잡는 높이만 추가로 더 낮춘다(m). grip_z 에서만 빼고 place_z_seat 에는 쓰지
# 않아, 검증된 place 안착 높이는 그대로 두고 grab 만 더 아래에서 잡게 한다.
GRAB_EXTRA_DROP_M = 0.02

TABLE_Z          = 0.05    # 테이블 표면 z (base frame, m). 실측 조정.

# flange(link_6) → 그리퍼 손가락 closing 평면 거리. 수평 그립에서는 flange 가
# TCP 에서 -EE_Z(=접근 반대) 방향으로 이만큼 떨어진다.
TOOL_LENGTH_M    = 0.20

APPROACH_OFFSET  = 0.12    # 수평 접근 back-off (m). 컵 바깥에서 이만큼 떨어져 정렬 후
                           # EE_Z 방향으로 직진 삽입.
LIFT_Z           = 0.30    # 그립 후 들어올리는 TCP z 상한(절대, m). 수평 손목자세를
                           # 유지한 채라 너무 높으면 IK 가 안 풀려(손목 관절한계),
                           # 실제 lift 는 이 값부터 내려가며 IK 가 풀리는 높이를 채택.
LIFT_Z_MIN_CLEAR = 0.10    # 그립점 대비 최소 상승량(m). 테이블 클리어 보장.
LIFT_Z_STEP      = 0.03    # IK 탐색 하강 간격(m)
# HOME 복귀 직전 수직 상승(LIN) step 크기(m). 짧을수록 특이점을 덜 가로질러 안정.
HOME_LIFT_STEP_M = 0.06
LIFT_HOLD_SEC    = 1.0
# 컵을 엎어 내려놓을 위치 (base frame, m). fallen-cup-recovery 처럼 작업영역 중앙이
# 아니라 Y-끝지점에 세운다. PLACE_Y 는 런타임에 컵 y 부호로 ±PLACE_Y_MAG 자동 선택.
PLACE_X          = 0.30
PLACE_Y_MAG      = 0.20    # |PLACE_Y|. 작업영역 Y-끝쪽(SAFE_Y_MAX 0.30 안). 더 끝으로.
# place 직전 joint_1 을 place 쪽으로 미리 스윙(pre-base-yaw)해 팔꿈치를 작업영역
# 바깥으로 뺀다(fallen-cup pre-base-yaw 미러). 절대 joint_1 = place_side · 이 각도.
# lift 높이에서 joint-space PTP 라 안전. 이후 carry/flip/lower seed IK 가 이 elbow-out
# 분기를 이어받아 손목/팔꿈치가 작업영역 바닥에 닿지 않는다(2026-06-15 실로봇 요청).
PLACE_BASE_YAW_DEG = 45.0
PLACE_MARGIN_Z   = 0.010   # mouth-down rim 안착 시 테이블 위 여유 (m)
# 안착 z 추가 하강(m). 팔꿈치를 바깥으로 뺀 뒤 컵이 공중에서 떨어지며 튕기지 않게
# seat 목표를 이만큼 더 낮춘다. PLACE_FLANGE_CLEAR 하한은 지켜 flange 가 바닥에 안 닿음.
PLACE_Z_DROP     = 0.07
# mouth-down flip 후 place 의 ee_z 는 위로 tilt 만큼 기울어 flange 가 TCP 보다
# TOOL·sin(tilt) 아래로 내려간다. place release z 를 띄워 flange 가 최소 이만큼
# 바닥에서 떨어지게 보정.
PLACE_FLANGE_CLEAR = 0.05
# 안착 직전 컵을 '더 세우는' 보정각(도). flip 후 컵 축은 vertical 에서 2·grip_tilt
# (≈30°) 만큼 접근쪽으로 기운 채 내려가 가끔 넘어진다. release 직전 EE 를 월드 X 축
# 둘레로 이만큼 돌려 컵 축을 vertical 쪽으로 당겨 안착 성공률을 높인다. full-flip(부호
# 반전) 이 아니라 작은 bounded 보정이라 손목 한계/도달성에 거의 영향 없음. 3~5° 권장.
PLACE_TILT_FIX_DEG = 5.0
# mouth-down flip = 순수 joint_6 180° 롤을 관절공간에서 단계 분할(매끄러운 swing).
# 단계마다 blocking PTP 라 매 단계 끝에서 감속·정지 → 단계가 많을수록 느리고 끊긴다.
# 팔은 정지하고 손목(joint_6)만 도는 안전 동작이라 2단계로 줄여 start-stop 을 절반으로.
FLIP_STAGES          = 2
RELEASE_HOLD_SEC = 1.0

# grab TCP 의 최저 안전 z(m). grip_z 클램프에만 쓰인다. tilt 그립에서 flange 는
# TCP 보다 TOOL·sin(tilt)≈5cm 위에 있어 TCP 가 이 값이어도 flange/테이블 여유 충분.
# GRAB_EXTRA_DROP_M(2cm) 만큼 더 낮춰 잡을 수 있도록 0.04 로 둔다.
SAFE_Z_MIN       = 0.04

# IK 시드 분기 가드: set_from_ik 해가 시드(현재자세)에서 이보다 크게 벗어난 관절이
# 하나라도 있으면 '다른 분기(천장 elbow-up/wrist-flip)'로 보고 거부. 정상 접근/이동은
# 보통 관절당 1rad 안쪽이라 2.0rad 면 정상해는 통과하고 천장 휘젓기 해만 걸러낸다.
IK_MAX_JOINT_DELTA = 2.0

# 그리퍼 raw 단위 (1/10 mm). 옆면 중간 지름 65mm.
GRIP_OPEN_WIDTH  = 850     # 85.0 mm (65mm 컵 + 여유)
GRIP_CLOSE_WIDTH = 600     # 60.0 mm (65mm 컵에 약한 압박)
GRIP_FORCE       = 250

# Hand-eye calibration 잔여 오차 보정 (base frame, m). stand_fallen_cup 와 동일 값.
BASE_OFFSET = np.array([0.0, 0.0, 0.080])

# 인식 안정화
SAMPLE_COLLECT_SEC = 4.0
MIN_SAMPLES        = 3


# run() 반환 상태(통합 오케스트레이터 outlier_cup_recovery 가 라우팅에 사용).
#   RECOVER_DONE : mouth-up 컵 1개를 잡아 세우는 데 성공 → 같은 종류 더 있는지 재시도
#   RECOVER_NONE : 처리할 mouth-up 컵 감지 0개 → mouth-up 단계 정상 종료
#   RECOVER_FAIL : 감지했으나 처리 실패(HOME/IK/place 실패) → 실패로 통보
RECOVER_DONE = "recovered"
RECOVER_NONE = "none"
RECOVER_FAIL = "failed"


def _rot_z(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def _rot_x(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ])


def side_grip_rotmat(approach_sign, tilt_deg=0.0):
    """측면 그립용 EE 회전행렬 (columns = [EE_X, EE_Y, EE_Z], base frame).

    approach_sign: +1 → 컵의 +Y(왼쪽)에서 -Y 방향으로 접근,
                   -1 → 컵의 -Y(오른쪽)에서 +Y 방향으로 접근.
    tilt_deg: 0 이면 완전 수평 그립(EE_Z 가 XY 평면). 양수면 EE_Z 가 그만큼
        **아래로** 기운다 → flange 가 grip 점보다 위에 위치(tool 이 비스듬히
        아래를 향함) → 짧은 컵을 낮은 높이에서 잡을 때 손목 자세가 자연스러워져
        IK 가 잘 풀리고 특이점을 피한다. 단, flip 후 place 단계는 반대로
        flange 가 낮아지므로 너무 크게(>30°) 주면 place reach 가 나빠질 수 있음.
    - EE_Z(접근축) = 컵을 향하는 수평성분(-approach_sign·Y) + 하향성분(-Z).
    - EE_X(손가락 닫힘축) = (1, 0, 0)  → 월드 X. 컵 양옆(±X)을 잡는다(수평 유지).
    - EE_Y = EE_Z × EE_X.
    """
    a = math.radians(tilt_deg)
    s = float(approach_sign)
    ee_z = np.array([0.0, -s * math.cos(a), -math.sin(a)])
    ee_z = ee_z / np.linalg.norm(ee_z)
    ee_x = np.array([1.0, 0.0, 0.0])          # 월드 X (EE_Z 와 직교: x성분 0)
    ee_y = np.cross(ee_z, ee_x)
    ee_y = ee_y / np.linalg.norm(ee_y)
    return np.column_stack([ee_x, ee_y, ee_z])


class PlaceMouthUpCupNode(Node):
    def __init__(self, moveit_py=None, gripper=None):
        super().__init__("place_mouth_up_cup")
        log = self.get_logger()

        # outlier_cup_recovery 통합 실행 시 오케스트레이터가 MoveItPy·그리퍼를
        # 1개만 만들어 주입한다. standalone 이면 둘 다 None → 자체 생성(기존 동작).
        self._injected_moveit_py = moveit_py
        self._injected_gripper = gripper

        self.declare_parameter("dry_run", False)
        # false(기본): 코드의 고정 원점 HOME_JOINTS 로 시작 시 이동 + 복귀.
        # true: launch 시점 자세를 세션 HOME 으로 캡처(이동 스킵).
        self.declare_parameter("use_current_as_home", False)
        # 접근 방향 강제: "auto"(기본, cy 부호) / "left"(+Y) / "right"(-Y)
        self.declare_parameter("approach_side", "auto")
        self.declare_parameter("robot_namespace", "")
        # sim: 카메라/그리퍼 HW 없이 MoveIt virtual 시각화
        self.declare_parameter("sim", False)
        self.declare_parameter("sim_cup_x", 0.45)
        self.declare_parameter("sim_cup_y", 0.20)
        self.declare_parameter("sim_cup_z", TABLE_Z + CUP_HEIGHT)
        # 측면 그립 하향 tilt(도). 0=완전 수평. 양수면 그리퍼가 그만큼 아래로 기울어
        # 잡는다(사선 그립) → 손목/카메라가 grip 점보다 TOOL_LENGTH·sin(tilt) 위로 올라가
        # 낮은 컵을 잡을 때 카메라가 바닥에 닿는 충돌을 피한다. 기본 15°(2026-06-09 실로봇:
        # 수평 그립이 카메라-바닥 충돌 유발. 25°는 접근 orientation 이 awkward 해져 IK 가
        # 천장 분기로 튐 → 15°로 낮춰 수평에 가깝게 유지하면서 손목 +5cm 확보).
        # tilt 의 주목적은 grab 시 카메라-바닥 클리어. mouth-down flip 단계에서 생기는
        # 2·tilt 잔여 기울기는 목표를 tilt-0 클린 mouth-down 으로 잡아 측지선 보간으로 흡수.
        self.declare_parameter("grip_tilt_deg", 15.0)
        # 그립 높이 미세조정(m). 기본 -0.03 = 기하중심(옆면 중간)보다 3cm 아래에서 잡기.
        # 2026-06-09 실로봇: 중간높이로는 너무 높게 잡혀 3cm 낮춤(낮을수록 CoG 안정).
        self.declare_parameter("grip_z_offset", -0.03)

        self.dry_run = bool(self.get_parameter("dry_run").value)
        self.use_current_as_home = bool(
            self.get_parameter("use_current_as_home").value)
        self.approach_side = str(
            self.get_parameter("approach_side").value).strip().lower()
        if self.approach_side not in ("auto", "left", "right"):
            log.warn(f"unknown approach_side '{self.approach_side}' → 'auto'")
            self.approach_side = "auto"
        self.robot_namespace = str(
            self.get_parameter("robot_namespace").value or "").strip()
        self.sim = bool(self.get_parameter("sim").value)
        self.sim_cup_x = float(self.get_parameter("sim_cup_x").value)
        self.sim_cup_y = float(self.get_parameter("sim_cup_y").value)
        self.sim_cup_z = float(self.get_parameter("sim_cup_z").value)
        self.grip_tilt_deg = float(self.get_parameter("grip_tilt_deg").value)
        self.grip_z_offset = float(self.get_parameter("grip_z_offset").value)

        if self.sim:
            log.warn(
                f"=== SIM MODE: 인식/그리퍼 우회. cup=("
                f"{self.sim_cup_x:.3f},{self.sim_cup_y:.3f},{self.sim_cup_z:.3f}) ===")
        if self.dry_run:
            log.warn("=== DRY-RUN: 접근 자세까지만, gripper/insert/lift 스킵 ===")

        # Hand-Eye
        calib_file = (
            Path(get_package_share_directory("dsr_practice"))
            / "config" / "T_gripper2camera.npy")
        self.gripper2cam = np.load(str(calib_file)).astype(float)
        self.gripper2cam[:3, 3] /= 1000.0  # mm → m
        log.info(f"Hand-Eye 로드: {calib_file}")

        # 그리퍼 (sim 이면 HW 없어도 무시)
        if self._injected_gripper is not None:
            log.info("주입된 그리퍼 재사용 (outlier_cup_recovery)")
            self.gripper = self._injected_gripper
        else:
            try:
                self.gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)
            except Exception as e:
                if self.sim:
                    log.warn(f"[sim] gripper HW 연결 안 됨 (정상): {e}")
                    self.gripper = None
                else:
                    raise

        # MoveItPy (namespace 처리는 stand_fallen_cup 와 동일)
        log.info("MoveItPy 초기화 중…")
        if self._injected_moveit_py is not None:
            log.info("주입된 MoveItPy 재사용 (outlier_cup_recovery)")
            self.robot = self._injected_moveit_py
        else:
            moveit_kwargs = {"node_name": "place_mouth_up_cup_moveit_py"}
            ns = self.robot_namespace
            if ns and ns != "/":
                moveit_kwargs["name_space"] = ns
                moveit_kwargs["remappings"] = {"__ns": ns}
                log.info(f"MoveItPy namespace: {ns}")
            self.robot = MoveItPy(**moveit_kwargs)
        self.arm = self.robot.get_planning_component(GROUP_NAME)
        self.robot_model = self.robot.get_robot_model()
        log.info("MoveItPy 초기화 완료")

        # Plan 파라미터
        self.ompl_params = PlanRequestParameters(self.robot)
        self.ompl_params.planning_pipeline = "ompl"
        self.ompl_params.planner_id = "RRTConnect"
        self.ompl_params.max_velocity_scaling_factor = 0.30
        self.ompl_params.max_acceleration_scaling_factor = 0.15
        self.ompl_params.planning_time = 3.0

        self.pilz_params = PlanRequestParameters(self.robot)
        self.pilz_params.planning_pipeline = "pilz_industrial_motion_planner"
        self.pilz_params.planner_id = "PTP"
        self.pilz_params.max_velocity_scaling_factor = 0.20
        self.pilz_params.max_acceleration_scaling_factor = 0.10
        self.pilz_params.planning_time = 2.0

        self.lin_params = PlanRequestParameters(self.robot)
        self.lin_params.planning_pipeline = "pilz_industrial_motion_planner"
        self.lin_params.planner_id = "LIN"
        self.lin_params.max_velocity_scaling_factor = 0.10
        self.lin_params.max_acceleration_scaling_factor = 0.06
        self.lin_params.planning_time = 2.0

        # mouth-down flip 전용 PTP. 순수 joint_6 롤(팔 정지)이라 손목만 도는
        # 안전한 동작 → 일반 PTP 보다 빠른 속도/가속 스케일을 써서 빠르게 돌린다.
        self.flip_params = PlanRequestParameters(self.robot)
        self.flip_params.planning_pipeline = "pilz_industrial_motion_planner"
        self.flip_params.planner_id = "PTP"
        self.flip_params.max_velocity_scaling_factor = 0.60
        self.flip_params.max_acceleration_scaling_factor = 0.40
        self.flip_params.planning_time = 2.0

        self.grasp_samples = []
        self.create_subscription(
            PoseStamped, "/mouth_up_cup/grasp_pose", self._grasp_cb, 10)

    # ── 비전 콜백 ─────────────────────────────────────────────
    def _grasp_cb(self, msg: PoseStamped):
        self.grasp_samples.append(msg)

    def _read_current_joints(self):
        psm = self.robot.get_planning_scene_monitor()
        with psm.read_only() as scene:
            return dict(scene.current_state.joint_positions)

    def _gripper_move(self, width, force):
        if self.gripper is not None:
            self.gripper.move_gripper(width, force)
        else:
            self.get_logger().info(f"[sim] gripper.move_gripper({width},{force}) skip")

    # ── IK 시드 잠금 (wrist branch 고정) ──────────────────────
    def _seed_ik_raw(self, pose_stamped, timeout=1.0):
        """seed=현재자세로 IK. (state, jmax_joint_name, max_delta) 반환.
        IK 자체 실패면 (None, None, None). 가드(거부)는 적용하지 않는다."""
        psm = self.robot.get_planning_scene_monitor()
        with psm.read_only() as scene:
            current_joints = dict(scene.current_state.joint_positions)
        target_state = RobotState(self.robot_model)
        target_state.joint_positions = current_joints
        target_state.update()
        ok = target_state.set_from_ik(
            GROUP_NAME, pose_stamped.pose, EE_LINK, timeout)
        if not ok:
            return None, None, None
        target_state.update()
        sol = dict(target_state.joint_positions)
        deltas = {j: abs(sol[j] - current_joints[j])
                  for j in current_joints if j in sol}
        jmax = max(deltas, key=deltas.get)
        return target_state, jmax, deltas[jmax]

    def ik_state_with_current_seed(self, pose_stamped, timeout=1.0,
                                   max_joint_delta=IK_MAX_JOINT_DELTA):
        log = self.get_logger()
        state, jmax, dmax = self._seed_ik_raw(pose_stamped, timeout)
        if state is None:
            p = pose_stamped.pose.position
            log.error(f"IK 실패: pose=({p.x:.3f},{p.y:.3f},{p.z:.3f})")
            return None
        # set_from_ik 는 시드(현재자세)를 '시작 추정'으로만 쓸 뿐, 수치해가 멀리
        # 튀어 다른 wrist 분기/elbow-up(천장) 해로 수렴할 수 있다. 그러면 PTP 가
        # 그 먼 목표까지 팔을 크게 휘둘러(=천장 휘젓기, 과거 반복 버그) 위험하다.
        # → 시드 대비 관절 변화가 임계치를 넘으면 '먼 분기'로 보고 거부(fail-safe).
        if dmax > max_joint_delta:
            p = pose_stamped.pose.position
            log.error(
                f"IK 해가 시드에서 과도하게 벗어남: {jmax} Δ={dmax:.2f}rad "
                f"> {max_joint_delta:.2f} → 거부(천장 분기 방지). "
                f"pose=({p.x:.3f},{p.y:.3f},{p.z:.3f})")
            return None
        log.info(f"  IK ok (max Δ={dmax:.2f}rad @ {jmax})")
        return state

    def _choose_grip_roll(self, approach_tcp, R_base):
        """2지 평행 그리퍼는 접근축(ee_z) 둘레 180° 대칭 → R 과 R·Rz(π) 가 같은 그립.
        approach 포즈에서 손목이 덜 도는(시드에 가까운) 쪽을 골라 불필요한 ~210°
        joint_6 roll(가드에 걸려 종료되던 원인)을 피한다. 선택된 R 을 전체 시퀀스에 사용."""
        log = self.get_logger()
        best_R, best_d, best_tag = None, None, None
        for tag, Rz in (("roll0", np.eye(3)), ("roll180", _rot_z(math.pi))):
            R = R_base @ Rz
            st, jmax, d = self._seed_ik_raw(self._flange_pose(approach_tcp, R))
            if st is None:
                log.info(f"  [roll] {tag}: IK 실패")
                continue
            log.info(f"  [roll] {tag}: max Δ={d:.2f}rad @ {jmax}")
            if best_d is None or d < best_d:
                best_R, best_d, best_tag = R, d, tag
        if best_R is None:
            return R_base
        log.info(f"  [roll] 선택: {best_tag} (max Δ={best_d:.2f}rad)")
        return best_R

    # ── TCP(손가락 평면) 목표 → flange(link_6) PoseStamped ────
    def _flange_pose(self, tcp_xyz, R):
        """원하는 TCP 위치/자세(R) 를 link_6 flange PoseStamped 로 환산.
        TCP = flange + TOOL_LENGTH_M · EE_Z  →  flange = TCP - TOOL_LENGTH_M · EE_Z.
        """
        ee_z = R[:, 2]
        fp = np.asarray(tcp_xyz, dtype=float) - TOOL_LENGTH_M * ee_z
        qx, qy, qz, qw = rotmat_to_quat_xyzw(R)
        return make_pose(fp[0], fp[1], fp[2],
                         {"x": qx, "y": qy, "z": qz, "w": qw})

    # ── 카메라 frame 샘플 → base_link 컵 중심 ─────────────────
    def _compute_cup_base(self):
        log = self.get_logger()
        if len(self.grasp_samples) < MIN_SAMPLES:
            log.error(
                f"grasp_pose 샘플 부족: {len(self.grasp_samples)} < {MIN_SAMPLES}")
            return None
        ps = np.array([
            [s.pose.position.x, s.pose.position.y, s.pose.position.z]
            for s in self.grasp_samples
        ])
        p_cam = ps.mean(axis=0)
        T_base_cam = get_ee_matrix(self.robot) @ self.gripper2cam
        p_base = (T_base_cam @ np.append(p_cam, 1.0))[:3] - BASE_OFFSET
        log.info(
            f"camera p=({p_cam[0]:.3f},{p_cam[1]:.3f},{p_cam[2]:.3f}) → "
            f"base p=({p_base[0]:.3f},{p_base[1]:.3f},{p_base[2]:.3f})")
        return p_base

    # ── 수직 상승 헬퍼 (HOME 복귀 전, fallen-cup 미러) ──────────
    def _lift_straight_up(self, target_z):
        """현재 EE 자세(XY·orientation)를 그대로 유지하며 z 만 target_z 까지 수직
        상승. clamp=False 로 현재 XY 를 고정해 대각선화 방지. 한 step 이 실패하면
        그때까지 도달한 높이에서 멈춘다. 반환: 도달한 z(m)."""
        log = self.get_logger()
        cur_T = get_ee_matrix(self.robot)
        cur_x = float(cur_T[0, 3])
        cur_y = float(cur_T[1, 3])
        start_z = float(cur_T[2, 3])
        if start_z >= target_z - 1e-3:
            return start_z
        cqx, cqy, cqz, cqw = rotmat_to_quat_xyzw(cur_T[:3, :3])
        cur_ori = {"x": cqx, "y": cqy, "z": cqz, "w": cqw}
        log.info(f"[lift] 수직 상승 (LIN, XY 고정·no-clamp) "
                 f"z {start_z:.3f}→{target_z:.3f} @ XY=({cur_x:+.3f},{cur_y:+.3f})")
        full_pose = make_pose(cur_x, cur_y, target_z, cur_ori)
        if plan_and_execute(self.robot, self.arm, log, pose_goal=full_pose,
                            params=self.lin_params, clamp=False):
            log.info(f"[lift] 단일 LIN 성공 → z={target_z:.3f}")
            return target_z
        log.warn("[lift] 단일 LIN 실패 — 짧은 step fallback (계단식)")
        step = HOME_LIFT_STEP_M
        reached = start_z
        n_steps = max(1, int(math.ceil((target_z - start_z) / step)))
        for i in range(n_steps):
            z_goal = min(target_z, start_z + step * (i + 1))
            step_pose = make_pose(cur_x, cur_y, z_goal, cur_ori)
            if plan_and_execute(self.robot, self.arm, log, pose_goal=step_pose,
                                params=self.lin_params, clamp=False):
                reached = z_goal
            else:
                log.warn(f"[lift] LIN step z→{z_goal:.3f} 실패 "
                         f"(확보 높이 {reached:.3f}m)")
                break
        return reached

    # ── 메인 ──────────────────────────────────────────────────
    def run(self):
        log = self.get_logger()

        log.info("[Init] controller 연결 대기 3s")
        time.sleep(3.0)

        # 1) HOME 결정
        if self.use_current_as_home:
            self._session_home_joints = self._read_current_joints()
            log.info("[Init] use_current_as_home=true → 현재 자세를 세션 HOME 캡처")
        else:
            log.info("[Init] HOME 이동 (코드 HOME_JOINTS, Pilz PTP)")
            home_state = RobotState(self.robot_model)
            home_state.joint_positions = HOME_JOINTS
            home_state.update()
            # PTP(관절공간 단조 보간) 우선. OMPL(RRTConnect)은 무작위 샘플링이라
            # 시작 자세가 HOME 과 멀면 팔을 천장으로 쳐들었다 내려오는 경로를
            # 만들 수 있다(실로봇 사고). PTP 는 각 관절이 직접 보간돼 휘젓지 않음.
            if not plan_and_execute(self.robot, self.arm, log,
                                    state_goal=home_state,
                                    params=self.pilz_params):
                log.warn("[Init] Pilz PTP HOME 실패 — OMPL 재시도")
                if not plan_and_execute(self.robot, self.arm, log,
                                        state_goal=home_state,
                                        params=self.ompl_params):
                    log.error("HOME 이동 실패 — 종료")
                    return RECOVER_FAIL
            self._session_home_joints = dict(HOME_JOINTS)

        # 2) 그리퍼 열기
        self._gripper_move(GRIP_OPEN_WIDTH, GRIP_FORCE)
        time.sleep(1.0)

        # 3) 컵 위치 확보
        if self.sim:
            p_base = np.array([self.sim_cup_x, self.sim_cup_y, self.sim_cup_z])
            log.info(f"[sim] cup_base=({p_base[0]:.3f},{p_base[1]:.3f},{p_base[2]:.3f})")
        else:
            log.info("[Sense] HOME settle 대기 1s")
            time.sleep(1.0)
            self.grasp_samples.clear()
            log.info(
                f"[Sense] /mouth_up_cup/grasp_pose 수집 "
                f"(최대 {SAMPLE_COLLECT_SEC}s, 최소 {MIN_SAMPLES}개)")
            t0 = time.time()
            while rclpy.ok() and time.time() - t0 < SAMPLE_COLLECT_SEC:
                rclpy.spin_once(self, timeout_sec=0.05)
                if len(self.grasp_samples) >= MIN_SAMPLES:
                    break
            if len(self.grasp_samples) == 0:
                log.error("=== /mouth_up_cup/grasp_pose 한 개도 못 받음 ===")
                log.error("  비전 노드(mouth_up_cup_pose_node) 가 떠 있는지, "
                          "컵이 화각 안인지 확인")
                return RECOVER_NONE
            p_base = self._compute_cup_base()
            if p_base is None:
                return RECOVER_FAIL

        if not self._pick_and_place(p_base):
            return RECOVER_FAIL

        # 먼저 수직으로 상승(LIN, XY·orientation 고정)한 뒤 HOME 복귀(fallen-cup 미러).
        #   낮은 place 자세에서 곧장 HOME PTP 를 하면 EE 가 작업영역을 대각선으로
        #   가로질러 옆 컵/테이블을 칠 수 있다. 수직으로 빠져나온 뒤 복귀하면 안전.
        log.info("[Final] HOME 복귀 전 수직 상승")
        self._lift_straight_up(LIFT_Z)

        # HOME 복귀 (Pilz PTP 우선 — startup 과 동일 이유로 천장 휘젓기 방지)
        log.info("[Final] HOME 복귀 (Pilz PTP)")
        home_back = RobotState(self.robot_model)
        home_back.joint_positions = self._session_home_joints
        home_back.update()
        if not plan_and_execute(self.robot, self.arm, log,
                                state_goal=home_back, params=self.pilz_params):
            log.warn("[Final] Pilz PTP HOME 복귀 실패 — OMPL 재시도")
            plan_and_execute(self.robot, self.arm, log,
                             state_goal=home_back, params=self.ompl_params)
        self._gripper_move(GRIP_OPEN_WIDTH, GRIP_FORCE)
        return RECOVER_DONE

    def _pick_and_place(self, p_base):
        log = self.get_logger()
        cx, cy, cz = float(p_base[0]), float(p_base[1]), float(p_base[2])

        # place 쪽 자동 선택(auto ±Y): 컵이 있는 Y 부호 쪽 작업영역 끝지점에 세운다.
        place_side = +1 if cy >= 0.0 else -1
        place_x = PLACE_X
        place_y = place_side * PLACE_Y_MAG
        log.info(f"[Plan] place=({place_x:.3f},{place_y:+.3f}) Y-끝지점 "
                 f"(side={'+Y' if place_side > 0 else '-Y'})")

        # 접근 방향 후보 결정.
        #   auto: 컵과 '같은 쪽'(cy 부호) 접근을 **우선** 시도하고, 그 쪽 IK 가
        #   안 풀리면(먼 +Y/-Y 컵은 flange 가 안전영역 밖으로 0.32m 더 나가 도달
        #   불가) **반대쪽 접근으로 폴백**한다. 같은 쪽이면 grab→lift→pre-base-yaw
        #   →place 가 모두 같은 Y 쪽에서 일어나 cross-body 스윙이 없어 더 좋지만,
        #   먼 컵에선 반대쪽 접근만 도달 가능하므로 자동 선택한다.
        if self.approach_side == "left":
            cand_signs = [+1]
        elif self.approach_side == "right":
            cand_signs = [-1]
        else:
            same = +1 if cy >= 0.0 else -1
            cand_signs = [same, -same]   # 같은 쪽 우선, 반대쪽 폴백

        grip_z = TABLE_Z + GRIP_HEIGHT + self.grip_z_offset - GRAB_EXTRA_DROP_M
        if grip_z < SAFE_Z_MIN:
            grip_z = SAFE_Z_MIN
        grip_tcp = np.array([cx, cy, grip_z])

        # 후보별로 approach/insert IK 를 모두 검증해, 둘 다 풀리는 첫 방향을 채택.
        approach_sign = None
        R_grip = None
        approach_tcp = None
        approach_state = None
        for sign in cand_signs:
            R_try = side_grip_rotmat(sign, tilt_deg=self.grip_tilt_deg)
            ee_z = R_try[:, 2]   # ee_z 는 roll(Rz) 에 불변
            appr_tcp = grip_tcp - APPROACH_OFFSET * ee_z   # 바깥(-EE_Z) back-off
            # 롤 대칭 후보 중 손목이 덜 도는 쪽 확정 → 그립~상승까지 동일 R 사용.
            R_try = self._choose_grip_roll(appr_tcp, R_try)
            appr_state = self.ik_state_with_current_seed(
                self._flange_pose(appr_tcp, R_try))
            if appr_state is None:
                log.warn(f"[1] approach_sign={sign:+d} approach IK 실패 — 다른 쪽 시도")
                continue
            if self.ik_state_with_current_seed(
                    self._flange_pose(grip_tcp, R_try)) is None:
                log.warn(f"[1] approach_sign={sign:+d} insert IK 실패 — 다른 쪽 시도")
                continue
            approach_sign, R_grip, approach_tcp, approach_state = (
                sign, R_try, appr_tcp, appr_state)
            break
        if approach_state is None:
            log.error("[1] 모든 접근 방향 IK 실패 — 종료")
            return False

        side_name = "left(+Y)" if approach_sign > 0 else "right(-Y)"
        log.info(
            f"[Plan] cup=({cx:.3f},{cy:.3f},{cz:.3f}) approach={side_name} "
            f"flange_y={cy + approach_sign*(APPROACH_OFFSET+TOOL_LENGTH_M):+.3f} "
            f"grip_tcp=({grip_tcp[0]:.3f},{grip_tcp[1]:.3f},{grip_tcp[2]:.3f})")

        # 1) APPROACH (컵 바깥, 수평 정렬) — 위 루프에서 IK 검증된 자세로 이동.
        #   현재(HOME) 관절을 seed 로 푼 손목 분기라 PTP 경로가 거의 직선이다.
        log.info("[1] Approach (수평 정렬, 현재관절 seed IK)")
        if not plan_and_execute(self.robot, self.arm, log,
                                state_goal=approach_state,
                                params=self.pilz_params):
            log.warn("[1] Pilz PTP(seed) 실패 — OMPL(seed) 재시도")
            if not plan_and_execute(self.robot, self.arm, log,
                                    state_goal=approach_state,
                                    params=self.ompl_params):
                return False

        if self.dry_run:
            log.warn("=== DRY-RUN: 접근 자세 도달. 30초 정지 ===")
            log.warn("  → 그리퍼 손가락이 컵 양옆(월드 X)을 향하면 OK")
            t0 = time.time()
            while rclpy.ok() and time.time() - t0 < 30.0:
                rclpy.spin_once(self, timeout_sec=0.1)
            return True

        # 2) INSERT (EE_Z 방향 수평 직진 → 그립 위치)
        insert_pose = self._flange_pose(grip_tcp, R_grip)
        insert_state = self.ik_state_with_current_seed(insert_pose)
        if insert_state is None:
            log.error("insert IK 실패 — 종료")
            return False
        #   PTP(joint-space) 우선. 과거 LIN(Cartesian 직선)이 손목 특이점을 지나
        #   joint_4 속도한계(31 vs 3.9)를 위반해 죽었다. seed IK 라 goal 관절이
        #   현재와 가까워 PTP 경로도 거의 직선이고 특이점을 회피한다.
        log.info("[2] Insert (현재관절 seed PTP, orientation 잠금)")
        if not plan_and_execute(self.robot, self.arm, log,
                                state_goal=insert_state, params=self.pilz_params):
            log.warn("[2] Pilz PTP 실패 — LIN 재시도")
            if not plan_and_execute(self.robot, self.arm, log,
                                    state_goal=insert_state,
                                    params=self.lin_params):
                return False

        # 3) CLOSE
        log.info("[3] Gripper CLOSE")
        self._gripper_move(GRIP_CLOSE_WIDTH, GRIP_FORCE)
        time.sleep(1.0)

        # 4) LIFT (수직 상승, 그립 자세 유지)
        #   수평 손목자세를 유지한 채 높이 들면 IK 가 안 풀릴 수 있다(손목 관절한계).
        #   LIFT_Z 상한부터 내려가며 IK 가 풀리는 가장 높은 z 를 채택. 실패해서
        #   바로 종료하던 과거 버그(회복동작 누락)를 방지.
        z_floor = grip_z + LIFT_Z_MIN_CLEAR
        lift_state = None
        lift_z = None
        z = LIFT_Z
        while z >= z_floor - 1e-6:
            cand = self._flange_pose(np.array([cx, cy, z]), R_grip)
            st = self.ik_state_with_current_seed(cand)
            if st is not None:
                lift_state, lift_z = st, z
                break
            log.warn(f"[4] lift IK 실패 z={z:.3f} — {LIFT_Z_STEP*100:.0f}cm 낮춰 재시도")
            z -= LIFT_Z_STEP
        if lift_state is None:
            log.error(f"lift IK 실패(모든 높이 {z_floor:.3f}~{LIFT_Z:.3f}) — 종료")
            return False
        lift_tcp = np.array([cx, cy, lift_z])
        log.info(f"[4] Lift → z={lift_z:.3f}")
        if not plan_and_execute(self.robot, self.arm, log,
                                state_goal=lift_state, params=self.pilz_params):
            return False
        time.sleep(LIFT_HOLD_SEC)

        # 4.5) PRE-BASE-YAW — joint_1 을 place 쪽으로 '더 바깥으로만' 스윙해 팔꿈치를
        #   작업영역 바깥으로 뺀다(fallen-cup pre-base-yaw 미러). 같은 쪽 접근이라 grab
        #   직후 이미 joint_1 이 충분히 바깥일 수 있는데, 이때 절대각으로 맞추면 오히려
        #   안쪽으로 당겨져 이후 carry seed-IK 가 큰 점프를 요구→거부된다(2026-06-15
        #   실로봇: -55°→-45° 로 당긴 뒤 carry IK 실패). 그래서 |joint_1| 이 목표보다
        #   작을 때만(=덜 바깥) 바깥으로 밀고, 이미 바깥이면 생략한다.
        cur_joints = self._read_current_joints()
        cur_j1 = float(cur_joints.get("joint_1", 0.0))
        desired_j1 = place_side * math.radians(PLACE_BASE_YAW_DEG)
        target_j1 = min(cur_j1, desired_j1) if place_side < 0 \
            else max(cur_j1, desired_j1)
        if abs(target_j1 - cur_j1) < math.radians(1.0):
            log.info(f"[4.5] pre-base-yaw 생략 (이미 바깥: joint_1="
                     f"{math.degrees(cur_j1):+.1f}°)")
        else:
            log.info(f"[4.5] pre-base-yaw joint_1 {math.degrees(cur_j1):+.1f}° → "
                     f"{math.degrees(target_j1):+.1f}° @ lift z={lift_z:.2f}m "
                     f"(팔꿈치 바깥 swing)")
            pre_yaw_joints = dict(cur_joints)
            pre_yaw_joints["joint_1"] = target_j1
            pre_yaw_state = RobotState(self.robot_model)
            pre_yaw_state.joint_positions = pre_yaw_joints
            pre_yaw_state.update()
            if not plan_and_execute(self.robot, self.arm, log,
                                    state_goal=pre_yaw_state, params=self.pilz_params):
                log.warn("[4.5] pre-base-yaw 실패 — swing 없이 그대로 진행")

        # 5) MOVE above PLACE — 컵을 든 채(아직 mouth-up, R_grip) PLACE Y-끝지점 위로.
        #   grasp 지점(컵이 한쪽으로 치우침)에서 바로 flip 하면 180° 후 flange 가 컵 너머
        #   바깥으로 나가 도달 불가일 수 있어, 먼저 place 위로 옮긴 뒤 flip 한다.
        #   수평 그리퍼는 lift 높이에서 옆(Y) 도달거리가 제한적이라 PLACE_Y_MAG 끝값이
        #   도달 불가일 수 있다 → 끝값부터 시작해 seed-IK 가 풀릴 때까지 안쪽으로 당긴다.
        #   채택된 place_y 는 이후 lower/retreat 에도 그대로 쓴다.
        place_y_full = place_side * PLACE_Y_MAG
        carry_state = None
        for shrink in (1.0, 0.75, 0.5, 0.25, 0.0):
            cand_y = place_y_full * shrink
            cand_tcp = np.array([place_x, cand_y, lift_z])
            cand_state = self.ik_state_with_current_seed(
                self._flange_pose(cand_tcp, R_grip))
            if cand_state is not None:
                place_y = cand_y
                place_high_tcp = cand_tcp
                carry_state = cand_state
                break
            log.warn(f"[5] place_y={cand_y:+.3f} seed IK 실패 — 안쪽으로 당겨 재시도")
        if carry_state is None:
            log.error("[5] 모든 place_y 후보 seed IK 실패 — 종료")
            return False
        log.info(f"[5] Move above PLACE=({place_x:.3f},{place_y:+.3f}) "
                 f"(mouth-up 유지, seed PTP)")
        if not plan_and_execute(self.robot, self.arm, log,
                                state_goal=carry_state, params=self.pilz_params):
            log.warn("[5] Pilz PTP(seed) 실패 — OMPL(seed) 재시도")
            if not plan_and_execute(self.robot, self.arm, log,
                                    state_goal=carry_state, params=self.ompl_params):
                return False

        # 6) MOUTH-DOWN FLIP — 순수 joint_6 180° 롤을 '관절공간'에서 직접 수행.
        #   회전이 순수 joint_6 임에도, 이를 Cartesian pose 로 만들어 IK(set_from_ik)로
        #   풀면 솔버가 'joint_6 만 증가한 해'가 아니라 손목이 뒤집힌 다른 분기(elbow-up/
        #   천장)를 반환한다 → 가드 거부 → OMPL 랜덤폴백이 컵을 들고 작업영역 밖으로
        #   휘두름(실로봇에서 joint_5 Δ=4rad 등으로 확인). IK 를 아예 거치지 말고 현재
        #   관절각의 joint_6 에만 ±180° 를 더한 목표 상태를 PTP 로 보내면 분기 모호성 0.
        #   flip 직후 컵 축은 vertical 에서 2·grip_tilt 만큼 접근쪽(±Y)으로 기운다
        #   (cup_axis = [0, approach_sign·sin2tilt, -cos2tilt]). release 직전 월드 X
        #   둘레로 작은 보정각만큼 돌려 컵을 더 세운다. 기운 방향이 approach_sign 부호이므로
        #   보정 회전은 -approach_sign 방향(컵을 vertical 쪽으로 당김).
        tilt_fix = -approach_sign * math.radians(PLACE_TILT_FIX_DEG)
        R_place = _rot_x(tilt_fix) @ R_grip @ _rot_z(math.pi)   # 결과 EE 자세(7·9 단계용)
        log.info(f"[6] place tilt-fix {PLACE_TILT_FIX_DEG:+.0f}° (컵 더 세움, 월드X)")
        base_joints = self._read_current_joints()
        if "joint_6" not in base_joints:
            log.error(f"[6] joint_6 키 없음: {list(base_joints)} — 종료")
            return False
        j6_0 = base_joints["joint_6"]
        # limit-safe 방향: 결과 |joint_6| 가 작아지는 쪽으로 180°(±2π 한계 안쪽 유지).
        flip_dir = -1.0 if j6_0 > 0.0 else 1.0
        log.info(f"[6] Mouth-down flip — joint_6 {math.degrees(j6_0):.0f}° → "
                 f"{math.degrees(j6_0 + flip_dir*math.pi):.0f}° "
                 f"(순수 관절 180°, IK 우회) {FLIP_STAGES}단계")
        for i in range(1, FLIP_STAGES + 1):
            frac = i / FLIP_STAGES
            stage_joints = dict(base_joints)
            stage_joints["joint_6"] = j6_0 + flip_dir * math.pi * frac
            stage_state = RobotState(self.robot_model)
            stage_state.joint_positions = stage_joints
            stage_state.update()
            log.info(f"  [6.{i}] joint_6={math.degrees(stage_joints['joint_6']):.0f}° "
                     f"(관절 PTP)")
            if not plan_and_execute(self.robot, self.arm, log,
                                    state_goal=stage_state, params=self.flip_params):
                log.error(f"  [6.{i}] flip 단계 PTP 실패 — 종료")
                return False

        # 7) LOWER — 엎은 컵(R_place, 수직 mouth-down)을 테이블에 안착. TCP=그립접점은
        #   엎으면 mouth rim 위로 (CUP_HEIGHT - 그립높이) 떨어진 지점 → 그만큼 띄워 내린다.
        #   place 의 ee_z 는 위로 tilt 만큼 기울어 flange 가 TCP 보다 아래 → flange 가
        #   PLACE_FLANGE_CLEAR 이상 바닥에서 떨어지게 release z 를 보정.
        place_z_seat = (TABLE_Z + (CUP_HEIGHT - GRIP_HEIGHT - self.grip_z_offset)
                        + PLACE_MARGIN_Z)
        flange_drop = TOOL_LENGTH_M * max(0.0, R_place[2, 2])   # ee_z 상향분 → flange 하강
        place_z = max(place_z_seat - PLACE_Z_DROP, PLACE_FLANGE_CLEAR + flange_drop)
        place_tcp = np.array([place_x, place_y, place_z])
        lower_pose = self._flange_pose(place_tcp, R_place)
        log.info(f"[7] Lower → z={place_z:.3f} (seat={place_z_seat:.3f}, "
                 f"flange_z={place_z - flange_drop:.3f}) (mouth-down 안착)")
        # seed IK 가 in-branch 해를 주면 PTP. 분기 점프로 거부/실패하면 현재(post-flip)
        # 자세에서 곧장 내리는 Cartesian 직선 LIN 으로 폴백 — 현재 관절 분기를 그대로
        # 따라가 IK 분기 점프가 없다(seed IK 가 None 이어도 LIN 은 pose_goal 로 시도).
        lower_state = self.ik_state_with_current_seed(lower_pose)
        lowered = False
        if lower_state is not None:
            lowered = plan_and_execute(self.robot, self.arm, log,
                                       state_goal=lower_state, params=self.pilz_params)
            if not lowered:
                log.warn("[7] seed PTP 실패 — Cartesian LIN 재시도")
        else:
            log.warn("[7] seed IK 거부(분기) — Cartesian LIN 직선하강 재시도")
        if not lowered:
            if not plan_and_execute(self.robot, self.arm, log,
                                    pose_goal=lower_pose, params=self.lin_params):
                log.error("[7] lower 실패(PTP·LIN 모두) — 종료")
                return False

        # 8) RELEASE — 엎은 컵 안착.
        log.info("[8] Gripper OPEN (release) → mouth-down 안착 완료")
        self._gripper_move(GRIP_OPEN_WIDTH, GRIP_FORCE)
        time.sleep(RELEASE_HOLD_SEC)

        # 9) RETREAT — 위로 빠져 컵에서 손 뗌.
        retreat_pose = self._flange_pose(place_high_tcp, R_place)
        retreat_state = self.ik_state_with_current_seed(retreat_pose)
        if retreat_state is not None:
            log.info("[9] Retreat 위로")
            plan_and_execute(self.robot, self.arm, log,
                             state_goal=retreat_state, params=self.pilz_params)
        log.info("=== mouth-down 배치 완료 ===")
        return True


def main(args=None):
    rclpy.init(args=args)
    node = PlaceMouthUpCupNode()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
