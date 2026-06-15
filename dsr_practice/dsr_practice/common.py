#!/usr/bin/env python3
"""
common.py

cup-manipulation 스킬들(stand_fallen_cup, place_mouth_up_cup, …)이 공유하는
robot/geometry/motion 범용 유틸. 특정 스킬(누운 컵 세우기 / mouth-up 뒤집기)에
종속되지 않는 것들만 둔다. 컵 치수·작업영역 PLACE 좌표·피라미드 회피 등
스킬별로 튜닝되는 상수는 각 스킬 파일에 그대로 남긴다.
"""

import math

import numpy as np

from geometry_msgs.msg import PoseStamped


# ─────────────────────────────────────────────────────────
#  로봇/그룹 기본 (모든 스킬 공통)
# ─────────────────────────────────────────────────────────
GROUP_NAME = "manipulator"
BASE_FRAME = "base_link"
EE_LINK    = "link_6"

# 고정 원점(HOME) 자세. 2026-06-09 사용자가 실로봇으로 직접 만든 자세를
# /joint_states 에서 캡처한 라디안 원값. fallen-cup / mouth-up-cup 두 task 공유.
HOME_JOINTS = {
    "joint_1": -0.049552422016859055,
    "joint_2": -0.26035377383232117,
    "joint_3": 1.5442062616348267,
    "joint_4": -0.05165897682309151,
    "joint_5": 1.8779057264328003,
    "joint_6": 1.5116537809371948,
}

# 안전 작업 영역 (m, base_link 기준)
SAFE_X_MIN = 0.0
SAFE_Y_MIN = -0.30
SAFE_Y_MAX =  0.30
SAFE_Z_MIN =  0.05   # link_6 flange 기준 최저 안전 z

# DOWN_ORI 쿼터니언 (0,1,0,0) = Y축 180° 회전 → 회전행렬로 미리 캐시
R_DOWN = np.array([
    [-1.0, 0.0,  0.0],
    [ 0.0, 1.0,  0.0],
    [ 0.0, 0.0, -1.0],
])


# ─────────────────────────────────────────────────────────
#  작업영역 클램프 / 모션 실행
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
