#!/usr/bin/env python3
"""
outlier_cup_recovery.py

fallen-cup-recovery(stand_fallen_cup) 와 mouth-up-cup recovery(place_mouth_up_cup)
를 단일 프로세스·단일 MoveItPy 로 묶는 통합 오케스트레이터.

런치 하나로 실행하면:
  1) fallen-cup 을 먼저 전부 처리 (multi-cup 루프: base_link 최근접 우선),
  2) 이어서 mouth-up-cup 이 더 없을 때까지 반복 처리한다.
분류는 상류 vision 이 끝냄 — fallen 은 /fallen_cup/* 토픽군, mouth-up 은
/mouth_up_cup/grasp_pose 만 구독한다. 여기서는 두 스킬 노드를 같은 프로세스에
띄워 라우팅 순서만 강제한다.

아키텍처(권장안): 비싼 자원(MoveItPy, RG 그리퍼)은 fallen 노드가 1개만 만들고
mouth-up 노드에 주입한다. 두 스킬 노드는 각자 rclpy Node 로 유지돼 자기
subscription/param/logger/spin 을 그대로 보존한다. launch 가 넘긴 union 파라미터는
/** 와일드카드로 두 노드에 동일하게 적용되며, 각 노드는 자기가 declare 한 것만 읽는다.

종료코드 정책(서버 task 상태 전파용):
  recovered ≥ 1  또는  detected == 0(치울 것 없음)  → 성공(exit 0)
  detected > 0  인데  recovered == 0                 → 실패(exit 1)
"""

import sys

import rclpy

from dsr_practice.stand_fallen_cup import StandFallenCupNode
from dsr_practice.place_mouth_up_cup import (
    PlaceMouthUpCupNode,
    RECOVER_DONE,
    RECOVER_NONE,
    RECOVER_FAIL,
)

# mouth-up 단계가 무한 루프에 빠지지 않도록 하는 안전 상한.
# 정상이면 RECOVER_NONE(처리할 컵 0개) 으로 빠져나온다.
MOUTH_UP_MAX_ITERATIONS = 10


def main(args=None):
    rclpy.init(args=args)

    fallen = None
    mouthup = None

    # 통계
    fallen_detected = False
    fallen_recovered = 0
    mouthup_detected = False
    mouthup_recovered = 0

    try:
        # ── 비싼 자원은 fallen 노드가 1개만 생성 → mouth-up 에 주입 ──
        fallen = StandFallenCupNode()
        mouthup = PlaceMouthUpCupNode(
            moveit_py=fallen.robot,
            gripper=fallen.gripper,
        )
        log = fallen.get_logger()

        # fallen 은 항상 최근접·multi-cup 루프로 돌린다(사용자 확정: fallen 먼저,
        # 같은 종류 안에서 base_link 최근접 우선). sim 모드면 multi 루프를 타지
        # 않으므로 그땐 단일-컵 경로로 동작(fallen.run 내부 분기).
        if not fallen.sim:
            fallen.multi_cup = True

        # ── 1단계: fallen-cup 전부 처리 ──
        log.info("[outlier] ===== 1단계: fallen-cup recovery =====")
        fallen.run()
        fallen_detected = bool(fallen.saw_candidate)
        fallen_recovered = int(fallen.recovered_count)
        log.info(
            f"[outlier] fallen 단계 종료: detected={fallen_detected} "
            f"recovered={fallen_recovered}"
        )

        # ── 2단계: mouth-up-cup 이 없을 때까지 반복 처리 ──
        log.info("[outlier] ===== 2단계: mouth-up-cup recovery =====")
        for it in range(MOUTH_UP_MAX_ITERATIONS):
            status = mouthup.run()
            if status == RECOVER_NONE:
                log.info("[outlier] mouth-up 처리할 컵 없음 → 단계 종료")
                break
            if status == RECOVER_DONE:
                mouthup_detected = True
                mouthup_recovered += 1
                log.info(
                    f"[outlier] mouth-up {mouthup_recovered}개 처리 완료 "
                    f"— 다음 컵 재탐색"
                )
                continue
            # RECOVER_FAIL
            mouthup_detected = True
            log.error(
                "[outlier] mouth-up 컵 감지했으나 처리 실패 — 단계 중단"
            )
            break
        else:
            log.warn(
                f"[outlier] mouth-up 최대 반복({MOUTH_UP_MAX_ITERATIONS}) 도달 "
                "— 단계 중단"
            )

    finally:
        if mouthup is not None:
            mouthup.destroy_node()
        if fallen is not None:
            fallen.destroy_node()
        rclpy.shutdown()

    detected = fallen_detected or mouthup_detected
    recovered = fallen_recovered + mouthup_recovered

    # 종료코드 정책: 하나라도 세웠거나 애초에 치울 게 없으면 성공.
    # 감지는 됐는데 하나도 못 세웠으면 실패.
    ok = (recovered >= 1) or (not detected)

    summary = (
        f"[outlier_cup_recovery] detected={detected} "
        f"(fallen={fallen_detected}, mouth-up={mouthup_detected}) "
        f"recovered={recovered} "
        f"(fallen={fallen_recovered}, mouth-up={mouthup_recovered})"
    )
    if ok:
        print(summary + " → 성공(exit 0)", file=sys.stderr)
    else:
        print(
            summary + " → 실패(exit 1, 서버에 failed 통보)",
            file=sys.stderr,
        )
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
