# fallen-cup-recovery

Doosan M0609 + RG2 그리퍼 + Intel RealSense 카메라로 **넘어진 컵을 인식하여 잡고 세우는** ROS 2 (Humble) 시스템.

YOLOv26-seg가 카메라 영상에서 `fallen-cup` / `upright-cup` 두 클래스를 분리해 segmentation하고,
`fallen-cup` mask에서만 yaw 방향벡터를 추출하여 그리퍼가 컵 머리 옆을 잡고 들어 올린 뒤 세웁니다.
<p align="left">
  <img src="assets/yaw_vector.png" alt="yaw vector example" width="400"/>
</p>

## Demo Video
[![Fallen Cup Recovery Demo](https://img.youtube.com/vi/p8Zon8WmvEE/0.jpg)](https://www.youtube.com/watch?v=p8Zon8WmvEE)

## 구성

두 ROS 2 패키지가 한 쌍으로 동작합니다.

### `speed_stack_yolo_seg` (인식 측)
- `/camera/color/image_raw`, `/camera/aligned_depth_to_color/image_raw` 구독
- 학습 클래스: `fallen-cup`, `upright-cup`
- [학습 코드 참고](https://github.com/2026-yarr-robotics/vision-YOLO/tree/main/hand-eye-view)

- 출력 토픽
  - `/fallen_cup/pose2d` (`std_msgs/Float32MultiArray`) — top/bottom 픽셀, 방향벡터, yaw, grip 픽셀, conf, 폭
  - `/fallen_cup/grasp_pose` (`geometry_msgs/PoseStamped`, camera optical frame) — 3D grasp point
  - `/fallen_cup/debug_image` (`sensor_msgs/Image`) — 디버그 오버레이
- 방향벡터는 `target_class_name=fallen-cup` mask에서만 추출 (upright-cup mask는 자동 제외)

### `dsr_practice` (로봇 제어 측)
- `stand_fallen_cup` 노드: 위 토픽 구독 → MoveIt + RG2로 컵 머리 옆을 잡고 들어 올린 뒤
  - `mode:=drop` — 그 자리에서 컵 떨어뜨림
  - `mode:=place` — 손목 pitch 회전으로 세워서 작업공간 다른 위치에 내려놓음
- `multi_cup:=true` — 한 프레임에 여러 fallen cup 이 있으면 가까운 순서로 순차 처리
- `pyramid_avoid:=true` (기본 ON) — 쌓인 컵 피라미드를 MoveIt collision object 로 등록해
  recovery 궤적이 그 영역을 회피. ([아래 참고](#피라미드-영역-회피-place-모드))

## 외부 의존성 (별도 설치 필요)

본 리포지토리에 포함되지 않은 항목:

- ROS 2 Humble
- [Doosan Robotics 공식 패키지](https://github.com/doosan-robotics/doosan-robot2) — `dsr_bringup2`, `dsr_controller2`, `dsr_moveit2`, `dsr_msgs2`, `dsr_moveit_config_m0609` 등
- MoveIt 2 (`moveit_py` 포함)
- `realsense2_camera` (Intel RealSense ROS 2 wrapper)
- Python: `ultralytics`, `torch`, `opencv-python`, `pymodbus`, `numpy`

## 빌드

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# 본 리포지토리
git clone https://github.com/2026-yarr-robotics/fallen-cup-recovery.git

# Doosan 공식 패키지 (같은 src/ 아래)
# 동작 검증 버전: humble 브랜치, release 20260324 (commit ec92425)
git clone -b humble https://github.com/doosan-robotics/doosan-robot2.git

cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## 실행

세 개의 터미널이 필요합니다.

**1) Doosan bringup + MoveIt**
```bash
# 실기(real) — host 는 로봇 컨트롤러 IP 로 교체
ros2 launch dsr_bringup2 dsr_bringup2_moveit.launch.py \
    mode:=real model:=m0609 host:=192.168.1.100
```
> `dsr_bringup2` 등 bringup 패키지는 두산 공식 코드(위 clone)를 **수정 없이** 그대로 사용합니다.
> 본 리포지토리에는 로봇 제어 로직(`dsr_practice/stand_fallen_cup`)만 포함됩니다.

**2) RealSense + YOLO 인식 노드**
```bash
ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
ros2 launch speed_stack_yolo_seg fallen_cup_pose.launch.py \
    weights_path:=<path/to/best.pt> \
    use_depth:=true \
    imgsz:=1280 \
    conf:=0.70 \
    target_class_name:=fallen-cup
```

**3) 로봇 제어 노드**
```bash
ros2 launch dsr_practice stand_fallen_cup.launch.py mode:=drop
# 또는 mode:=place

# 멀티컵 + 피라미드 회피까지 한 번에 (place 모드)
ros2 launch dsr_practice stand_fallen_cup.launch.py \
    mode:=place multi_cup:=true pyramid_avoid:=true \
    pyramid_config_url:=<피라미드 config GET 엔드포인트>
```
> `pyramid_avoid` / `place_plus_y_auto_swing` 은 기본 ON 이라 생략해도 됩니다.
> 피라미드 회피는 `/stack` 점유 슬롯 + `pyramid_config_url` 응답이 모두 있을 때만 실제로
> 동작하며, 없으면 graceful no-op 입니다. ([피라미드 영역 회피](#피라미드-영역-회피-place-모드))

## 실행 파이프라인

`stand_fallen_cup` 노드 1회 실행은 다음 순서로 진행됩니다.

```
초기화
  ├─ controller 연결 대기 → HOME 이동 (Pilz PTP, 실패 시 OMPL)
  ├─ 그리퍼 부피를 link_6 에 attach (이후 모든 plan 이 그리퍼 충돌 회피)
  ├─ 피라미드 점유 슬롯을 collision object 로 등록 (pyramid_avoid)
  └─ 그리퍼 open

컵 처리  (single = 1회 / multi_cup = cup 마다 반복)
  ├─ Sense     : /fallen_cup/* 샘플 수집 → 클러스터링 → 가장 가까운 cup 선정
  ├─ 정상 컵을 collision object 로 등록 (avoid_upright_cups)
  ├─ Preflight : approach/descend IK 사전 검사 — 해 없으면 skip + /fallen_cup/unreachable
  ├─ PLACE 결정 : single = PLACE_X/Y(또는 override) · multi = 빈 안전지점 선택(또는 제자리)
  └─ Pick & handle
       [1] Approach   컵 위로 이동 (Pilz)
       [2] Descend    IK current-seed 잠금(+랜덤 seed fallback)으로 수직 하강
       [3] Close      그리퍼를 닫아 컵 머리 옆을 잡음
       [4] Lift       수직 상승 (LIFT_Z)
       └─ mode=place : ±Y auto-swing 판단 → base swing → twist 분리 → 기울여 세움 → release → retreat
          mode=drop  : 들어 올린 채 대기 후 release

마무리
  ├─ HOME 복귀 (OMPL, 실패 시 Pilz PTP) + 시작/종료 자세 비교 검증
  └─ 종료코드로 성공/실패 전파 (0 = 성공, 1 = 실패 → 통합 task 상태)
```

- **`multi_cup`**: 매 iteration 마다 HOME 으로 복귀해 카메라 시야를 확보한 뒤 재sense →
  다음 cup 처리. 더 이상 후보가 없거나 `multi_cup_max_iterations` 에 도달하면 종료합니다.
- **단계별 세부**: 실패 전파·사전 도달성 검사는
  [실패 보고 & 사전 도달성 검사](#실패-보고--사전-도달성-검사),
  ±Y auto-swing·피라미드 회피는 [피라미드 영역 회피](#피라미드-영역-회피-place-모드) 참고.

## 피라미드 영역 회피 (place 모드)

`place` 모드로 세운 컵을 작업공간에 내려놓을 때, 옆에 **이미 쌓여 있는 컵 피라미드**를
건드리지 않도록 회피합니다.

- **장애물 등록**: 피라미드 점유 슬롯(`/stack` 토픽의 occupied slot)을 MoveIt
  `CollisionObject` 로 등록. 피라미드 center/degree 는 `pyramid_config_url` API polling
  으로 동기화하여 실제 물리 위치와 일치시킵니다. MoveIt 이 이 충돌체를 피하는 궤적만
  계획하므로, 회피 불가 시엔 실패-안전(plan fail)으로 빠집니다.
- **활성 조건**: `pyramid_avoid:=true`(기본) + `/stack` 점유 슬롯 + `pyramid_config_url`
  GET 응답, **셋 다** 있어야 실제 회피가 동작합니다. vision 스택/서버가 미가용이거나
  점유 슬롯이 없으면 **graceful no-op**(장애물 0개, 기존 동작과 동일)으로 떨어집니다.

### place 위치 / ±Y auto-swing

세운 컵은 피라미드에서 멀고 base_link 와 가까운 +Y 작업영역(`PLACE = (0.30, 0.10)`)에
내려놓습니다. 넘어진 컵의 누운 방향(wide 면이 +Y/-Y)에 따라 팔꿈치(link_4/5/6)가
피라미드를 쓸지 않도록 base swing 전략을 자동 적용합니다.

- `place_plus_y_auto_swing:=true` (기본 ON) — `sin(cup_yaw)` 부호로 ±Y 케이스를 자동 감지.
  - **+Y** (wide 면이 +Y, yaw≈+90°): side=left / base_yaw=+60° / tilt=25° 로 swing → +Y 쪽에서 접근.
  - **-Y** (wide 면이 -Y, yaw≈-90°): 대칭으로 side=right / base_yaw=-60° 로 swing.
  - +Y 미감지(다른 컵)면 no-op 이라 항상 켜 둬도 안전. 한 명령으로 ±Y 양쪽 케이스 처리.
- `place_x` / `place_y` — PLACE 좌표를 리빌드 없이 override(base_link, m). `nan`(기본)이면 모듈 상수 사용.

## 실패 보고 & 사전 도달성 검사

### 종료코드로 실패 전파 (통합)

`stand_fallen_cup` 은 recovery 결과를 **프로세스 종료코드**로 알립니다 — 성공 `0`, 실패 `1`.
통합(`cup-stack-integration`)은 recovery 를 1회성 launch 태스크로 실행하고 서버가 그
종료코드를 태스크 상태(`idle`=성공 / `failed`=실패)로 반영하므로, 컵을 세우지 못하면
상위 파이프라인이 실패를 인지할 수 있습니다.

- **노드**: HOME 이동 / 인식 / IK / 집기 중 하나라도 실패하면 종료코드 `1`.
  `multi_cup` 모드는 **실제로 세운 컵이 0개**일 때만 실패로 봅니다. (마지막 HOME 복귀
  실패는 컵이 이미 섰으므로 성공으로 간주하고 경고만 남깁니다.)
- **launch**: ROS 2 launch 는 자식 노드가 비정상 종료해도 `ros2 launch` 종료코드를 `0`
  으로 두기 때문에, `stand_fallen_cup.launch.py` 가 노드 종료코드를 감지해 non-zero 면
  launch 자체를 실패시켜 종료코드를 그대로 전파합니다.

### 사전 도달성 검사 (`preflight_reach_check`, 기본 ON)

컵을 집기 **전에** approach/descend 자세의 IK 해를 미리 풀어봐서, 해가 없을 컵은
**집지 않고 건너뜁니다**. 건너뛴 컵은 blacklist 에 추가(다음 sense 에서도 제외)하고
`/fallen_cup/unreachable` (`std_msgs/String`, JSON `{x, y, z, cup_yaw_deg, reason, stamp}`)
로 통보합니다.

- 베이스 안쪽 경계처럼 KDL 단일-seed IK 가 해가 있어도 수렴하지 못하는 경우를 위해,
  IK 는 1차 실패 시 base seed 주변 랜덤 seed 로 재시도하는 fallback 을 거칩니다
  (wrist-flip 방지를 위해 base seed 대비 관절 편차가 작은 해를 우선 채택).
- 사전 검사도 이 fallback 을 거치므로 '집어보면 풀렸을 컵' 을 도달불가로 오판하지 않습니다.

## 주요 파라미터

`stand_fallen_cup`
- `mode` — `drop` | `place`
- `dry_run` — `true`면 approach까지만 가서 그리퍼 yaw 정렬 확인 (close/lift 없음)
- `cup_yaw_override_deg` — NaN이 아니면 인식 yaw 무시하고 강제 값 사용
- `sim` — HW 없이 MoveIt virtual에서 시뮬레이션
- `multi_cup` — `true`면 한 프레임의 여러 fallen cup 을 가까운 순서로 순차 처리
- `pyramid_avoid` — `true`(기본)면 쌓인 피라미드 영역을 collision object 로 회피
- `pyramid_config_url` — 피라미드 center/degree 를 GET 으로 받는 API 엔드포인트 (빈 문자열이면 polling 끔)
- `pyramid_stack_topic` — 피라미드 슬롯 점유 토픽(기본 `/stack`)
- `place_plus_y_auto_swing` — `true`(기본)면 ±Y 누운 케이스 자동 감지 후 base swing
- `place_x` / `place_y` — PLACE 좌표 override (base_link, m; `nan`이면 모듈 상수)
- `preflight_reach_check` — `true`(기본)면 집기 전 approach/descend IK 를 미리 검사,
  해가 없으면 그 컵을 skip + `/fallen_cup/unreachable` 통보

`fallen_cup_pose_node`
- `target_class_name` — 기본 `fallen-cup`. 빈 문자열이면 클래스 필터링 끔
- `mode` — `auto` | `silhouette` | `two_face`
- `use_depth` — true일 때만 3D grasp_pose publish

## 카메라 캘리브레이션 (Hand-Eye)

`stand_fallen_cup` 가 카메라 좌표계의 grasp_pose 를 로봇 base 좌표계로 변환할 때 쓰는
**그리퍼-카메라 변환행렬** `T_gripper2camera` 를 직접 측정하기 위한 절차. 코드는
[`Calibration_Tutorial/`](Calibration_Tutorial) 에 있습니다.

**필요 자재**
- 7×5 내부 코너 체커보드 (정사각형 한 변 25 mm).
- Doosan bringup + RealSense 가 같은 머신에서 떠 있어야 함.

**1) 데이터 수집** — `data_recording.py`
티치펜던트로 다양한 자세 (≥15개 권장) 를 잡고 각 자세에서 실행. 매 호출마다 현재
robot pose + RealSense 컬러 이미지를 `data/` 에 함께 저장.
```bash
cd Calibration_Tutorial
python3 data_recording.py
```

**2) 캘리브 계산** — 카메라 마운팅 방식에 따라 둘 중 하나
- **eye-in-hand** (카메라가 그리퍼/링크에 장착 — 본 프로젝트 기본): `handeye_calibration.py`
- **eye-to-hand** (카메라가 작업공간 고정): `eye2hand_calibration.py`
```bash
python3 handeye_calibration.py   # 또는 eye2hand_calibration.py
```
실행 결과로 `T_gripper2camera.npy` (4×4, mm 단위) 가 생성됨.

**3) 검증** — `test.py`
캘리브 결과로 카메라 좌표 → base 좌표 역변환을 수행, 알려진 마커/물체 위치와 비교.
```bash
python3 test.py
```

**4) 적용**
검증 통과 시 `T_gripper2camera.npy` 를 `dsr_practice/config/` 로 복사. `stand_fallen_cup`
런치 시 자동으로 로드.
```bash
cp T_gripper2camera.npy ../dsr_practice/dsr_practice/config/T_gripper2camera.npy
```

> 본 리포지토리에 포함된 `T_gripper2camera.npy` 는 본 저자 셋업의 결과. 카메라 마운팅
> 위치/각도가 다르면 동작이 어긋나므로 **본인 로봇에서 재캘리브 권장**.

## 자산 및 캘리브레이션 참고사항

- **`dsr_practice/config/T_gripper2camera.npy`**
  본 저자 셋업의 핸드아이 캘리브 결과(4×4, 단위 mm). **참고용으로만 사용하세요.**
  카메라 마운팅과 RG2 장착 위치가 다르면 동작이 어긋나므로, **본인 로봇에서 재캘리브 강력 권장**합니다.
- **`speed_stack_yolo_seg/weights/best.pt`**
  `.gitignore`에 의해 repo에서 제외됨. YOLOv26-seg를 `fallen-cup` / `upright-cup` 두 클래스로 직접 학습한 뒤
  `weights_path` 런치 인자로 경로를 지정하세요.

## 라이선스

TODO
