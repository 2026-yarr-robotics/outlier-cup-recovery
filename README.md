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
git clone https://github.com/doosan-robotics/doosan-robot2.git

cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## 실행

세 개의 터미널이 필요합니다.

**1) Doosan bringup + MoveIt**
```bash
ros2 launch dsr_bringup2 dsr_bringup2_moveit.launch.py
```

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
```

## 주요 파라미터

`stand_fallen_cup`
- `mode` — `drop` | `place`
- `dry_run` — `true`면 approach까지만 가서 그리퍼 yaw 정렬 확인 (close/lift 없음)
- `cup_yaw_override_deg` — NaN이 아니면 인식 yaw 무시하고 강제 값 사용
- `sim` — HW 없이 MoveIt virtual에서 시뮬레이션

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
