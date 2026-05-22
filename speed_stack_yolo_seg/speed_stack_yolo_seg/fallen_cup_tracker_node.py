#!/usr/bin/env python3
"""
fallen_cup_tracker_node.py

기존 fallen_cup_pose_node가 publish하는 인식 결과를 받아서 컵의
이동 속도를 추정하고, debug image에 속도 벡터를 시각화하는 노드.

이 노드는 잡기/세우기 로직과는 완전히 분리되어 있다. Phase 1은 추적/시각화만.

입력 토픽:
  /fallen_cup/grasp_pose   (geometry_msgs/PoseStamped)  - camera optical frame 3D
  /fallen_cup/pose2d       (std_msgs/Float32MultiArray) - pixel + yaw
  /fallen_cup/debug_image  (sensor_msgs/Image)          - YOLO overlay 이미지

출력 토픽:
  /fallen_cup/tracked_state         (std_msgs/Float32MultiArray)
    data layout (총 13개):
      [0]  pose2d_x_px           (현재 grip pixel x)
      [1]  pose2d_y_px           (현재 grip pixel y)
      [2]  pose3d_x_cam (m)      (camera optical frame X)
      [3]  pose3d_y_cam (m)
      [4]  pose3d_z_cam (m)      (depth)
      [5]  vel2d_x_px_per_s
      [6]  vel2d_y_px_per_s
      [7]  vel3d_x_cam_m_per_s
      [8]  vel3d_y_cam_m_per_s
      [9]  vel3d_z_cam_m_per_s
      [10] speed_3d_m_per_s      (3D 속도 크기)
      [11] yaw_cam_rad           (camera frame 컵 축 yaw)
      [12] age_sec               (마지막 detection 이후 경과 시간)

  /fallen_cup/tracker_debug_image   (sensor_msgs/Image)
    YOLO debug image 위에 속도 화살표 + speed 텍스트를 추가로 그린 것.

설계 노트:
- 속도는 deque에 저장된 최근 N개 (timestamp, 위치) 샘플의 finite difference 평균.
- 카메라 프레임 기준으로만 동작. base 프레임은 TF tree가 분리돼 있어서 안 함.
  base 속도가 필요해지면 calibration matrix로 static TF 추가하거나
  MoveItPy로 link_6 transform을 얻어서 곱하면 됨.
- 컵 mask가 일정 시간(예: 1초) 동안 안 보이면 추적을 reset.
"""

import math
import time
from collections import deque

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped

# 기존 모듈의 cv_bridge 대체 helper 재사용
from .fallen_cup_pose_node import imgmsg_to_cv2, cv2_to_imgmsg


# 데이터 인덱스 (publish_state에서 사용, downstream에서도 참고)
TS_PX_X       = 0
TS_PX_Y       = 1
TS_CAM_X      = 2
TS_CAM_Y      = 3
TS_CAM_Z      = 4
TS_VPX_X      = 5
TS_VPX_Y      = 6
TS_VCAM_X     = 7
TS_VCAM_Y     = 8
TS_VCAM_Z     = 9
TS_SPEED_3D   = 10
TS_YAW_CAM    = 11
TS_AGE_SEC    = 12
TS_LEN        = 13


class FallenCupTrackerNode(Node):
    def __init__(self):
        super().__init__("fallen_cup_tracker_node")

        # ---- parameters ----
        # 속도 추정 window 길이 (초). 너무 짧으면 잡음, 너무 길면 지연.
        self.declare_parameter("window_sec", 0.5)
        # 최근 N초 동안 detection 없으면 stale로 간주.
        self.declare_parameter("stale_timeout_sec", 1.0)
        # 속도 화살표 시각화 스케일 (pixel per cm/s for 2D, 화면 가독성용).
        self.declare_parameter("arrow_scale_px_per_cm_s", 3.0)
        # 입력 토픽 이름 (override 가능)
        self.declare_parameter("grasp_pose_topic", "/fallen_cup/grasp_pose")
        self.declare_parameter("pose2d_topic", "/fallen_cup/pose2d")
        self.declare_parameter("debug_image_in_topic", "/fallen_cup/debug_image")
        # 출력 토픽 이름
        self.declare_parameter("tracked_state_topic", "/fallen_cup/tracked_state")
        self.declare_parameter("tracker_debug_topic",
                               "/fallen_cup/tracker_debug_image")

        self.window_sec = float(self.get_parameter("window_sec").value)
        self.stale_timeout_sec = float(
            self.get_parameter("stale_timeout_sec").value
        )
        self.arrow_scale = float(
            self.get_parameter("arrow_scale_px_per_cm_s").value
        )

        grasp_pose_topic = str(self.get_parameter("grasp_pose_topic").value)
        pose2d_topic = str(self.get_parameter("pose2d_topic").value)
        debug_image_in_topic = str(
            self.get_parameter("debug_image_in_topic").value
        )
        tracked_state_topic = str(
            self.get_parameter("tracked_state_topic").value
        )
        tracker_debug_topic = str(
            self.get_parameter("tracker_debug_topic").value
        )

        # ---- buffers ----
        # 각 buffer: deque of (timestamp_sec, value_tuple)
        # 너무 오래된 샘플은 drop.
        self._buf_cam = deque()    # (t, np.array(3) in camera frame)
        self._buf_px  = deque()    # (t, np.array(2) pixels)

        self._last_cam = None      # 최신 camera 3D pose (np.array(3))
        self._last_px = None       # 최신 grip pixel (np.array(2))
        self._last_yaw_cam = 0.0   # 최신 yaw_rad in camera frame
        self._last_seen_t = 0.0    # 마지막 detection 수신 시각

        # ---- QoS: 카메라 스트림과 동일 (BEST_EFFORT) ----
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.create_subscription(
            PoseStamped, grasp_pose_topic, self._grasp_cb, qos
        )
        self.create_subscription(
            Float32MultiArray, pose2d_topic, self._pose2d_cb, qos
        )
        self.create_subscription(
            Image, debug_image_in_topic, self._image_cb, qos
        )

        self.state_pub = self.create_publisher(
            Float32MultiArray, tracked_state_topic, 10
        )
        self.debug_pub = self.create_publisher(
            Image, tracker_debug_topic, 10
        )

        # 주기적 state publish (인식 callback 빈도와 무관하게 일정 주기).
        self.create_timer(0.1, self._publish_state_tick)

        self.get_logger().info("fallen_cup_tracker_node started.")
        self.get_logger().info(f"  window_sec={self.window_sec}")
        self.get_logger().info(f"  stale_timeout_sec={self.stale_timeout_sec}")
        self.get_logger().info(
            f"  in: {grasp_pose_topic} | {pose2d_topic} | {debug_image_in_topic}"
        )
        self.get_logger().info(
            f"  out: {tracked_state_topic} | {tracker_debug_topic}"
        )

    # ---------- callbacks ----------
    def _grasp_cb(self, msg: PoseStamped):
        now = time.monotonic()
        pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ], dtype=np.float64)
        self._buf_cam.append((now, pos))
        self._last_cam = pos
        self._last_seen_t = now
        self._trim(self._buf_cam, now)

    def _pose2d_cb(self, msg: Float32MultiArray):
        # data layout (fallen_cup_pose_node 기준):
        #   [0..1] top_xy, [2..3] bottom_xy, [4..5] dir_xy, [6] yaw,
        #   [7..8] grip_xy, [9] conf, [10] top_w, [11] bot_w
        if len(msg.data) < 9:
            return
        now = time.monotonic()
        grip_px = np.array([msg.data[7], msg.data[8]], dtype=np.float64)
        self._buf_px.append((now, grip_px))
        self._last_px = grip_px
        self._last_yaw_cam = float(msg.data[6])
        self._last_seen_t = now
        self._trim(self._buf_px, now)

    def _image_cb(self, msg: Image):
        # 매 debug_image 도착 시 속도 화살표를 그려서 다시 publish.
        try:
            frame = imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"image decode failed: {e}")
            return

        vis = frame.copy()
        self._draw_overlay(vis)

        try:
            out = cv2_to_imgmsg(vis, encoding="bgr8")
            out.header = msg.header
            self.debug_pub.publish(out)
        except Exception as e:
            self.get_logger().warn(f"image publish failed: {e}")

    # ---------- helpers ----------
    def _trim(self, buf: deque, now: float):
        """buf에서 window_sec 보다 오래된 샘플을 앞에서 제거."""
        cutoff = now - self.window_sec
        while buf and buf[0][0] < cutoff:
            buf.popleft()

    def _estimate_vel(self, buf: deque):
        """
        deque의 첫 샘플과 마지막 샘플의 finite difference로 속도 계산.
        샘플이 2개 미만 또는 시간 간격 < 1ms 이면 None 반환.
        """
        if len(buf) < 2:
            return None
        t0, p0 = buf[0]
        t1, p1 = buf[-1]
        dt = t1 - t0
        if dt < 1e-3:
            return None
        return (p1 - p0) / dt

    def _is_stale(self, now: float) -> bool:
        if self._last_seen_t == 0.0:
            return True
        return (now - self._last_seen_t) > self.stale_timeout_sec

    # ---------- state publish ----------
    def _publish_state_tick(self):
        now = time.monotonic()
        data = [0.0] * TS_LEN

        if self._last_px is not None:
            data[TS_PX_X] = float(self._last_px[0])
            data[TS_PX_Y] = float(self._last_px[1])
        if self._last_cam is not None:
            data[TS_CAM_X] = float(self._last_cam[0])
            data[TS_CAM_Y] = float(self._last_cam[1])
            data[TS_CAM_Z] = float(self._last_cam[2])

        vpx = self._estimate_vel(self._buf_px)
        if vpx is not None:
            data[TS_VPX_X] = float(vpx[0])
            data[TS_VPX_Y] = float(vpx[1])

        vcam = self._estimate_vel(self._buf_cam)
        if vcam is not None:
            data[TS_VCAM_X] = float(vcam[0])
            data[TS_VCAM_Y] = float(vcam[1])
            data[TS_VCAM_Z] = float(vcam[2])
            data[TS_SPEED_3D] = float(np.linalg.norm(vcam))

        data[TS_YAW_CAM] = self._last_yaw_cam
        data[TS_AGE_SEC] = (
            (now - self._last_seen_t) if self._last_seen_t > 0.0 else 0.0
        )

        msg = Float32MultiArray()
        msg.data = data
        self.state_pub.publish(msg)

    # ---------- debug overlay ----------
    def _draw_overlay(self, image_bgr):
        """YOLO debug image 위에 속도 화살표 + 텍스트 오버레이."""
        now = time.monotonic()
        stale = self._is_stale(now)

        # 상태 텍스트
        if stale:
            status = "STALE (no detection)"
            status_color = (0, 0, 200)
        else:
            status = "TRACKING"
            status_color = (0, 200, 0)
        cv2.putText(
            image_bgr, status, (20, image_bgr.shape[0] - 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2,
        )

        # 위치/속도가 있을 때만 화살표
        if stale or self._last_px is None:
            return
        vpx = self._estimate_vel(self._buf_px)
        vcam = self._estimate_vel(self._buf_cam)

        cur_px = tuple(np.round(self._last_px).astype(int))

        # 픽셀 속도 화살표 (감각적인 시각화).
        # vpx 는 px/s. 화살표 길이는 안정성을 위해 적당히 스케일.
        if vpx is not None:
            speed_px_per_s = float(np.linalg.norm(vpx))
            if speed_px_per_s > 1.0:  # 1 px/s 이하는 정지로 간주
                # 0.3초 후 예상 위치를 화살표 끝으로 (직관적).
                horizon = 0.3
                end = self._last_px + vpx * horizon
                end_i = tuple(np.round(end).astype(int))
                cv2.arrowedLine(
                    image_bgr, cur_px, end_i, (0, 255, 255), 2,
                    tipLength=0.25,
                )

        # 텍스트: 카메라 frame 기준 3D speed (cm/s) 가 의미 있음
        speed_3d_cm_s = 0.0
        if vcam is not None:
            speed_3d_cm_s = float(np.linalg.norm(vcam)) * 100.0
        cv2.putText(
            image_bgr,
            f"v_3d = {speed_3d_cm_s:5.1f} cm/s",
            (20, image_bgr.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )


def main(args=None):
    rclpy.init(args=args)
    node = FallenCupTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
