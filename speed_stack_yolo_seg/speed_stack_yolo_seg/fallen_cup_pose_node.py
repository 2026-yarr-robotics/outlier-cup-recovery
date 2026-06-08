#!/usr/bin/env python3

import math
import time
import cv2
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from geometry_msgs.msg import PoseStamped, PoseArray, Pose

from ultralytics import YOLO


# cv_bridge 대체 (numpy 1.x↔2.x ABI 충돌 회피).
# cv_bridge의 C extension은 numpy 1.x 기준으로 컴파일돼 있어서 numpy 2.x 환경에서
# `_ARRAY_API not found` 에러로 죽음. 직접 numpy로 변환하면 의존성 제거.
def imgmsg_to_cv2(msg, desired_encoding="passthrough"):
    """sensor_msgs/Image → cv2/numpy. desired_encoding은 bgr8 또는 passthrough."""
    h, w = msg.height, msg.width
    enc = msg.encoding

    if enc == "16UC1":
        arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)
    elif enc == "32FC1":
        arr = np.frombuffer(msg.data, dtype=np.float32).reshape(h, w)
    elif enc == "mono8":
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w)
    elif enc in ("bgr8", "rgb8"):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 3)
    elif enc in ("bgra8", "rgba8"):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, w, 4)
    else:
        raise ValueError(f"Unsupported encoding: {enc}")

    if desired_encoding == "passthrough" or desired_encoding == enc:
        return arr.copy()

    if desired_encoding == "bgr8":
        if enc == "rgb8":
            return arr[:, :, ::-1].copy()
        if enc == "bgra8":
            return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        if enc == "rgba8":
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        if enc == "mono8":
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

    raise ValueError(f"Cannot convert {enc} → {desired_encoding}")


def cv2_to_imgmsg(image, encoding="bgr8"):
    """cv2/numpy → sensor_msgs/Image. bgr8/rgb8/mono8 지원."""
    msg = Image()
    h, w = image.shape[:2]
    msg.height = h
    msg.width = w
    msg.encoding = encoding
    msg.is_bigendian = 0
    if encoding in ("bgr8", "rgb8"):
        msg.step = w * 3
    elif encoding == "mono8":
        msg.step = w
    else:
        raise ValueError(f"Unsupported encoding: {encoding}")
    msg.data = image.tobytes()
    return msg


def as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ["true", "1", "yes", "y"]
    return bool(value)


class FallenCupPoseNode(Node):
    """
    Output:
      /fallen_cup/debug_image:
        방향벡터, top center, bottom center, grip point가 그려진 image

      /fallen_cup/pose2d:
        Float32MultiArray
        data = [
          top_x_px, top_y_px,
          bottom_x_px, bottom_y_px,
          dir_x, dir_y,
          yaw_rad,
          grip_x_px, grip_y_px,
          confidence,
          top_width_px,
          bottom_width_px
        ]

      /fallen_cup/grasp_pose:
        PoseStamped
        use_depth:=true일 때만 3D grasp point publish
        frame_id는 입력 color image의 frame_id를 그대로 사용
    """

    def __init__(self):
        super().__init__("fallen_cup_pose_node")

        # -----------------------------
        # Parameters
        # -----------------------------
        self.declare_parameter("weights_path", "")
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")

        self.declare_parameter("debug_image_topic", "/fallen_cup/debug_image")
        self.declare_parameter("pose2d_topic", "/fallen_cup/pose2d")
        self.declare_parameter("grasp_pose_topic", "/fallen_cup/grasp_pose")
        # 신규 multi-cup 토픽 (Phase 1)
        self.declare_parameter("cups_pose2d_topic", "/fallen_cup/cups_pose2d")
        self.declare_parameter("cups_grasp_poses_topic", "/fallen_cup/cups_grasp_poses")

        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("iou", 0.45)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("half", False)

        # auto: mask가 2개 이상이면 top/bottom face 방식 시도, 아니면 silhouette PCA 방식
        # silhouette: 넘어진 컵 전체 mask 하나로 방향 추정
        # two_face: 작은 원/큰 원 두 mask로 방향 추정
        self.declare_parameter("mode", "auto")

        # 방향벡터를 추출할 대상 클래스 이름.
        # YOLO 모델이 fallen-cup / upright-cup 두 클래스를 학습한 경우,
        # 넘어진 컵에 대해서만 yaw를 뽑아야 그리퍼가 올바르게 잡을 수 있다.
        # 빈 문자열("")로 두면 클래스 필터링을 끈다(이전 동작과 호환).
        self.declare_parameter("target_class_name", "fallen-cup")

        self.declare_parameter("use_depth", False)

        # top_center에서 bottom 방향으로 얼마나 안쪽을 잡을지.
        # speed-stack recovery 용으로 컵의 narrow 끝 부근을 잡아 들어 올린 뒤
        # 회전시켜 wide 면을 바닥으로 향하게 세움. 가까운 끝(narrow)을 잡아야
        # 회전 시 그리퍼/로봇이 바닥에 안 부딪히고 안정적으로 매달림.
        # 기본 0.015 m = 1.5 cm. 컵이 더 길거나 narrow 끝 직경이 크면 키워야.
        self.declare_parameter("grip_offset_m", 0.015)

        # depth를 안 쓸 때 pixel offset 계산용.
        # 실제 컵의 작은 원 지름을 재서 넣는 것을 추천.
        # 예: 작은 원 지름이 4.5cm면 0.045
        self.declare_parameter("top_diameter_m", 0.045)

        # 직접 pixel/meter를 알고 있으면 넣기. 모르면 0.
        self.declare_parameter("pixels_per_meter", 0.0)

        self.declare_parameter("min_mask_area", 300.0)
        self.declare_parameter("min_pair_distance_px", 20.0)
        self.declare_parameter("max_pair_distance_px", 10000.0)
        # two_face pair 가 "한 컵의 narrow/wide 두 face" 인지 판단하는 직경 비 임계값.
        # narrow≈4.5cm vs wide≈7cm 인 컵이면 비≈1.5. 별개 두 컵끼리는 비≈1.0 이라
        # 1.3 정도로 자르면 cross-cup pairing 을 막을 수 있다.
        self.declare_parameter("min_pair_diameter_ratio", 1.3)

        # 축 길이 sanity 필터.
        # 넘어진 컵의 top→bottom 축 실제 길이는 거의 일정하다. 잘못된 mask
        # (두 컵 병합 / cross-cup pairing)는 축이 비정상적으로 길어지므로,
        # depth로 축 양 끝을 3D deproject해서 실제 길이(m)를 재고 정상 밴드
        # [expected ± tol] 밖이면 그 estimate를 거부한다(미터라 카메라 거리 불변).
        # depth가 없으면 길이를 못 재므로 필터를 통과시킨다(경고 로그).
        self.declare_parameter("enable_axis_length_filter", True)
        self.declare_parameter("expected_axis_length_m", 0.075)
        self.declare_parameter("axis_length_tol_m", 0.02)

        self.weights_path = str(self.get_parameter("weights_path").value)
        self.image_topic = str(self.get_parameter("image_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)

        self.debug_image_topic = str(self.get_parameter("debug_image_topic").value)
        self.pose2d_topic = str(self.get_parameter("pose2d_topic").value)
        self.grasp_pose_topic = str(self.get_parameter("grasp_pose_topic").value)
        self.cups_pose2d_topic = str(self.get_parameter("cups_pose2d_topic").value)
        self.cups_grasp_poses_topic = str(
            self.get_parameter("cups_grasp_poses_topic").value
        )

        self.imgsz = int(self.get_parameter("imgsz").value)
        self.conf = float(self.get_parameter("conf").value)
        self.iou = float(self.get_parameter("iou").value)
        self.device = str(self.get_parameter("device").value)
        self.half = as_bool(self.get_parameter("half").value)

        self.mode = str(self.get_parameter("mode").value)
        self.target_class_name = str(self.get_parameter("target_class_name").value)
        self.use_depth = as_bool(self.get_parameter("use_depth").value)

        self.grip_offset_m = float(self.get_parameter("grip_offset_m").value)
        self.top_diameter_m = float(self.get_parameter("top_diameter_m").value)
        self.pixels_per_meter = float(self.get_parameter("pixels_per_meter").value)

        self.min_mask_area = float(self.get_parameter("min_mask_area").value)
        self.min_pair_distance_px = float(self.get_parameter("min_pair_distance_px").value)
        self.max_pair_distance_px = float(self.get_parameter("max_pair_distance_px").value)
        self.min_pair_diameter_ratio = float(
            self.get_parameter("min_pair_diameter_ratio").value
        )
        self.enable_axis_length_filter = as_bool(
            self.get_parameter("enable_axis_length_filter").value
        )
        self.expected_axis_length_m = float(
            self.get_parameter("expected_axis_length_m").value
        )
        self.axis_length_tol_m = float(
            self.get_parameter("axis_length_tol_m").value
        )

        if self.weights_path == "":
            raise RuntimeError("weights_path is empty.")

        if self.device != "cpu" and not torch.cuda.is_available():
            self.get_logger().warn("CUDA is not available. Falling back to CPU.")
            self.device = "cpu"
            self.half = False

        if self.device == "cpu":
            self.half = False

        # cv_bridge 제거 — 모듈 단위 helper(imgmsg_to_cv2, cv2_to_imgmsg) 사용

        self.last_depth_m = None
        self.last_depth_header = None

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.get_logger().info(f"Loading YOLO model: {self.weights_path}")
        self.model = YOLO(self.weights_path)

        try:
            self.model.fuse()
        except Exception as e:
            self.get_logger().warn(f"model.fuse() skipped: {e}")

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            qos,
        )

        if self.use_depth:
            self.depth_sub = self.create_subscription(
                Image,
                self.depth_topic,
                self.depth_callback,
                qos,
            )

            self.info_sub = self.create_subscription(
                CameraInfo,
                self.camera_info_topic,
                self.camera_info_callback,
                qos,
            )

        self.debug_pub = self.create_publisher(Image, self.debug_image_topic, 10)
        self.pose2d_pub = self.create_publisher(Float32MultiArray, self.pose2d_topic, 10)
        self.grasp_pose_pub = self.create_publisher(PoseStamped, self.grasp_pose_topic, 10)
        # Multi-cup (Phase 1)
        self.cups_pose2d_pub = self.create_publisher(
            Float32MultiArray, self.cups_pose2d_topic, 10
        )
        self.cups_grasp_poses_pub = self.create_publisher(
            PoseArray, self.cups_grasp_poses_topic, 10
        )

        self.get_logger().info("fallen_cup_pose_node started.")
        self.get_logger().info(f"image_topic: {self.image_topic}")
        self.get_logger().info(f"mode: {self.mode}")
        self.get_logger().info(f"use_depth: {self.use_depth}")
        self.get_logger().info(f"target_class_name: '{self.target_class_name}'")
        self.get_logger().info(f"model classes: {getattr(self.model, 'names', None)}")

    # -----------------------------
    # Depth / camera info
    # -----------------------------
    def depth_callback(self, msg: Image):
        try:
            depth = imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().warn(f"depth conversion failed: {e}")
            return

        if msg.encoding == "16UC1":
            depth_m = depth.astype(np.float32) * 0.001
        elif msg.encoding == "32FC1":
            depth_m = depth.astype(np.float32)
        else:
            self.get_logger().warn(f"Unsupported depth encoding: {msg.encoding}")
            return

        self.last_depth_m = depth_m
        self.last_depth_header = msg.header

    def camera_info_callback(self, msg: CameraInfo):
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])

    def get_depth_at_pixel(self, u, v, window=5):
        if self.last_depth_m is None:
            return None

        h, w = self.last_depth_m.shape[:2]
        u = int(round(u))
        v = int(round(v))

        if u < 0 or u >= w or v < 0 or v >= h:
            return None

        r = window // 2
        x0 = max(0, u - r)
        x1 = min(w, u + r + 1)
        y0 = max(0, v - r)
        y1 = min(h, v + r + 1)

        patch = self.last_depth_m[y0:y1, x0:x1]
        valid = patch[np.isfinite(patch)]
        valid = valid[valid > 0.05]

        if valid.size == 0:
            return None

        return float(np.median(valid))

    def deproject_pixel_to_3d(self, u, v, z):
        if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
            return None

        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy

        return x, y, z

    def compute_axis_length_m(self, top_center, bottom_center):
        """top→bottom 축의 실제 길이(m). depth/intrinsics 없으면 None.

        축 양 끝 픽셀을 각자의 depth로 3D(camera optical frame) deproject 한 뒤
        유클리드 거리를 잰다. 미터 단위라 카메라-컵 거리가 변해도 일정하고,
        두 컵을 잇는 잘못된 축은 정상 대비 크게 길어져 쉽게 구분된다.
        """
        if not self.use_depth or self.fx is None or self.fy is None:
            return None
        z_t = self.get_depth_at_pixel(top_center[0], top_center[1], window=7)
        z_b = self.get_depth_at_pixel(bottom_center[0], bottom_center[1], window=7)
        if z_t is None or z_b is None:
            return None
        p_t = self.deproject_pixel_to_3d(top_center[0], top_center[1], z_t)
        p_b = self.deproject_pixel_to_3d(bottom_center[0], bottom_center[1], z_b)
        if p_t is None or p_b is None:
            return None
        return float(np.linalg.norm(np.asarray(p_b) - np.asarray(p_t)))

    # -----------------------------
    # YOLO mask extraction
    # -----------------------------
    def extract_detections(self, result, image_h, image_w):
        detections = []

        if result.masks is None or result.masks.data is None:
            return detections

        masks = result.masks.data.detach().cpu().numpy()

        boxes = result.boxes
        confs = None
        clss = None

        if boxes is not None:
            if boxes.conf is not None:
                confs = boxes.conf.detach().cpu().numpy()
            if boxes.cls is not None:
                clss = boxes.cls.detach().cpu().numpy()

        for i, mask in enumerate(masks):
            if mask.shape[:2] != (image_h, image_w):
                mask = cv2.resize(mask, (image_w, image_h), interpolation=cv2.INTER_NEAREST)

            binary = (mask > 0.5).astype(np.uint8) * 255

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                continue

            contour = max(contours, key=cv2.contourArea)
            area = float(cv2.contourArea(contour))

            if area < self.min_mask_area:
                continue

            M = cv2.moments(contour)
            if abs(M["m00"]) < 1e-6:
                continue

            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])

            equivalent_diameter = math.sqrt(4.0 * area / math.pi)

            conf = float(confs[i]) if confs is not None and i < len(confs) else 1.0
            cls_id = int(clss[i]) if clss is not None and i < len(clss) else -1
            cls_name = self._class_id_to_name(cls_id)

            detections.append({
                "mask": binary,
                "contour": contour,
                "area": area,
                "center": np.array([cx, cy], dtype=np.float32),
                "diameter": float(equivalent_diameter),
                "conf": conf,
                "cls_id": cls_id,
                "cls_name": cls_name,
            })

        return detections

    def _class_id_to_name(self, cls_id):
        if cls_id is None or cls_id < 0:
            return None
        names = getattr(self.model, "names", None)
        if names is None:
            return None
        if isinstance(names, dict):
            return names.get(cls_id)
        try:
            return names[cls_id]
        except (IndexError, KeyError, TypeError):
            return None

    def filter_target_detections(self, detections):
        """target_class_name과 일치하는 detection만 남긴다.

        target_class_name이 빈 문자열이면 필터링하지 않고 모두 통과시킨다(이전 동작).
        모델 클래스 이름을 얻지 못한 detection은 안전하게 제외한다.
        """
        if not self.target_class_name:
            return list(detections)
        return [
            d for d in detections
            if d.get("cls_name") == self.target_class_name
        ]

    # -----------------------------
    # Method 1: two face masks
    # -----------------------------
    def estimate_from_two_faces(self, detections):
        """detections 전체에서 가장 점수 높은 한 pair 만 골라 estimate 반환 (legacy)."""
        if len(detections) < 2:
            return None

        best_pair = None
        best_score = -1.0

        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                score = self._two_face_pair_score(detections[i], detections[j])
                if score is None:
                    continue
                if score > best_score:
                    best_score = score
                    best_pair = (detections[i], detections[j])

        if best_pair is None:
            return None

        return self._make_two_face_estimate(best_pair[0], best_pair[1])

    def _two_face_pair_score(self, a, b):
        """pair (a, b)의 valid 여부 + score 계산. valid 아니면 None."""
        ca = a["center"]
        cb = b["center"]
        dist = float(np.linalg.norm(ca - cb))

        if dist < self.min_pair_distance_px:
            return None
        if dist > self.max_pair_distance_px:
            return None

        da = max(a["diameter"], 1.0)
        db = max(b["diameter"], 1.0)
        ratio = max(da, db) / min(da, db)

        # 한 컵의 두 face 는 narrow/wide 비가 분명히 다르고(≥~1.5), 별개 두 컵
        # 끼리는 비≈1.0. 비슷한 크기 mask 쌍은 cross-cup pairing 이라고 보고 거부.
        if ratio < self.min_pair_diameter_ratio:
            return None

        # 크기 차이가 있고, 중심 간 거리가 긴 pair를 선호
        return ratio * dist * 0.5 * (a["conf"] + b["conf"])

    def _make_two_face_estimate(self, a, b):
        """두 detection (top+bottom face 후보) 으로부터 estimate dict 생성. 실패 시 None."""
        # 사용자가 정의한 가정:
        # 밑면 원 지름이 크고, 윗면 원 지름이 작다.
        if a["diameter"] >= b["diameter"]:
            bottom = a
            top = b
        else:
            bottom = b
            top = a

        top_center = top["center"]
        bottom_center = bottom["center"]

        axis = bottom_center - top_center
        norm = float(np.linalg.norm(axis))

        if norm < 1e-6:
            return None

        direction = axis / norm
        yaw = math.atan2(float(direction[1]), float(direction[0]))

        top_width_px = float(top["diameter"])
        bottom_width_px = float(bottom["diameter"])

        offset_px = self.compute_grip_offset_px(
            top_center=top_center,
            top_width_px=top_width_px,
            fallback_length_px=norm,
            direction=direction,
        )

        grip_point = top_center + direction * offset_px

        return {
            "method": "two_face",
            "top_center": top_center,
            "bottom_center": bottom_center,
            "direction": direction,
            "yaw": yaw,
            "grip_point": grip_point,
            "top_width_px": top_width_px,
            "bottom_width_px": bottom_width_px,
            "confidence": 0.5 * (top["conf"] + bottom["conf"]),
            "axis_length_m": self.compute_axis_length_m(top_center, bottom_center),
            "axis_length_px": float(norm),
        }

    # -----------------------------
    # Multi-cup: cup 마다 한 estimate
    # -----------------------------
    def estimate_all_cups(self, detections):
        """전체 detections 에서 cup 별 estimate 리스트 반환.

        - mode in (auto, two_face): greedy pairing 으로 가능한 모든 (top, bottom) pair
          를 cup 으로 묶음. 한 detection 은 하나의 pair 에만 속함.
        - mode in (auto, silhouette): two_face 에 사용되지 않은(또는 mode 가 silhouette
          단독인 경우 전체) detection 각각에 silhouette PCA 적용.
        - 각 estimate dict 에 cup_id 부여 (리스트 인덱스).
        """
        estimates = []
        used_indices = set()

        if self.mode in ("auto", "two_face") and len(detections) >= 2:
            scored_pairs = []
            for i in range(len(detections)):
                for j in range(i + 1, len(detections)):
                    s = self._two_face_pair_score(detections[i], detections[j])
                    if s is not None:
                        scored_pairs.append((s, i, j))

            scored_pairs.sort(key=lambda t: t[0], reverse=True)
            for s, i, j in scored_pairs:
                if i in used_indices or j in used_indices:
                    continue
                est = self._make_two_face_estimate(detections[i], detections[j])
                if est is None:
                    continue
                estimates.append(est)
                used_indices.add(i)
                used_indices.add(j)

        if self.mode in ("auto", "silhouette"):
            for i, det in enumerate(detections):
                if i in used_indices:
                    continue
                est = self.estimate_from_silhouette(det)
                if est is None:
                    continue
                estimates.append(est)
                used_indices.add(i)

        # cup_id = 리스트 인덱스 (간단히)
        for idx, est in enumerate(estimates):
            est["cup_id"] = idx

        return estimates

    def split_by_axis_length(self, estimates):
        """축 실제 길이 기준으로 (정상, 거부) 로 분리.

        - 필터 off 면 전부 정상으로 통과.
        - axis_length_m 가 None(depth 없음) 이면 거를 수 없으므로 통과(경고 1회).
        - 정상 밴드 [expected ± tol] 밖이면 거부(로그). 거부분은 debug 표시용 반환.
        """
        if not self.enable_axis_length_filter:
            return list(estimates), []

        lo = self.expected_axis_length_m - self.axis_length_tol_m
        hi = self.expected_axis_length_m + self.axis_length_tol_m
        kept, rejected = [], []
        warned_no_depth = False
        for est in estimates:
            L = est.get("axis_length_m")
            if L is None:
                if not warned_no_depth:
                    self.get_logger().warn(
                        "axis length(m) 계산 불가(depth 없음) — length filter 스킵"
                    )
                    warned_no_depth = True
                kept.append(est)
                continue
            if lo <= L <= hi:
                kept.append(est)
            else:
                rejected.append(est)
                self.get_logger().info(
                    f"[len-filter] reject cup: axis={L * 100.0:.1f}cm "
                    f"(기대 {self.expected_axis_length_m * 100.0:.1f}"
                    f"±{self.axis_length_tol_m * 100.0:.1f}cm)"
                )
        return kept, rejected

    # -----------------------------
    # Method 2: one full silhouette mask
    # -----------------------------
    def estimate_from_silhouette(self, detection):
        mask = detection["mask"]

        ys, xs = np.where(mask > 0)
        if xs.size < 20:
            return None

        pts = np.stack([xs, ys], axis=1).astype(np.float32)

        center = np.mean(pts, axis=0)
        centered = pts - center

        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eig(cov)

        max_idx = int(np.argmax(eigvals))
        v = eigvecs[:, max_idx].astype(np.float32)

        norm_v = float(np.linalg.norm(v))
        if norm_v < 1e-6:
            return None

        v = v / norm_v

        # PCA 축의 수직 방향
        n = np.array([-v[1], v[0]], dtype=np.float32)

        proj = centered @ v
        perp = centered @ n

        t_min = float(np.percentile(proj, 2))
        t_max = float(np.percentile(proj, 98))
        length = t_max - t_min

        if length < 20.0:
            return None

        band = max(5.0, 0.06 * length)

        # 양쪽 끝에서 너무 극단적인 edge가 아니라 약간 안쪽 단면을 사용
        t_low = t_min + 0.10 * length
        t_high = t_max - 0.10 * length

        low_center, low_width = self.cross_section_center(
            center=center,
            pts_centered=centered,
            proj=proj,
            perp=perp,
            v=v,
            n=n,
            target_t=t_low,
            band=band,
        )

        high_center, high_width = self.cross_section_center(
            center=center,
            pts_centered=centered,
            proj=proj,
            perp=perp,
            v=v,
            n=n,
            target_t=t_high,
            band=band,
        )

        if low_center is None or high_center is None:
            return None

        # 사용자가 정의한 가정:
        # 폭이 큰 쪽 = 밑면, 폭이 작은 쪽 = 윗면
        if low_width >= high_width:
            bottom_center = low_center
            top_center = high_center
            bottom_width_px = low_width
            top_width_px = high_width
        else:
            bottom_center = high_center
            top_center = low_center
            bottom_width_px = high_width
            top_width_px = low_width

        axis = bottom_center - top_center
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-6:
            return None

        direction = axis / axis_norm
        yaw = math.atan2(float(direction[1]), float(direction[0]))

        offset_px = self.compute_grip_offset_px(
            top_center=top_center,
            top_width_px=top_width_px,
            fallback_length_px=axis_norm,
            direction=direction,
        )

        grip_point = top_center + direction * offset_px

        return {
            "method": "silhouette",
            "top_center": top_center,
            "bottom_center": bottom_center,
            "direction": direction,
            "yaw": yaw,
            "grip_point": grip_point,
            "top_width_px": float(top_width_px),
            "bottom_width_px": float(bottom_width_px),
            "confidence": float(detection["conf"]),
            "axis_length_m": self.compute_axis_length_m(top_center, bottom_center),
            "axis_length_px": float(axis_norm),
        }

    def cross_section_center(self, center, pts_centered, proj, perp, v, n, target_t, band):
        idx = np.abs(proj - target_t) <= band

        if np.count_nonzero(idx) < 10:
            return None, None

        p_local = proj[idx]
        q_local = perp[idx]

        p_mean = float(np.mean(p_local))
        q_min = float(np.min(q_local))
        q_max = float(np.max(q_local))
        q_center = 0.5 * (q_min + q_max)

        width = q_max - q_min

        point = center + p_mean * v + q_center * n

        return point.astype(np.float32), float(width)

    # -----------------------------
    # Grip offset
    # -----------------------------
    def compute_grip_offset_px(self, top_center, top_width_px, fallback_length_px, direction):
        # 1) 사용자가 직접 pixel/meter를 준 경우
        if self.pixels_per_meter > 0.0:
            return self.grip_offset_m * self.pixels_per_meter

        # 2) depth + camera_info가 있는 경우
        if self.use_depth and self.fx is not None and self.fy is not None:
            z = self.get_depth_at_pixel(top_center[0], top_center[1], window=7)
            if z is not None and z > 0.05:
                f_mean = 0.5 * (self.fx + self.fy)
                return self.grip_offset_m * f_mean / z

        # 3) 작은 원 실제 지름을 알고 있다고 가정하고 pixel scale 추정
        if self.top_diameter_m > 0.0 and top_width_px > 1.0:
            return self.grip_offset_m * top_width_px / self.top_diameter_m

        # 4) 최후 fallback
        return 0.15 * fallback_length_px

    # -----------------------------
    # Main callback
    # -----------------------------
    def image_callback(self, msg: Image):
        try:
            frame_bgr = imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"image conversion failed: {e}")
            return

        h, w = frame_bgr.shape[:2]
        debug = frame_bgr.copy()

        start_time = time.time()

        try:
            with torch.inference_mode():
                results = self.model.predict(
                    source=frame_bgr,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    iou=self.iou,
                    device=self.device,
                    half=self.half,
                    verbose=False,
                    retina_masks=True,
                )
        except Exception as e:
            self.get_logger().error(f"YOLO inference failed: {e}")
            return

        result = results[0]
        detections = self.extract_detections(result, h, w)

        if len(detections) == 0:
            self.publish_debug(debug, msg.header)
            return

        # 방향벡터는 target 클래스(fallen-cup)에서만 추출한다.
        # upright-cup mask가 섞여서 잘못된 top/bottom face 페어가 만들어지는 것을 막는다.
        target_detections = self.filter_target_detections(detections)

        if len(target_detections) == 0:
            self.get_logger().info(
                f"target class '{self.target_class_name}'에 해당하는 detection 없음 "
                f"(전체 {len(detections)}개) — yaw 추출 스킵"
            )
            self.draw_debug_detections_only(debug, detections)
            self.publish_debug(debug, msg.header)
            return

        # Multi-cup: target_detections 전체에 대해 cup 마다 estimate.
        # 빈 리스트면 (estimate 실패) early return.
        estimates = self.estimate_all_cups(target_detections)

        # 축 길이 sanity 필터: 잘못된 mask(두 컵 병합 / cross-cup pairing)는 축이
        # 비정상적으로 길어짐 → 정상 길이 밴드 밖이면 거부. 거부분은 debug 에 빨간색.
        estimates, rejected = self.split_by_axis_length(estimates)
        for idx, est in enumerate(estimates):
            est["cup_id"] = idx

        if len(estimates) == 0:
            # 전부 거부돼도 rejected 를 그려 왜 걸러졌는지 보이게 한다.
            self.draw_debug(debug, detections, [], rejected)
            self.publish_debug(debug, msg.header)
            return

        # Backward compat: 기존 single-cup 토픽은 first cup (cup_id=0) 으로 publish.
        # 새 consumer 는 cups_pose2d / cups_grasp_poses 를 사용.
        primary = estimates[0]
        self.publish_pose2d(primary)
        self.publish_grasp_pose3d(primary, msg.header)

        # Multi-cup 토픽
        self.publish_cups_pose2d(estimates)
        self.publish_cups_grasp_poses(estimates, msg.header)

        self.draw_debug(debug, detections, estimates, rejected)
        self.publish_debug(debug, msg.header)

        elapsed = time.time() - start_time
        methods = ",".join(e["method"] for e in estimates)
        yaws = ",".join(f"{math.degrees(e['yaw']):.0f}" for e in estimates)
        lens = ",".join(
            f"{e['axis_length_m'] * 100.0:.1f}"
            if e.get("axis_length_m") is not None else "NA"
            for e in estimates
        )
        self.get_logger().info(
            f"cups={len(estimates)} rej={len(rejected)} methods=[{methods}] "
            f"yaws_deg=[{yaws}] axis_cm=[{lens}] "
            f"time={elapsed * 1000.0:.1f} ms"
        )

    # -----------------------------
    # Publish
    # -----------------------------
    def publish_pose2d(self, estimate):
        top = estimate["top_center"]
        bottom = estimate["bottom_center"]
        direction = estimate["direction"]
        grip = estimate["grip_point"]

        msg = Float32MultiArray()
        msg.data = [
            float(top[0]),
            float(top[1]),
            float(bottom[0]),
            float(bottom[1]),
            float(direction[0]),
            float(direction[1]),
            float(estimate["yaw"]),
            float(grip[0]),
            float(grip[1]),
            float(estimate["confidence"]),
            float(estimate["top_width_px"]),
            float(estimate["bottom_width_px"]),
        ]

        self.pose2d_pub.publish(msg)

    # -- Multi-cup publish (Phase 1) ---------------------------------
    # 두 토픽이 같은 cup 집합 / 같은 순서로 publish 됨 → consumer 는 index 로 매칭.
    # cups_pose2d:
    #   Float32MultiArray, layout dim [{label:"cup", size:N, stride:N*13},
    #                                  {label:"field", size:13, stride:13}]
    #   row 마다 13 fields:
    #     [cup_id, top_x_px, top_y_px, bot_x_px, bot_y_px,
    #      dir_x, dir_y, yaw_rad, grip_x_px, grip_y_px,
    #      conf, top_w_px, bot_w_px]
    # cups_grasp_poses:
    #   PoseArray. header 는 image header.
    #   각 Pose: position = camera optical frame 3D grip point. orientation = image yaw.
    #   depth 가 없는 cup 은 position 을 NaN 으로 채워 publish (index 동기 유지).
    def publish_cups_pose2d(self, estimates):
        if len(estimates) == 0:
            return
        msg = Float32MultiArray()
        cup_dim = MultiArrayDimension()
        cup_dim.label = "cup"
        cup_dim.size = len(estimates)
        cup_dim.stride = len(estimates) * 13
        field_dim = MultiArrayDimension()
        field_dim.label = "field"
        field_dim.size = 13
        field_dim.stride = 13
        msg.layout.dim.append(cup_dim)
        msg.layout.dim.append(field_dim)
        data = []
        for est in estimates:
            top = est["top_center"]
            bot = est["bottom_center"]
            d = est["direction"]
            grip = est["grip_point"]
            data.extend([
                float(est["cup_id"]),
                float(top[0]), float(top[1]),
                float(bot[0]), float(bot[1]),
                float(d[0]), float(d[1]),
                float(est["yaw"]),
                float(grip[0]), float(grip[1]),
                float(est["confidence"]),
                float(est["top_width_px"]),
                float(est["bottom_width_px"]),
            ])
        msg.data = data
        self.cups_pose2d_pub.publish(msg)

    def publish_cups_grasp_poses(self, estimates, header):
        if not self.use_depth or len(estimates) == 0:
            return
        msg = PoseArray()
        msg.header = header
        nan = float("nan")
        for est in estimates:
            grip = est["grip_point"]
            z = self.get_depth_at_pixel(grip[0], grip[1], window=7)
            pose = Pose()
            if z is None:
                # depth 없는 cup: NaN 으로 표시. consumer 는 isnan 체크로 필터.
                pose.position.x = nan
                pose.position.y = nan
                pose.position.z = nan
                pose.orientation.x = 0.0
                pose.orientation.y = 0.0
                pose.orientation.z = 0.0
                pose.orientation.w = 1.0
            else:
                point_3d = self.deproject_pixel_to_3d(grip[0], grip[1], z)
                if point_3d is None:
                    pose.position.x = nan
                    pose.position.y = nan
                    pose.position.z = nan
                    pose.orientation.w = 1.0
                else:
                    x, y, zz = point_3d
                    pose.position.x = float(x)
                    pose.position.y = float(y)
                    pose.position.z = float(zz)
                    yaw = float(est["yaw"])
                    pose.orientation.x = 0.0
                    pose.orientation.y = 0.0
                    pose.orientation.z = math.sin(yaw * 0.5)
                    pose.orientation.w = math.cos(yaw * 0.5)
            msg.poses.append(pose)
        self.cups_grasp_poses_pub.publish(msg)

    def publish_grasp_pose3d(self, estimate, header):
        if not self.use_depth:
            return

        grip = estimate["grip_point"]
        z = self.get_depth_at_pixel(grip[0], grip[1], window=7)

        if z is None:
            return

        point_3d = self.deproject_pixel_to_3d(grip[0], grip[1], z)
        if point_3d is None:
            return

        x, y, z = point_3d

        pose_msg = PoseStamped()
        pose_msg.header = header

        pose_msg.pose.position.x = float(x)
        pose_msg.pose.position.y = float(y)
        pose_msg.pose.position.z = float(z)

        # 주의:
        # 이 yaw는 camera optical frame의 image x-y 평면 기준 yaw입니다.
        # 실제 그리퍼 base_link yaw로 쓰려면 tf 변환이 필요합니다.
        yaw = float(estimate["yaw"])
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = math.sin(yaw * 0.5)
        pose_msg.pose.orientation.w = math.cos(yaw * 0.5)

        self.grasp_pose_pub.publish(pose_msg)

    def publish_debug(self, image_bgr, header):
        try:
            msg = cv2_to_imgmsg(image_bgr, encoding="bgr8")
            msg.header = header
            self.debug_pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f"debug image publish failed: {e}")

    # -----------------------------
    # Debug drawing
    # -----------------------------
    def draw_debug_detections_only(self, image, detections):
        """추정이 없을 때 mask 윤곽만 표시 (target/non-target 색 구분)."""
        for det in detections:
            is_target = (
                not self.target_class_name
                or det.get("cls_name") == self.target_class_name
            )
            color = (0, 255, 255) if is_target else (80, 80, 80)
            cv2.drawContours(image, [det["contour"]], -1, color, 1)

    # cup_id 별 distinguishable BGR 색 팔레트 (debug 표시용).
    _CUP_COLORS = [
        (0, 255, 0),     # green
        (0, 200, 255),   # orange
        (255, 100, 0),   # blue
        (255, 0, 255),   # magenta
        (0, 255, 255),   # yellow
        (255, 255, 0),   # cyan
    ]

    def _cup_color(self, cup_id):
        return self._CUP_COLORS[cup_id % len(self._CUP_COLORS)]

    def draw_debug(self, image, detections, estimates, rejected=None):
        """estimates: cup 별 estimate 리스트. cup_id 마다 다른 색 + 번호 표시.

        rejected: 축 길이 필터에 걸린 estimate 리스트. 빨간 선 + 길이 라벨로 표시해
        튜닝 시 어떤 벡터가 왜 걸러졌는지 눈으로 확인 가능.
        """
        for det in detections:
            cv2.drawContours(image, [det["contour"]], -1, (80, 80, 80), 1)

        for est in (rejected or []):
            top_i = tuple(np.round(est["top_center"]).astype(int))
            bottom_i = tuple(np.round(est["bottom_center"]).astype(int))
            cv2.line(image, top_i, bottom_i, (0, 0, 255), 2)  # red
            L = est.get("axis_length_m")
            label = f"REJECT {L * 100.0:.0f}cm" if L is not None else "REJECT"
            mid = (
                (top_i[0] + bottom_i[0]) // 2,
                (top_i[1] + bottom_i[1]) // 2,
            )
            cv2.putText(
                image, label, (mid[0] + 8, mid[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
            )

        for est in estimates:
            cup_id = int(est.get("cup_id", 0))
            color = self._cup_color(cup_id)

            top = est["top_center"]
            bottom = est["bottom_center"]
            grip = est["grip_point"]
            direction = est["direction"]

            top_i = tuple(np.round(top).astype(int))
            bottom_i = tuple(np.round(bottom).astype(int))
            grip_i = tuple(np.round(grip).astype(int))

            arrow_end = grip + direction * 60.0
            arrow_i = tuple(np.round(arrow_end).astype(int))

            cv2.circle(image, top_i, 5, (0, 255, 255), -1)
            cv2.circle(image, bottom_i, 5, (0, 0, 255), -1)
            cv2.circle(image, grip_i, 6, color, -1)

            cv2.line(image, top_i, bottom_i, (255, 0, 0), 2)
            cv2.arrowedLine(image, top_i, arrow_i, color, 2, tipLength=0.25)

            # cup_id 라벨 (grip 점 옆 큰 글씨)
            cv2.putText(
                image,
                f"#{cup_id}",
                (grip_i[0] + 10, grip_i[1] + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
            )
            # method + yaw (작은 글씨)
            yaw_deg = math.degrees(float(est["yaw"]))
            cv2.putText(
                image,
                f"{est['method']} {yaw_deg:+.0f}d",
                (grip_i[0] + 10, grip_i[1] + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        # 헤더: 감지된 cup 수 (+ 길이 필터에 걸린 수)
        header = f"cups={len(estimates)}"
        if rejected:
            header += f" rej={len(rejected)}"
        cv2.putText(
            image,
            header,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )


def main(args=None):
    rclpy.init(args=args)

    node = FallenCupPoseNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()