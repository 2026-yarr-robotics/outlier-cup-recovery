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
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import PoseStamped

from cv_bridge import CvBridge
from ultralytics import YOLO


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

        self.declare_parameter("imgsz", 640)
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("iou", 0.45)
        self.declare_parameter("device", "cpu")
        self.declare_parameter("half", False)

        # auto: mask가 2개 이상이면 top/bottom face 방식 시도, 아니면 silhouette PCA 방식
        # silhouette: 넘어진 컵 전체 mask 하나로 방향 추정
        # two_face: 작은 원/큰 원 두 mask로 방향 추정
        self.declare_parameter("mode", "auto")

        self.declare_parameter("use_depth", False)

        # top_center에서 bottom 방향으로 얼마나 안쪽을 잡을지
        # 0.025 m = 2.5 cm
        self.declare_parameter("grip_offset_m", 0.025)

        # depth를 안 쓸 때 pixel offset 계산용.
        # 실제 컵의 작은 원 지름을 재서 넣는 것을 추천.
        # 예: 작은 원 지름이 4.5cm면 0.045
        self.declare_parameter("top_diameter_m", 0.045)

        # 직접 pixel/meter를 알고 있으면 넣기. 모르면 0.
        self.declare_parameter("pixels_per_meter", 0.0)

        self.declare_parameter("min_mask_area", 300.0)
        self.declare_parameter("min_pair_distance_px", 20.0)
        self.declare_parameter("max_pair_distance_px", 10000.0)

        self.weights_path = str(self.get_parameter("weights_path").value)
        self.image_topic = str(self.get_parameter("image_topic").value)
        self.depth_topic = str(self.get_parameter("depth_topic").value)
        self.camera_info_topic = str(self.get_parameter("camera_info_topic").value)

        self.debug_image_topic = str(self.get_parameter("debug_image_topic").value)
        self.pose2d_topic = str(self.get_parameter("pose2d_topic").value)
        self.grasp_pose_topic = str(self.get_parameter("grasp_pose_topic").value)

        self.imgsz = int(self.get_parameter("imgsz").value)
        self.conf = float(self.get_parameter("conf").value)
        self.iou = float(self.get_parameter("iou").value)
        self.device = str(self.get_parameter("device").value)
        self.half = as_bool(self.get_parameter("half").value)

        self.mode = str(self.get_parameter("mode").value)
        self.use_depth = as_bool(self.get_parameter("use_depth").value)

        self.grip_offset_m = float(self.get_parameter("grip_offset_m").value)
        self.top_diameter_m = float(self.get_parameter("top_diameter_m").value)
        self.pixels_per_meter = float(self.get_parameter("pixels_per_meter").value)

        self.min_mask_area = float(self.get_parameter("min_mask_area").value)
        self.min_pair_distance_px = float(self.get_parameter("min_pair_distance_px").value)
        self.max_pair_distance_px = float(self.get_parameter("max_pair_distance_px").value)

        if self.weights_path == "":
            raise RuntimeError("weights_path is empty.")

        if self.device != "cpu" and not torch.cuda.is_available():
            self.get_logger().warn("CUDA is not available. Falling back to CPU.")
            self.device = "cpu"
            self.half = False

        if self.device == "cpu":
            self.half = False

        self.bridge = CvBridge()

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

        self.get_logger().info("fallen_cup_pose_node started.")
        self.get_logger().info(f"image_topic: {self.image_topic}")
        self.get_logger().info(f"mode: {self.mode}")
        self.get_logger().info(f"use_depth: {self.use_depth}")

    # -----------------------------
    # Depth / camera info
    # -----------------------------
    def depth_callback(self, msg: Image):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
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

            detections.append({
                "mask": binary,
                "contour": contour,
                "area": area,
                "center": np.array([cx, cy], dtype=np.float32),
                "diameter": float(equivalent_diameter),
                "conf": conf,
                "cls_id": cls_id,
            })

        return detections

    # -----------------------------
    # Method 1: two face masks
    # -----------------------------
    def estimate_from_two_faces(self, detections):
        if len(detections) < 2:
            return None

        best_pair = None
        best_score = -1.0

        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                a = detections[i]
                b = detections[j]

                ca = a["center"]
                cb = b["center"]
                dist = float(np.linalg.norm(ca - cb))

                if dist < self.min_pair_distance_px:
                    continue
                if dist > self.max_pair_distance_px:
                    continue

                da = max(a["diameter"], 1.0)
                db = max(b["diameter"], 1.0)
                ratio = max(da, db) / min(da, db)

                # 크기 차이가 있고, 중심 간 거리가 긴 pair를 선호
                score = ratio * dist * 0.5 * (a["conf"] + b["conf"])

                if score > best_score:
                    best_score = score
                    best_pair = (a, b)

        if best_pair is None:
            return None

        a, b = best_pair

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
        }

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
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
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

        estimate = None

        if self.mode in ["auto", "two_face"]:
            estimate = self.estimate_from_two_faces(detections)

        if estimate is None and self.mode in ["auto", "silhouette"]:
            largest = max(detections, key=lambda x: x["area"])
            estimate = self.estimate_from_silhouette(largest)

        if estimate is None:
            self.publish_debug(debug, msg.header)
            return

        self.draw_debug(debug, detections, estimate)
        self.publish_pose2d(estimate)
        self.publish_grasp_pose3d(estimate, msg.header)
        self.publish_debug(debug, msg.header)

        elapsed = time.time() - start_time
        self.get_logger().info(
            f"method={estimate['method']}, "
            f"yaw={math.degrees(estimate['yaw']):.1f} deg, "
            f"grip=({estimate['grip_point'][0]:.1f}, {estimate['grip_point'][1]:.1f}), "
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
            msg = self.bridge.cv2_to_imgmsg(image_bgr, encoding="bgr8")
            msg.header = header
            self.debug_pub.publish(msg)
        except Exception as e:
            self.get_logger().warn(f"debug image publish failed: {e}")

    # -----------------------------
    # Debug drawing
    # -----------------------------
    def draw_debug(self, image, detections, estimate):
        for det in detections:
            cv2.drawContours(image, [det["contour"]], -1, (80, 80, 80), 1)

        top = estimate["top_center"]
        bottom = estimate["bottom_center"]
        grip = estimate["grip_point"]
        direction = estimate["direction"]

        top_i = tuple(np.round(top).astype(int))
        bottom_i = tuple(np.round(bottom).astype(int))
        grip_i = tuple(np.round(grip).astype(int))

        arrow_end = grip + direction * 80.0
        arrow_i = tuple(np.round(arrow_end).astype(int))

        cv2.circle(image, top_i, 6, (0, 255, 255), -1)
        cv2.circle(image, bottom_i, 6, (0, 0, 255), -1)
        cv2.circle(image, grip_i, 7, (0, 255, 0), -1)

        cv2.line(image, top_i, bottom_i, (255, 0, 0), 2)
        cv2.arrowedLine(image, top_i, arrow_i, (0, 255, 0), 3, tipLength=0.25)

        cv2.putText(
            image,
            "TOP",
            (top_i[0] + 8, top_i[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        cv2.putText(
            image,
            "BOTTOM",
            (bottom_i[0] + 8, bottom_i[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )

        cv2.putText(
            image,
            "GRIP",
            (grip_i[0] + 8, grip_i[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        yaw_deg = math.degrees(float(estimate["yaw"]))

        cv2.putText(
            image,
            f"method={estimate['method']} yaw={yaw_deg:.1f} deg",
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