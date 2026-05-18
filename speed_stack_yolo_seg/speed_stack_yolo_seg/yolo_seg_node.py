#!/usr/bin/env python3

import time
import cv2
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from std_msgs.msg import Int32
from cv_bridge import CvBridge

from ultralytics import YOLO


def as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ["true", "1", "yes", "y"]
    return bool(value)


class YoloSegNode(Node):
    def __init__(self):
        super().__init__("yolo_seg_node")

        # -----------------------------
        # ROS parameters
        # -----------------------------
        self.declare_parameter("weights_path", "")
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("overlay_topic", "/yolo_seg/overlay")
        self.declare_parameter("mask_topic", "/yolo_seg/mask")
        self.declare_parameter("count_topic", "/yolo_seg/count")

        self.declare_parameter("imgsz", 1280)
        self.declare_parameter("conf", 0.25)
        self.declare_parameter("iou", 0.45)
        self.declare_parameter("device", "0")      # "0" = CUDA GPU 0, "cpu" = CPU
        self.declare_parameter("half", True)       # CUDA에서만 권장
        self.declare_parameter("retina_masks", True)
        self.declare_parameter("frame_skip", 0)    # 0: 모든 프레임 처리, 1: 1프레임 건너뜀

        self.weights_path = str(self.get_parameter("weights_path").value)
        self.image_topic = str(self.get_parameter("image_topic").value)
        self.overlay_topic = str(self.get_parameter("overlay_topic").value)
        self.mask_topic = str(self.get_parameter("mask_topic").value)
        self.count_topic = str(self.get_parameter("count_topic").value)

        self.imgsz = int(self.get_parameter("imgsz").value)
        self.conf = float(self.get_parameter("conf").value)
        self.iou = float(self.get_parameter("iou").value)
        self.device = str(self.get_parameter("device").value)
        self.half = as_bool(self.get_parameter("half").value)
        self.retina_masks = as_bool(self.get_parameter("retina_masks").value)
        self.frame_skip = int(self.get_parameter("frame_skip").value)

        if self.weights_path == "":
            raise RuntimeError("weights_path parameter is empty. Please provide YOLO .pt path.")

        if self.device != "cpu" and not torch.cuda.is_available():
            self.get_logger().warn(
                "CUDA is not available. Falling back to CPU. "
                "Inference may be slow."
            )
            self.device = "cpu"
            self.half = False

        if self.device == "cpu":
            self.half = False

        self.get_logger().info(f"Loading YOLO segmentation model: {self.weights_path}")
        self.model = YOLO(self.weights_path)

        try:
            self.model.fuse()
        except Exception as e:
            self.get_logger().warn(f"model.fuse() skipped: {e}")

        self.bridge = CvBridge()
        self.frame_count = 0
        self.last_log_time = time.time()

        # 카메라 토픽은 실시간성이 중요하므로 queue를 작게 둡니다.
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.image_sub = self.create_subscription(
            Image,
            self.image_topic,
            self.image_callback,
            image_qos,
        )

        self.overlay_pub = self.create_publisher(Image, self.overlay_topic, 10)
        self.mask_pub = self.create_publisher(Image, self.mask_topic, 10)
        self.count_pub = self.create_publisher(Int32, self.count_topic, 10)

        self.get_logger().info("YOLO segmentation node started.")
        self.get_logger().info(f"Subscribing image topic: {self.image_topic}")
        self.get_logger().info(f"Publishing overlay topic: {self.overlay_topic}")
        self.get_logger().info(f"Publishing mask topic: {self.mask_topic}")

    def image_callback(self, msg: Image):
        self.frame_count += 1

        if self.frame_skip > 0:
            if self.frame_count % (self.frame_skip + 1) != 1:
                return

        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        h, w = frame_bgr.shape[:2]

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
                    retina_masks=self.retina_masks,
                    verbose=False,
                )
        except Exception as e:
            self.get_logger().error(f"YOLO inference failed: {e}")
            return

        result = results[0]

        # -----------------------------
        # 1) Overlay image 생성
        # -----------------------------
        try:
            overlay_bgr = result.plot()
        except Exception:
            overlay_bgr = frame_bgr.copy()

        # -----------------------------
        # 2) Combined binary mask 생성
        #    컵 윗면 instance들이 모두 흰색(255)으로 표시됨
        # -----------------------------
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        num_det = 0

        if result.masks is not None and result.masks.data is not None:
            masks = result.masks.data.detach().cpu().numpy()
            num_det = masks.shape[0]

            for mask in masks:
                # mask shape가 원본 이미지 크기와 다르면 resize
                if mask.shape[:2] != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

                binary = (mask > 0.5).astype(np.uint8) * 255
                combined_mask = np.maximum(combined_mask, binary)
        else:
            num_det = 0

        # -----------------------------
        # 3) ROS topic publish
        # -----------------------------
        try:
            overlay_msg = self.bridge.cv2_to_imgmsg(overlay_bgr, encoding="bgr8")
            overlay_msg.header = msg.header
            self.overlay_pub.publish(overlay_msg)

            mask_msg = self.bridge.cv2_to_imgmsg(combined_mask, encoding="mono8")
            mask_msg.header = msg.header
            self.mask_pub.publish(mask_msg)

            count_msg = Int32()
            count_msg.data = int(num_det)
            self.count_pub.publish(count_msg)

        except Exception as e:
            self.get_logger().error(f"Publishing failed: {e}")
            return

        # -----------------------------
        # 4) FPS 로그
        # -----------------------------
        elapsed = time.time() - start_time
        now = time.time()

        if now - self.last_log_time > 1.0:
            fps = 1.0 / elapsed if elapsed > 0 else 0.0
            self.get_logger().info(
                f"detections={num_det}, inference_time={elapsed*1000:.1f} ms, approx_fps={fps:.1f}"
            )
            self.last_log_time = now


def main(args=None):
    rclpy.init(args=args)

    node = YoloSegNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()