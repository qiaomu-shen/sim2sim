#!/usr/bin/env python3

import argparse
import os
import sys
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


MAGIC = "UTDEPTH1"


class DepthBridge(Node):
    def __init__(self, topic: str, output: str, encoding: str):
        super().__init__("unitree_g1_depth_bridge")
        self.topic = topic
        self.output = output
        self.encoding = encoding.upper()
        self.last_encoding_warn_time = 0.0
        self.last_size_warn_time = 0.0

        output_dir = os.path.dirname(os.path.abspath(output))
        os.makedirs(output_dir, exist_ok=True)

        self.sub = self.create_subscription(Image, topic, self.on_image, 5)
        self.get_logger().info(
            f"bridging {topic} ({self.encoding}) -> {output}"
        )

    def on_image(self, msg: Image) -> None:
        msg_encoding = msg.encoding.upper()
        if msg_encoding != self.encoding:
            now = time.monotonic()
            if now - self.last_encoding_warn_time > 2.0:
                self.get_logger().warn(
                    f"skip depth frame: expected {self.encoding}, got {msg.encoding}"
                )
                self.last_encoding_warn_time = now
            return

        raw = bytes(msg.data)
        expected_min = int(msg.step) * int(msg.height)
        if len(raw) < expected_min:
            now = time.monotonic()
            if now - self.last_size_warn_time > 2.0:
                self.get_logger().warn(
                    f"skip depth frame: data has {len(raw)} bytes, expected at least {expected_min}"
                )
                self.last_size_warn_time = now
            return

        header = (
            f"{MAGIC} {int(msg.width)} {int(msg.height)} "
            f"{int(msg.step)} {int(msg.is_bigendian)}\n"
        ).encode("ascii")

        tmp_path = f"{self.output}.{os.getpid()}.tmp"
        with open(tmp_path, "wb") as f:
            f.write(header)
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.output)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Bridge a ROS2 16UC1 depth Image topic to Unitree G1 deploy binary depth frames."
    )
    parser.add_argument(
        "--topic",
        default="/camera/depth/image_rect_raw",
        help="ROS2 depth image topic to subscribe.",
    )
    parser.add_argument(
        "--output",
        default="/tmp/unitree_g1_front_depth.bin",
        help="Binary bridge file read by the deploy process.",
    )
    parser.add_argument(
        "--encoding",
        default="16UC1",
        help="Expected ROS Image encoding. The deploy reader assumes 16UC1 millimeters.",
    )
    return parser.parse_known_args(argv)


def main(argv=None):
    args, ros_args = parse_args(sys.argv[1:] if argv is None else argv)
    rclpy.init(args=ros_args)
    node = DepthBridge(args.topic, args.output, args.encoding)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
