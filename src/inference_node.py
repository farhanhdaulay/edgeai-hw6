#!/usr/bin/env python3
# Copyright (c) 2026 Kishore and Farhan
# Tatung University 14210 AI實務專題
"""Inference node: runs YOLO26 TensorRT engine, publishes detections to MQTT.

Reads frames from a video file (or camera), runs detection with
the fine-tuned TensorRT engine from Lab 9, and publishes bounding-box
results as JSON messages to an MQTT topic.
"""

import argparse
import os
import time
import sys
import cv2
import numpy as np

from src.mqtt_publisher import MqttPublisher, PublisherConfig

GRAYSCALE_DIM = 2

def _default_model_factory(path: str, task: str) -> object:  # pragma: no cover
    """Real YOLO loader.
    
    Imported lazily so unit tests don't pull torch.
    Skipped from coverage because torch is Jetson-only and tests use the
    injected mock factory; real exercise happens in tests/integration/.
    """
    from ultralytics import YOLO
    return YOLO(path, task=task)

def preprocess_frame(frame: np.ndarray, target_size: tuple = (320, 320)) -> np.ndarray:
    """Convert a frame into a (1, 3, H, W) normalized float32 tensor."""
    if len(frame.shape) == GRAYSCALE_DIM:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(frame, target_size)
    tensor = np.transpose(resized, (2, 0, 1)).astype(np.float32) / 255.0
    return np.expand_dims(tensor, axis=0)

def apply_confidence_threshold(detections: list, conf_thresh: float) -> list:
    """Filter out detections below the confidence threshold."""
    return [d for d in detections if d.get("conf", 0.0) >= conf_thresh]

def detections_to_payload(frame_id: int, ts: float, detections: list) -> dict:
    """Package detections into the standard MQTT JSON schema."""
    return {
        "frame": frame_id,
        "ts": ts,
        "detections": detections
    }

# --- Graceful shutdown + Docker health check heartbeat ---
running = True

def signal_handler(sig: int, frame: object) -> None:
    """Handle SIGTERM/SIGINT for graceful shutdown."""
    global running
    print(f"\n[inference] Received signal {sig}, shutting down...")
    running = False

def write_health() -> None:
    """Timestamp heartbeat for Docker HEALTHCHECK.
    
    Silently no-ops if /tmp isn't writable.
    """
    try:
        with open("/tmp/inference_health", "w") as f:
            f.write(str(time.time()))
    except OSError:
        pass

def main() -> None:
    """Run the main inference loop."""
    parser = argparse.ArgumentParser(description="YOLO26 TensorRT inference node")
    parser.add_argument(
        "--model",
        default="/opt/models/best.engine",
        help="Path to TensorRT engine (built at image-build time)",
    )
    parser.add_argument(
        "--source", default="/opt/data/test_video.mp4", help="Video file or camera index"
    )
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--mqtt-broker", default=os.getenv("MQTT_BROKER", "localhost"))
    parser.add_argument("--mqtt-port", type=int, default=int(os.getenv("MQTT_PORT", "1883")))
    parser.add_argument("--mqtt-topic", default="/sense/vision/detections")
    args = parser.parse_args()

    # Load model. 
    print(f"[inference] Loading model: {args.model}")
    # Use the factory function since YOLO isn't imported at the top level
    # model = YOLO(args.model, task="detect") 
    model = _default_model_factory(args.model, task="detect")

    # NEW CODE: Set up the publisher configuration
    mqtt_config = PublisherConfig(
        # Use the arguments instead of hardcoding
        host=args.mqtt_broker, 
        port=args.mqtt_port,
        client_id="hw6_inference_node"
    )
    
    # Instantiate and connect
    publisher = MqttPublisher(config=mqtt_config)
    publisher.connect()

    # Open video
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"[inference] ERROR: Cannot open source: {args.source}")
        sys.exit(1)

    frame_count = 0
    fps_start = time.monotonic()
    print(f"[inference] Running inference on {args.source}...")

    while running:
        ret, frame = cap.read()
        if not ret:
            # Loop video for continuous testing
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if not ret:
                break

        results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)

        # Build detection payload
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append(
                    {
                        "class": r.names[int(box.cls)],
                        "confidence": round(float(box.conf), 3),
                        "bbox": [round(float(x), 1) for x in box.xyxy[0].tolist()],
                    }
                )

        payload = {
            "t": round(time.time(), 3),
            "frame": frame_count,
            "detections": detections,
            "count": len(detections),
        }
        
        # Use the requested topic instead of a hardcoded string
        # publisher.publish("jetson/vision/detections", payload)
        publisher.publish(args.mqtt_topic, payload)
        frame_count += 1

        # Heartbeat for Docker HEALTHCHECK
        if frame_count % 10 == 0:
            write_health()

        if frame_count % 100 == 0:
            elapsed = time.monotonic() - fps_start
            fps = frame_count / elapsed if elapsed > 0 else 0
            print(
                f"[inference] {frame_count} frames, {fps:.1f} FPS, "
                f"last frame: {len(detections)} detections"
            )

    # Cleanup
    cap.release()
    # The old 'client' is gone, tell the publisher to disconnect instead
    # client.loop_stop()
    # client.disconnect()
    publisher.disconnect()
    print(f"[inference] Shutdown complete. Processed {frame_count} frames.")

if __name__ == "__main__":
    main()
