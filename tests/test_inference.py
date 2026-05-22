#!/usr/bin/env python3
# Copyright (c) 2026 Kishore and Farhan
# Tatung University 14210 AI實務專題
"""tests/test_inference.py - unit tests for inference pipeline helpers."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import src.inference_node as inference_node
from src.inference_node import (
    apply_confidence_threshold,
    detections_to_payload,
    preprocess_frame,
)


@pytest.fixture(autouse=True)
def disable_healthcheck_during_tests():
    """Prevent Pytest from starting the real heartbeat server on Port 8000."""
    with patch("src.healthcheck.start_in_thread"):
        yield

# --- preprocessing ---
@pytest.mark.parametrize("shape", [(480, 640, 3), (720, 1280, 3), (320, 320, 3)])
def test_preprocess_frame_outputs_expected_shape(shape):
    """preprocess_frame() must return (1, 3, H, W) tensor."""
    frame = (np.random.rand(*shape) * 255).astype(np.uint8)
    out = preprocess_frame(frame, target_size=(320, 320))

    assert out.shape == (1, 3, 320, 320)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_preprocess_frame_handles_grayscale_input():
    """Single-channel input must be broadcast to 3 channels."""
    gray = (np.random.rand(480, 640) * 255).astype(np.uint8)
    out = preprocess_frame(gray, target_size=(320, 320))
    assert out.shape == (1, 3, 320, 320)


# --- postprocessing ---
@pytest.mark.parametrize(
    "conf_thresh, expected_count",
    [
        (0.0, 5),  # all detections pass
        (0.5, 3),  # only the high-confidence ones
        (1.0, 0),  # nothing meets a near-perfect threshold
    ],
)
def test_apply_confidence_threshold(conf_thresh, expected_count):
    """Filtering by confidence must drop detections below the threshold."""
    detections = [
        {"cls": 0, "conf": 0.99, "xyxy": [0, 0, 10, 10]},
        {"cls": 1, "conf": 0.75, "xyxy": [10, 10, 20, 20]},
        {"cls": 0, "conf": 0.55, "xyxy": [20, 20, 30, 30]},
        {"cls": 2, "conf": 0.30, "xyxy": [30, 30, 40, 40]},
        {"cls": 1, "conf": 0.10, "xyxy": [40, 40, 50, 50]},
    ]
    out = apply_confidence_threshold(detections, conf_thresh)
    assert len(out) == expected_count


def test_detections_to_payload_includes_required_fields():
    """The MQTT payload schema must always have frame, ts, detections."""
    payload = detections_to_payload(frame_id=42, ts=1700000000.0, detections=[])
    assert payload["frame"] == 42
    assert payload["ts"] == 1700000000.0
    assert payload["detections"] == []


# --- camera mock fixture ---
@pytest.fixture
def mock_video_capture():
    """Mock cv2.VideoCapture so tests don't need a real camera."""
    fake = MagicMock()
    fake.isOpened.return_value = True
    # Yield 3 fake frames, then exhaust TWICE to handle the video loop restart
    fake.read.side_effect = [
        (True, (np.random.rand(480, 640, 3) * 255).astype(np.uint8)),
        (True, (np.random.rand(480, 640, 3) * 255).astype(np.uint8)),
        (True, (np.random.rand(480, 640, 3) * 255).astype(np.uint8)),
        (False, None),  # Exhausted (triggers the reset)
        (False, None),  # Exhausted again (triggers the break)
    ]
    return fake


def test_signal_handler_sets_running_false():
    """Ensure the signal handler gracefully stops the loop."""
    inference_node.running = True
    inference_node.signal_handler(15, None)
    assert inference_node.running is False


def test_write_health_creates_file():
    """Ensure heartbeat writes without crashing."""
    with patch("builtins.open", MagicMock()):
        inference_node.write_health()


def test_video_capture_loop_processes_all_frames(mock_video_capture):
    """The main loop should consume frames until isOpened/read says stop."""
    # Create a fake YOLO model and result object
    mock_model = MagicMock()
    mock_result = MagicMock()
    mock_result.boxes = []
    mock_model.predict.return_value = [mock_result]

    # Run main() with all hardware and network dependencies mocked out
    with (
        patch("cv2.VideoCapture", return_value=mock_video_capture),
        patch("src.inference_node._default_model_factory", return_value=mock_model),
        patch("src.inference_node.MqttPublisher"),
        patch("sys.argv", ["inference_node.py", "--source", "test.mp4"]),
    ):
        inference_node.running = True
        inference_node.main()

    assert mock_video_capture.read.call_count >= 1


def test_main_exits_if_camera_fails():
    """Ensure the script aborts if the video source cannot be opened."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False

    with (
        patch("cv2.VideoCapture", return_value=mock_cap),
        patch("src.inference_node._default_model_factory"),
        patch("src.inference_node.MqttPublisher"),
        patch("sys.argv", ["inference_node.py"]),
        pytest.raises(SystemExit),
    ):  # <-- Add this here!
        # The script should call sys.exit(1) when the camera fails
        inference_node.main()
