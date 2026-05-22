#!/usr/bin/env python3
# Copyright (c) 2026 Kishore and Farhan
# Tatung University 14210 AI實務專題
"""tests/integration/test_jetson_e2e.py.

Integration test that runs on the Jetson hardware. Pulls the image, starts the container,
and verifies that MQTT detections are published.
"""

import json
import os
import subprocess
import time
from pathlib import Path

import paho.mqtt.client as mqtt
import pytest
from paho.mqtt.enums import CallbackAPIVersion

IMAGE = os.environ.get("IMAGE")

@pytest.fixture(scope="module")
def inference_container():
    """Pull the image, start container, and safely clean up afterwards."""
    assert IMAGE, "IMAGE environment variable must be set"

    # 1. Pull the per-commit image
    subprocess.run(["docker", "pull", IMAGE], check=True)

    container_name = "hw6_integration_test"
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

    sample_frame = Path(__file__).parent / "sample_frame.jpg"

    # 2. Start container WITHOUT --rm so we can capture crash logs
    cmd = [
        "docker",
        "run",
        "-d",
        "--name",
        container_name,
        "--runtime",
        "nvidia",
        "-v",
        "lab12-models:/opt/models",
        "-v",
        f"{sample_frame.resolve()}:/app/sample_frame.jpg:ro",
        "--network",
        "host",
        IMAGE,
        "python3",
        "/app/inference_node.py",
        "--source",
        "/app/sample_frame.jpg",
    ]
    subprocess.run(cmd, check=True)

    yield container_name

    # 3. Cleanup on failure/success
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)


def test_image_is_per_commit_sha_tagged():
    """Verify we are testing the right image."""
    assert IMAGE is not None
    assert "sha-" in IMAGE


def test_inference_publishes_mqtt_within_window(inference_container):
    """Subscribe to MQTT and wait for a detection payload."""
    messages = []

    def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            if "detections" in payload:
                messages.append(payload)
        except Exception:
            pass

    client = mqtt.Client(callback_api_version=CallbackAPIVersion.VERSION2)
    client.on_message = on_message
    client.connect("localhost", 1883, 60)
    client.subscribe("jetson/vision/detections")
    client.loop_start()

    # TEMPORARY FAST DEBUG: Wait only 30s to quickly catch the crash logs
    timeout = time.time() + 30
    found = False
    while time.time() < timeout:
        if len(messages) > 0:
            found = True
            break
        time.sleep(1)

    client.loop_stop()
    client.disconnect()

    # If it failed, extract and print the container's dying words!
    if not found:
        logs = subprocess.run(
            ["docker", "logs", inference_container],
            capture_output=True,
            text=True,
        )
        print("\n=== DOCKER CONTAINER LOGS ===")
        print(logs.stdout)
        print(logs.stderr)
        print("=============================\n")

    # 4. Assert an MQTT message lands
    assert found, "No MQTT detections received within the timeout window"
