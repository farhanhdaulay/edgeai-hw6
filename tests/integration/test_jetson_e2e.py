#!/usr/bin/env python3
# Copyright (c) 2026 Kishore and Farhan
# Tatung University 14210 AI實務專題
"""tests/integration/test_jetson_e2e.py."""

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

    subprocess.run(["docker", "pull", IMAGE], check=True)
    container_name = "hw6_integration_test"
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)

    sample_frame = Path(__file__).parent / "sample_frame.jpg"

    # Execute natively, then sleep the Python interpreter to flush the MQTT buffer!
    python_cmd = (
        "import os, sys, runpy, time; "
        "os.chdir('/app'); "
        "sys.argv=['inference_node.py', '--source', '/app/sample_frame.jpg']; "
        "runpy.run_path('/app/inference_node.py', run_name='__main__'); "
        "time.sleep(3)"
    )

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
        "-e",
        "MQTT_HOST=127.0.0.1",
        "-e",
        "MQTT_PORT=1883",
        IMAGE,
        "python3",
        "-c",
        python_cmd,
    ]
    subprocess.run(cmd, check=True)

    yield container_name
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)


def test_image_is_per_commit_sha_tagged():
    """Verify we are testing the right image."""
    assert IMAGE is not None
    assert "sha-" in IMAGE


def test_inference_publishes_mqtt_within_window(inference_container):
    """Subscribe to MQTT and wait for a detection payload."""
    messages = []

    def on_message(client, userdata, msg):
        messages.append(msg.payload.decode())

    client = mqtt.Client(callback_api_version=CallbackAPIVersion.VERSION2)
    client.on_message = on_message
    client.connect("127.0.0.1", 1883, 60)

    # Subscribe to literally everything on the broker
    client.subscribe("#")
    client.loop_start()

    timeout = time.time() + 30
    found = False
    while time.time() < timeout:
        if len(messages) > 0:
            found = True
            break
        time.sleep(1)

    client.loop_stop()
    client.disconnect()

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

    assert found, "No MQTT detections received within the timeout window"
