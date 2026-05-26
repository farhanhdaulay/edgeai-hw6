#!/usr/bin/env python3
# Copyright (c) 2026 Farhan Hikmatullah Daulay - 611451002
# Tatung University 14210 AI實務專題
"""tests/test_mqtt.py - unit tests for MqttPublisher."""

import json
from unittest.mock import MagicMock

import paho.mqtt.client as mqtt
import pytest

from src.mqtt_publisher import MqttPublisher, PublisherConfig


@pytest.fixture
def mock_client():
    """Return a MagicMock that behaves enough like paho.mqtt.client.Client."""
    client = MagicMock(spec=mqtt.Client)
    # publish() returns a paho-mqtt MQTTMessageInfo-like object
    info = MagicMock()
    info.rc = mqtt.MQTT_ERR_SUCCESS
    client.publish.return_value = info
    return client


@pytest.fixture
def publisher(mock_client):
    """Build a publisher wired to the mock client (no real network)."""
    return MqttPublisher(PublisherConfig(host="test"), client_factory=lambda: mock_client)


def test_publish_sends_json_payload(publisher, mock_client):
    """publish() must JSON-encode dicts and call client.publish()."""
    # Force connected state for the test
    publisher._connected = True
    payload = {"frame": 1, "ts": 1234567890.0}

    assert publisher.publish("jetson/vision/detections", payload) is True

    args, _ = mock_client.publish.call_args
    topic, body = args
    assert topic == "jetson/vision/detections"
    assert json.loads(body) == payload


def test_publish_when_disconnected_returns_false(publisher, mock_client):
    """publish() before connect() must NOT raise - just return False."""
    assert publisher.connected is False
    assert publisher.publish("any/topic", {"x": 1}) is False
    mock_client.publish.assert_not_called()


def test_publish_string_payload_is_passed_through(publisher, mock_client):
    """If caller passes a str, don't double-JSON-encode it."""
    publisher._connected = True
    publisher.publish("topic", "already-a-string")

    args, _ = mock_client.publish.call_args
    assert args[1] == "already-a-string"  # not '"already-a-string"'


def test_disconnect_stops_loop(publisher, mock_client):
    publisher._connected = True
    publisher.disconnect()

    mock_client.loop_stop.assert_called_once()
    mock_client.disconnect.assert_called_once()
    assert publisher.connected is False


def test_reconnect_delays_set(publisher, mock_client):
    """Verify the publisher configured paho's exponential reconnect."""
    mock_client.reconnect_delay_set.assert_called_once()


def test_connect_success(publisher, mock_client):
    assert publisher.connect() is True
    mock_client.connect.assert_called_once()
    assert publisher.connected is True


def test_connect_exception_returns_false(publisher, mock_client):
    mock_client.connect.side_effect = Exception("Network down")
    assert publisher.connect() is False
    assert publisher.connected is False


def test_publish_exception_returns_false(publisher, mock_client):
    publisher._connected = True
    mock_client.publish.side_effect = Exception("Broker dropped connection")
    assert publisher.publish("test/topic", {"data": 1}) is False
