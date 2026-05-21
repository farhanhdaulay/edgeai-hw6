#!/usr/bin/env python3
# Copyright (c) 2026 Kishore and Farhan
# Tatung University 14210 AI實務專題
"""src/mqtt_publisher.py - Thin paho-mqtt wrapper with reconnect + JSON encoding.

Pulled out of inference_node so it's unit-testable without starting a real broker.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, Optional

import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion

@dataclass
class PublisherConfig:
    host: str = "localhost"
    port: int = 1883
    keepalive: int = 60
    client_id: str = ""
    reconnect_min_delay: float = 1.0
    reconnect_max_delay: float = 30.0

class MqttPublisher:
    """Publish JSON messages to MQTT with automatic reconnection."""
    
    def __init__(self, config: PublisherConfig,
                 client_factory: Optional[Callable[[], mqtt.Client]] = None):
        self.config = config
        factory = client_factory or (lambda: mqtt.Client(
            callback_api_version=CallbackAPIVersion.VERSION2,
            client_id=config.client_id,
        ))
        self.client = factory()
        
        # Configure exponential reconnect delays
        self.client.reconnect_delay_set(
            min_delay=self.config.reconnect_min_delay, 
            max_delay=self.config.reconnect_max_delay
        )
        self._connected = False

    def connect(self, timeout: float = 5.0) -> bool:
        """Connect to the broker and start the background thread."""
        try:
            self.client.connect(self.config.host, self.config.port, self.config.keepalive)
            self.client.loop_start()
            self._connected = True
            return True
        except Exception:
            return False

    def publish(self, topic: str, payload: Any) -> bool:
        """Publish a message. Dicts are automatically JSON-encoded."""
        if not self._connected:
            return False
            
        try:
            # If caller passes a str, don't double-JSON-encode it
            if isinstance(payload, str):
                out_payload = payload
            else:
                out_payload = json.dumps(payload)
                
            self.client.publish(topic, out_payload)
            return True
        except Exception:
            return False

    def disconnect(self) -> None:
        """Stop the loop and disconnect."""
        self.client.loop_stop()
        self.client.disconnect()
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected