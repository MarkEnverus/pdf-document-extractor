"""Kafka client for async message production and consumption."""

from lib.kafka.config import KafkaConfig
from lib.kafka.kafka import Kafka

__all__ = ["Kafka", "KafkaConfig"]
