from typing import Any

from pydantic import BaseModel


class KafkaConfig(BaseModel):
    bootstrap_servers: str
    security_protocol: str = "PLAINTEXT"
    sasl_mechanism: str | None = None
    sasl_plain_username: str | None = None
    sasl_plain_password: str | None = None
    consumer_group_id: str | None = None
    fetch_timeout_ms: int = 30000
    # Consumer configuration for commit and timeout behavior
    enable_auto_commit: bool = True  # Default to auto-commit (services requiring manual commits should set False)
    auto_offset_reset: str = "earliest"  # Where to start if no committed offset
    session_timeout_ms: int = 10000  # Time before consumer is considered dead
    heartbeat_interval_ms: int = (
        3000  # Heartbeat interval (should be < session_timeout_ms / 3)
    )
    max_poll_interval_ms: int = 300000  # Max time between polls (5 minutes default)
    max_poll_records: int = 500  # Max records returned in single poll
    # Additional timeout configuration for WarpStream coordinator latency
    request_timeout_ms: int = (
        120000  # 2 minutes for coordinator operations (OffsetFetch, JoinGroup, etc.)
    )
    connections_max_idle_ms: int = 540000  # 9 minutes - keep connections alive longer

    def to_aiokafka_config(self) -> dict[str, Any]:
        config = {
            "bootstrap_servers": self.bootstrap_servers,
            "request_timeout_ms": self.request_timeout_ms,
            "connections_max_idle_ms": self.connections_max_idle_ms,
        }

        if self.security_protocol != "PLAINTEXT":
            config["security_protocol"] = self.security_protocol

            if self.security_protocol == "SASL_SSL":
                raise NotImplementedError(
                    "SASL_SSL security protocol is not yet implemented. "
                    "Please use PLAINTEXT or implement proper SSL/TLS verification."
                )

        if self.sasl_mechanism:
            config["sasl_mechanism"] = self.sasl_mechanism

        if self.sasl_plain_username and self.sasl_plain_password:
            config["sasl_plain_username"] = self.sasl_plain_username
            config["sasl_plain_password"] = self.sasl_plain_password

        return config

    def to_aiokafka_consumer_config(self) -> dict[str, object]:
        config: dict[str, object] = self.to_aiokafka_config()
        config.update(
            {
                "fetch_max_wait_ms": self.fetch_timeout_ms,
                "auto_offset_reset": self.auto_offset_reset,
                "enable_auto_commit": self.enable_auto_commit,
                "session_timeout_ms": self.session_timeout_ms,
                "heartbeat_interval_ms": self.heartbeat_interval_ms,
                "max_poll_interval_ms": self.max_poll_interval_ms,
                "max_poll_records": self.max_poll_records,
            }
        )
        return config
