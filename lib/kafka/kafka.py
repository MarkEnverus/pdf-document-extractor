import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer  # type: ignore[import-untyped]
from aiokafka.errors import KafkaError  # type: ignore[import-untyped]
from lib.logger import Logger
from pydantic import BaseModel
from pydantic_core import to_json

from lib.kafka.config import KafkaConfig

logger = Logger.get_logger(__name__)


class Kafka:
    def __init__(self, kafka_config: KafkaConfig) -> None:
        self._producer: AIOKafkaProducer | None = None
        self._consumer: AIOKafkaConsumer | None = None
        self._kafka_config = kafka_config

    async def start_producer(self) -> None:
        if self._producer is None:
            logger.info(
                f"Starting Kafka producer with bootstrap_servers={self._kafka_config.bootstrap_servers}"
            )
            try:
                self._producer = AIOKafkaProducer(
                    value_serializer=lambda v: to_json(v),
                    **self._kafka_config.to_aiokafka_config(),
                )
                logger.info("Producer created, starting connection...")
                await asyncio.wait_for(self._producer.start(), timeout=15.0)
                logger.info("Kafka producer started successfully")
            except TimeoutError:
                logger.error(
                    f"Kafka producer startup timed out after 15s. "
                    f"Bootstrap servers: {self._kafka_config.bootstrap_servers}. "
                    f"This may indicate a network connectivity issue or incorrect hostname."
                )
                await self.stop_producer()
                raise
            except Exception as e:
                logger.error(
                    "Kafka producer failed to start",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                await self.stop_producer()
                raise

    async def stop_producer(self) -> None:
        if self._producer:
            logger.info("Stopping Kafka producer")
            await self._producer.stop()
            self._producer = None
            logger.info("Kafka producer stopped")

    async def start_consumer(
        self, topics: list[str], group_id: str | None = None
    ) -> None:
        if self._consumer is None:
            consumer_group = group_id or self._kafka_config.consumer_group_id
            logger.info(
                "Starting Kafka consumer",
                topics=topics,
                consumer_group=consumer_group,
            )
            try:
                self._consumer = AIOKafkaConsumer(
                    *topics,
                    group_id=consumer_group,
                    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                    **self._kafka_config.to_aiokafka_consumer_config(),
                )
                logger.debug(
                    "Kafka consumer instance created",
                    topics=topics,
                    consumer_group=consumer_group,
                )
                await self._consumer.start()
                logger.info(
                    "Kafka consumer started",
                    topics=topics,
                    consumer_group=consumer_group,
                )
            except Exception as e:
                logger.error(
                    "Kafka consumer failed to start",
                    error=str(e),
                    error_args=e.args,
                    error_type=type(e).__name__,
                )
                await self.stop_consumer()
                raise

    async def stop_consumer(self) -> None:
        if self._consumer:
            logger.info("Stopping Kafka consumer")
            await self._consumer.stop()
            self._consumer = None
            logger.info("Kafka consumer stopped")

    async def produce(
        self, topic: str, message: BaseModel | dict[str, Any], key: str | bytes
    ) -> None:
        if self._producer is None:
            raise RuntimeError("Producer not started")

        try:
            if isinstance(message, BaseModel):
                message_data = message.model_dump(mode="json")
            else:
                message_data = message

            key_bytes = key if isinstance(key, bytes) else key.encode("utf-8")
            await self._producer.send_and_wait(topic, message_data, key=key_bytes)
            logger.info(
                f"Produced message topic={topic} key={key_bytes.decode('utf-8')}"
            )
        except KafkaError as e:
            key_str = (
                key if isinstance(key, str) else key.decode("utf-8", errors="ignore")
            )
            logger.error(
                f"Failed to produce message topic={topic} key={key_str} error={e} args={e.args}"
            )
            raise

    async def consume(
        self, handler: Callable[[str, dict[str, Any], dict[str, Any]], Awaitable[None]]
    ) -> None:
        if self._consumer is None:
            raise RuntimeError("Consumer not started")

        try:
            async for msg in self._consumer:
                try:
                    logger.info(
                        "Processing Kafka message",
                        topic=msg.topic,
                        partition=msg.partition,
                        offset=msg.offset,
                        timestamp=msg.timestamp,
                    )
                    # Pass partition/offset metadata to handler
                    metadata = {
                        "partition": msg.partition,
                        "offset": msg.offset,
                        "timestamp": msg.timestamp,
                        "key": msg.key.decode("utf-8") if msg.key else None,
                    }
                    await handler(msg.topic, msg.value, metadata)
                except Exception as e:
                    logger.error(
                        "Consumer handler error",
                        topic=msg.topic,
                        partition=msg.partition,
                        offset=msg.offset,
                        timestamp=msg.timestamp,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
        except KafkaError as e:
            logger.error(
                "Kafka consumer failed", error=str(e), error_type=type(e).__name__
            )
            raise

    async def commit(self) -> None:
        """
        Manually commit current offsets.

        Only effective when enable_auto_commit=False in KafkaConfig.
        Commits offsets for all consumed messages up to the current position.

        Raises:
            RuntimeError: If consumer is not started
            KafkaError: If commit fails
        """
        if self._consumer is None:
            raise RuntimeError("Consumer not started")
        try:
            await self._consumer.commit()
            logger.info("Kafka offsets committed successfully")
        except KafkaError as e:
            logger.error(
                "Failed to commit Kafka offsets",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    async def seek_back(self, messages: int = 10) -> None:
        """
        Seek back N messages from current position on all assigned partitions.

        Useful for local development to replay recent messages.

        Args:
            messages: Number of messages to seek back (default 10)

        Raises:
            RuntimeError: If consumer is not started
        """
        if self._consumer is None:
            raise RuntimeError("Consumer not started")

        try:
            # Get all assigned partitions
            partitions = self._consumer.assignment()
            if not partitions:
                logger.warning(
                    "No partitions assigned to consumer yet - cannot seek back"
                )
                return

            logger.info(
                "Seeking back messages on partitions",
                messages=messages,
                partition_count=len(partitions),
            )

            for partition in partitions:
                # Get current position (next offset to be consumed)
                current_offset = await self._consumer.position(partition)

                # Calculate new offset (ensure it doesn't go below 0)
                new_offset = max(0, current_offset - messages)

                # Seek to new position
                self._consumer.seek(partition, new_offset)

                logger.info(
                    "Seeked back on partition",
                    topic=partition.topic,
                    partition=partition.partition,
                    from_offset=current_offset,
                    to_offset=new_offset,
                )

            logger.info("Successfully seeked back messages", messages=messages)

        except Exception as e:
            logger.error(
                "Failed to seek back messages",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise

    async def safe_start_producer(self) -> None:
        try:
            await self.start_producer()
        except Exception as e:
            logger.error(
                "Producer safe start failed",
                error=str(e),
                error_type=type(e).__name__,
            )

    async def safe_start_consumer(
        self, topics: list[str], group_id: str | None = None
    ) -> None:
        try:
            await self.start_consumer(topics, group_id)
        except Exception as e:
            logger.error(
                "Consumer safe start failed",
                error=str(e),
                error_type=type(e).__name__,
            )

    async def safe_stop_producer(self) -> None:
        try:
            await self.stop_producer()
        except Exception as e:
            logger.error(
                "Producer safe stop failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            self._producer = None

    async def safe_stop_consumer(self) -> None:
        try:
            await self.stop_consumer()
        except Exception as e:
            logger.error(
                "Consumer safe stop failed",
                error=str(e),
                error_type=type(e).__name__,
            )
            self._consumer = None

    async def safe_stop(self) -> None:
        await self.safe_stop_producer()
        await self.safe_stop_consumer()

    def is_producer_initialized(self) -> bool:
        return bool(self._producer and not self._producer._closed)
