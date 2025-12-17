#!/usr/bin/env python3
"""
Storage adapters for training data output.

Supports local filesystem and S3 with a common interface.

Usage:
    # Local storage (default)
    storage = LocalStorage("output/")
    storage.save("training.jsonl", data)

    # S3 storage
    storage = S3Storage("my-bucket", "training-data/")
    storage.save("training.jsonl", data)
"""

import json
import os
import socket
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional


def get_machine_id() -> str:
    """Get a unique identifier for this machine."""
    username = os.environ.get("USER", os.environ.get("USERNAME", "unknown"))
    hostname = socket.gethostname().split(".")[0]
    return f"{username}-{hostname}"


class StorageAdapter(ABC):
    """Abstract base class for storage adapters."""

    @abstractmethod
    def save(self, filename: str, data: bytes) -> str:
        """Save data to storage. Returns the full path/URL."""
        pass

    @abstractmethod
    def list_files(self, prefix: str = "") -> list[str]:
        """List files in storage with optional prefix filter."""
        pass

    @abstractmethod
    def load(self, filename: str) -> bytes:
        """Load data from storage."""
        pass

    @abstractmethod
    def exists(self, filename: str) -> bool:
        """Check if file exists."""
        pass


class LocalStorage(StorageAdapter):
    """Local filesystem storage."""

    def __init__(self, base_path: str = "output"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(self, filename: str, data: bytes) -> str:
        path = self.base_path / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return str(path)

    def list_files(self, prefix: str = "") -> list[str]:
        if prefix:
            return [str(p.name) for p in self.base_path.glob(f"{prefix}*")]
        return [str(p.name) for p in self.base_path.iterdir() if p.is_file()]

    def load(self, filename: str) -> bytes:
        return (self.base_path / filename).read_bytes()

    def exists(self, filename: str) -> bool:
        return (self.base_path / filename).exists()


class S3Storage(StorageAdapter):
    """AWS S3 storage."""

    def __init__(self, bucket: str, prefix: str = ""):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client("s3")
            except ImportError:
                raise ImportError("boto3 required for S3 storage: pip install boto3")
        return self._client

    def _key(self, filename: str) -> str:
        if self.prefix:
            return f"{self.prefix}/{filename}"
        return filename

    def save(self, filename: str, data: bytes) -> str:
        key = self._key(filename)
        self.client.put_object(Bucket=self.bucket, Key=key, Body=data)
        return f"s3://{self.bucket}/{key}"

    def list_files(self, prefix: str = "") -> list[str]:
        full_prefix = self._key(prefix)
        response = self.client.list_objects_v2(Bucket=self.bucket, Prefix=full_prefix)
        files = []
        for obj in response.get("Contents", []):
            key = obj["Key"]
            # Remove the base prefix to get just the filename
            if self.prefix:
                key = key[len(self.prefix) + 1:]
            files.append(key)
        return files

    def load(self, filename: str) -> bytes:
        key = self._key(filename)
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()

    def exists(self, filename: str) -> bool:
        try:
            key = self._key(filename)
            self.client.head_object(Bucket=self.bucket, Key=key)
            return True
        except:
            return False


def get_storage(storage_type: str = "local", **kwargs) -> StorageAdapter:
    """
    Factory function to get storage adapter.

    Args:
        storage_type: "local" or "s3"
        **kwargs: Additional arguments for the storage adapter
            - local: base_path (default: "output")
            - s3: bucket (required), prefix (optional)

    Returns:
        StorageAdapter instance
    """
    if storage_type == "local":
        return LocalStorage(kwargs.get("base_path", "output"))
    elif storage_type == "s3":
        bucket = kwargs.get("bucket")
        if not bucket:
            bucket = os.environ.get("DISTILLER_S3_BUCKET")
        if not bucket:
            raise ValueError("S3 bucket required: --s3-bucket or DISTILLER_S3_BUCKET env var")
        return S3Storage(bucket, kwargs.get("prefix", ""))
    else:
        raise ValueError(f"Unknown storage type: {storage_type}")
