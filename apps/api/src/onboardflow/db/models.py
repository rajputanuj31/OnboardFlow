"""Relational schema.

Design notes:
- A snapshot is one repo at one commit. It is shared by every user who asks about that
  commit (UNIQUE repo_id + commit_sha), so re-ingesting an unchanged commit is free.
- File contents are content-addressed (`blobs`, keyed by sha256). Unchanged files across
  commits share a blob, so later stages (parsing, embedding) never redo work for them.
- `snapshot_files` maps paths to blobs per snapshot; that mapping is what changes between
  commits.
- Jobs live in Postgres and are claimed with `FOR UPDATE SKIP LOCKED` plus a lease, so a
  crashed worker's job is picked up again when its lease expires.
"""

from __future__ import annotations

import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class SnapshotStatus(enum.StrEnum):
    QUEUED = "queued"
    CLONING = "cloning"
    SCANNING = "scanning"
    READY = "ready"
    FAILED = "failed"


class JobStatus(enum.StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


def _enum(cls: type[enum.StrEnum], name: str) -> Enum:
    return Enum(cls, name=name, values_callable=lambda e: [m.value for m in e])


class Repo(Base):
    __tablename__ = "repos"
    __table_args__ = (
        # GitHub owner/repo names are case-insensitive.
        Index("uq_repos_owner_name_ci", text("lower(owner)"), text("lower(name)"), unique=True),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner: Mapped[str] = mapped_column(String(100))
    name: Mapped[str] = mapped_column(String(100))
    default_branch: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Snapshot(Base):
    __tablename__ = "snapshots"
    __table_args__ = (UniqueConstraint("repo_id", "commit_sha", name="uq_snapshots_repo_commit"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("repos.id", ondelete="CASCADE"))
    commit_sha: Mapped[str] = mapped_column(String(40))
    status: Mapped[SnapshotStatus] = mapped_column(
        _enum(SnapshotStatus, "snapshot_status"), default=SnapshotStatus.QUEUED
    )
    progress_done: Mapped[int] = mapped_column(Integer, default=0)
    progress_total: Mapped[int] = mapped_column(Integer, default=0)
    file_count: Mapped[int | None] = mapped_column(Integer)
    indexed_file_count: Mapped[int | None] = mapped_column(Integer)
    total_bytes: Mapped[int | None] = mapped_column(BigInteger)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Blob(Base):
    __tablename__ = "blobs"

    sha256: Mapped[str] = mapped_column(String(64), primary_key=True)
    language: Mapped[str | None] = mapped_column(String(40))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    line_count: Mapped[int | None] = mapped_column(Integer)
    parsed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class SnapshotFile(Base):
    __tablename__ = "snapshot_files"
    __table_args__ = (Index("ix_snapshot_files_blob", "blob_sha256"),)

    snapshot_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("snapshots.id", ondelete="CASCADE"), primary_key=True
    )
    path: Mapped[str] = mapped_column(Text, primary_key=True)
    blob_sha256: Mapped[str] = mapped_column(ForeignKey("blobs.sha256"))
    indexed: Mapped[bool] = mapped_column(Boolean)
    skip_reason: Mapped[str | None] = mapped_column(String(40))


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (Index("ix_jobs_claimable", "status", "run_after"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    kind: Mapped[str] = mapped_column(String(40))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB)
    status: Mapped[JobStatus] = mapped_column(
        _enum(JobStatus, "job_status"), default=JobStatus.QUEUED
    )
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    run_after: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    lease_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
