"""Add a repository: resolve its current commit and ingest it once (shared by all users).

POST /repos {url}
  - 200 with the snapshot if that commit is already indexed (free, instant),
  - 202 with a queued snapshot otherwise; follow progress at /snapshots/{id}/events.
Exactly one snapshot and one ingest job exist per (repo, commit), even under concurrent
requests: the snapshot insert and the job insert happen in one transaction, guarded by the
UNIQUE (repo_id, commit_sha) constraint.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel
from sqlalchemy import select, text
from sqlalchemy.dialects.postgresql import insert

from onboardflow.db.models import Repo, Snapshot, SnapshotStatus
from onboardflow.github import (
    GitHubClient,
    GitHubUnavailableError,
    InvalidRepoURLError,
    RepoNotFoundError,
    parse_repo_url,
)
from onboardflow.ingest.pipeline import INGEST_JOB
from onboardflow.jobs import new_job
from onboardflow.routers.snapshots import SnapshotOut, snapshot_out
from onboardflow.settings import Settings

router = APIRouter(tags=["repos"])


class AddRepoIn(BaseModel):
    url: str


@router.post(
    "/repos",
    status_code=status.HTTP_202_ACCEPTED,
    responses={200: {"model": SnapshotOut}, 400: {}, 404: {}, 413: {}, 502: {}},
)
async def add_repo(body: AddRepoIn, request: Request, response: Response) -> SnapshotOut:
    settings: Settings = request.app.state.settings
    github: GitHubClient = request.app.state.github
    try:
        info = await github.repo_info(parse_repo_url(body.url))
    except InvalidRepoURLError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except RepoNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except GitHubUnavailableError as exc:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, "GitHub is unavailable, try again."
        ) from exc
    if info.size_kb > settings.ingest.max_repo_mb * 1024:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"Repository is {info.size_kb // 1024} MB; "
            f"the limit is {settings.ingest.max_repo_mb} MB.",
        )

    async with request.app.state.sessions.begin() as session:
        repo_id = (
            await session.execute(
                insert(Repo)
                .values(
                    id=uuid.uuid4(),
                    owner=info.ref.owner,
                    name=info.ref.name,
                    default_branch=info.default_branch,
                )
                .on_conflict_do_update(
                    index_elements=[text("lower(owner)"), text("lower(name)")],
                    set_={
                        "owner": info.ref.owner,
                        "name": info.ref.name,
                        "default_branch": info.default_branch,
                    },
                )
                .returning(Repo.id)
            )
        ).scalar_one()
        new_snapshot_id = (
            await session.execute(
                insert(Snapshot)
                .values(
                    id=uuid.uuid4(),
                    repo_id=repo_id,
                    commit_sha=info.head_sha,
                    status=SnapshotStatus.QUEUED,
                    progress_done=0,
                    progress_total=0,
                )
                .on_conflict_do_nothing(constraint="uq_snapshots_repo_commit")
                .returning(Snapshot.id)
            )
        ).scalar_one_or_none()
        if new_snapshot_id is not None:
            session.add(
                new_job(
                    INGEST_JOB, {"snapshot_id": str(new_snapshot_id)}, settings.worker.max_attempts
                )
            )
        snapshot = (
            await session.execute(
                select(Snapshot).where(
                    Snapshot.repo_id == repo_id, Snapshot.commit_sha == info.head_sha
                )
            )
        ).scalar_one()

    if snapshot.status == SnapshotStatus.READY:
        response.status_code = status.HTTP_200_OK
    return snapshot_out(snapshot, info.ref.owner, info.ref.name)
