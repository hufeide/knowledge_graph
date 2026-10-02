"""FastAPI backend exposing the multi-agent KG pipeline."""
from __future__ import annotations

import asyncio
import io
import re
import threading
import time
import uuid
from datetime import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel
from .utils import decode_text_bytes

from .config import config
from .supervisor import BuildCancelled, Supervisor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND = PROJECT_ROOT / "frontend" / "index.html"

app = FastAPI(title="多 Agent 知识图谱构建服务", version="1.0.0")

_supervisor: Optional[Supervisor] = None

# ---- build job registry: server-side source of truth for the UI queue ----
# Tasks live here so a browser refresh — which wipes JS state and the SSE
# connection — can recover in-progress / finished builds by polling /api/jobs.
@dataclass
class Job:
    job_id: str
    filename: str
    status: str = "构建中"          # 构建中 / 完成 / 失败
    message: str = "等待中..."
    cur: int = 0
    total: int = 0
    result: object = None
    error: str = ""
    deleted: bool = False
    created_at: float = field(default_factory=time.time)
    finished_at: float = 0.0


JOBS: dict[str, Job] = {}
_JOBS_LOCK = threading.Lock()

# job_ids whose build has been requested to stop (via deletion of an in-progress task)
CANCELLED_JOBS: set[str] = set()


def get_supervisor() -> Supervisor:
    global _supervisor
    if _supervisor is None:
        _supervisor = Supervisor()
    return _supervisor


def _start_async_build(job_id: str, source: str, text: str) -> None:
    """Run a build in the thread pool, storing progress/result in the JOBS registry.

    Shared by file and text builds so both behave identically (appear in the
    queue, can be cancelled / deleted, and surface in the per-task graph filter).
    """
    sup = get_supervisor()
    loop = asyncio.get_event_loop()

    def progress(msg: str) -> None:
        m = re.search(r"\((\d+)/(\d+)\)", msg)
        with _JOBS_LOCK:
            j = JOBS.get(job_id)
            if j:
                j.message = msg
                if m:
                    j.cur = int(m.group(1))
                    j.total = int(m.group(2))

    def should_cancel() -> bool:
        return job_id in CANCELLED_JOBS

    def run() -> None:
        try:
            result = sup.build(text, source=source, progress=progress, should_cancel=should_cancel)
            with _JOBS_LOCK:
                j = JOBS.get(job_id)
                if j:
                    if job_id in CANCELLED_JOBS:
                        j.status = "已取消"
                        j.message = "构建已取消"
                    else:
                        j.status = "完成"
                        j.result = result
                    j.finished_at = time.time()
        except BuildCancelled:
            with _JOBS_LOCK:
                j = JOBS.get(job_id)
                if j:
                    j.status = "已取消"
                    j.message = "构建已取消"
                    j.finished_at = time.time()
        except Exception as e:  # keep the job visible instead of silently dying
            import traceback
            traceback.print_exc()
            with _JOBS_LOCK:
                j = JOBS.get(job_id)
                if j:
                    j.status = "失败"
                    j.error = f"{type(e).__name__}: {e}"
                    j.finished_at = time.time()

    loop.run_in_executor(None, run)


class BuildRequest(BaseModel):
    text: str
    source: str = "document"


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5


class DeleteSourceRequest(BaseModel):
    source: str                       # 文件名（构建时用作 source 标识）
    job_id: str = ""                  # 可选：若是任务队列中的任务，一并取消


class ProgressResponse(BaseModel):
    message: str


@app.post("/api/build")
def build(req: BuildRequest):
    sup = get_supervisor()
    return sup.build(req.text, source=req.source or "document")


@app.post("/api/build_file")
async def build_file(file: UploadFile = File(...)):
    data = await file.read()
    text = _decode_file(data, file.filename)
    sup = get_supervisor()
    source = file.filename or "upload"
    return sup.build(text, source=source)


@app.get("/api/jobs")
def list_jobs():
    """Return every build job, newest first.

    The frontend renders its queue entirely from this endpoint, so a page
    refresh — which wipes in-browser JS state and the SSE connection — can
    recover the in-progress / finished tasks simply by polling again.
    """
    with _JOBS_LOCK:
        items = [asdict(j) for j in JOBS.values()]
    items.sort(key=lambda d: d["created_at"], reverse=True)
    return {"jobs": items}


@app.post("/api/build_file_job")
async def build_file_job(file: UploadFile = File(...)):
    """Start an asynchronous build for one file and return its job_id.

    The build runs in a thread pool; progress and the final result are stored
    server-side in the JOBS registry (keyed by job_id) and surfaced via
    GET /api/jobs, so the UI stays consistent across browser refreshes.
    """
    data = await file.read()
    text = _decode_file(data, file.filename)
    source = file.filename or "upload"
    job_id = uuid.uuid4().hex
    with _JOBS_LOCK:
        JOBS[job_id] = Job(job_id=job_id, filename=source)
    _start_async_build(job_id, source, text)
    return {"job_id": job_id, "filename": source}


@app.post("/api/build_text_job")
async def build_text_job(req: BuildRequest):
    """Start an asynchronous build for pasted text.

    The task is named ``输入文本_YYYYMMDD_HHMMSS`` so it behaves like an
    imported-file task: it shows up in the queue and the per-task graph filter,
    and can be cancelled / deleted just like a file import.
    """
    source = "输入文本_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    job_id = uuid.uuid4().hex
    with _JOBS_LOCK:
        JOBS[job_id] = Job(job_id=job_id, filename=source)
    _start_async_build(job_id, source, req.text)
    return {"job_id": job_id, "filename": source}


@app.post("/api/delete_source")
def delete_source(req: DeleteSourceRequest):
    """Delete all ingested data (graph entities/relations/events + vector chunks)
    tagged with ``source`` (normally a filename).

    If ``job_id`` refers to an in-progress build, that build is also cancelled so
    no partial result is written after the deletion.
    """
    if req.job_id:
        with _JOBS_LOCK:
            j = JOBS.get(req.job_id)
            if j:
                if j.status == "构建中":
                    CANCELLED_JOBS.add(req.job_id)
                    j.status = "已取消"
                    j.message = "已取消并删除入库数据"
                else:
                    j.status = "已删除"
                j.deleted = True
                j.finished_at = j.finished_at or time.time()
    result = get_supervisor().delete_source(req.source)
    return {
        "ok": True,
        "source": req.source,
        "nodes": result["graph"]["nodes"],
        "edges": result["graph"]["edges"],
        "vectors_removed": result["vectors_removed"],
    }


@app.post("/api/jobs/clear")
def clear_jobs():
    with _JOBS_LOCK:
        JOBS.clear()
    CANCELLED_JOBS.clear()
    return {"ok": True}


@app.get("/api/graph")
def graph():
    return get_supervisor().graph_json()


@app.get("/api/stats")
def stats():
    return get_supervisor().stats()


@app.post("/api/query")
def query(req: QueryRequest):
    try:
        return get_supervisor().query(req.question)
    except Exception as e:  # 容错：LLM/向量服务异常时返回结构化错误，前端可友好展示
        return {
            "answer": f"问答失败：{e}",
            "error": str(e),
            "question": req.question,
            "intent": None,
            "entities": [],
            "graph_context": "",
            "contexts": [],
            "evidence": [],
            "citations": [],
        }


@app.post("/api/reset")
def reset():
    get_supervisor().reset()
    with _JOBS_LOCK:
        JOBS.clear()
    CANCELLED_JOBS.clear()
    return {"ok": True}


@app.get("/")
def index():
    if FRONTEND.exists():
        return FileResponse(str(FRONTEND))
    return {"message": "frontend/index.html 不存在"}


def _decode_file(data: bytes, filename: Optional[str]) -> str:
    suffix = (filename or "").lower().split(".")[-1]
    if suffix == "pdf":
        try:
            from pypdf import PdfReader
        except Exception:
            raise RuntimeError("未安装 pypdf，无法解析 PDF")
        reader = PdfReader(io.BytesIO(data))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    return decode_text_bytes(data)
