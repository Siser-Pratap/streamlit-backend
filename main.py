from __future__ import annotations

import asyncio
from datetime import datetime
from io import BytesIO
from typing import List
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from processing.gentari import ProcessingError, process_file
from storage import ProcessedArtifact, artifact_store
from db import create_job, update_job_status, get_job, get_feedback_collection
from processing.pdfGeneration import generate_pdf_report
from processing.cloudinary import delete_from_cloudinary
from concurrent.futures import ThreadPoolExecutor


app = FastAPI(title="Rantau Data Processing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FeedbackRequest(BaseModel):
    name: str | None = ""
    email: str | None = ""
    feedback_type: str | None = ""
    feedback: str


feedback_entries: List[FeedbackRequest] = []


# ------------------------------
# Long-running processing logic
# ------------------------------
executor = ThreadPoolExecutor(max_workers=3)

async def long_processing_job(job_id: str, file_bytes: bytes, filename: str):
    try:
        loop = asyncio.get_running_loop()

        # Run CPU-heavy task in a thread
        output_filename, content = await loop.run_in_executor(
            executor, lambda: asyncio.run(process_file(file_bytes, filename))
        )

        artifact_store.put(
            ProcessedArtifact(
                job_id=job_id,
                filename=output_filename,
                content=content,
                created_at=datetime.utcnow(),
            )
        )

        await update_job_status(job_id, "completed")

    except Exception as e:
        await update_job_status(job_id, "failed", str(e))


# ------------------------------
# Routes
# ------------------------------

@app.post("/process")
async def process_upload(
    file: UploadFile = File(...),
    language: str = Form(""),
    geography: str = Form(""),
):
    """Starts a long-running async processing job."""
    file_bytes = await file.read()
    job_id = str(uuid4())

    await create_job(job_id, file.filename)

    # Run in background (non-blocking)
    asyncio.create_task(long_processing_job(job_id, file_bytes, file.filename))

    return {"job_id": job_id, "message": "Processing started"}


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Check job status from MongoDB."""
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/result/{job_id}")
async def get_result(job_id: str):
    """Retrieve the processed file after completion."""
    job = await get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] == "error":
        raise HTTPException(status_code=502, detail=job.error)

    elif job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not yet completed")

    artifact = artifact_store.get(job_id)
    if not artifact:
        raise HTTPException(status_code=404, detail="Artifact not found")

    return {
        "filename": artifact.filename,
        "content": artifact.content.decode(errors="ignore"),
    }


@app.get("/download/{job_id}")
async def download_output(job_id: str, format: str = "excel"):
    if format != "excel":
        raise HTTPException(status_code=400, detail="Only Excel downloads are supported.")

    artifact = artifact_store.get(job_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Job not found.")

    buffer = BytesIO(artifact.content)
    headers = {"Content-Disposition": f'attachment; filename="{artifact.filename}"'}
    return StreamingResponse(
        buffer,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers,
    )


@app.post("/pdf-report/generate")
async def generate_pdf(file: UploadFile = File(...)):
    """
    Endpoint to generate a PDF report from Excel data with insights and charts.
    """
    reportTitle = "Gentari"
    brandName = "Gentari"
    includeInsights = True
    includeCharts = True
    customPrompt = ""

    try:
        result = await generate_pdf_report(file, reportTitle, brandName, includeInsights, includeCharts, customPrompt)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/feedback")
async def submit_feedback(payload: FeedbackRequest):
    entry = payload.dict()
    entry["created_at"] = datetime.utcnow().isoformat()

    feedback_entries.append(entry)

    try:
        coll = get_feedback_collection()
        await coll.insert_one(entry)
    except Exception:
        pass

    return {"status": "received", "persisted": True}


@app.get("/feedback")
async def get_feedback(limit: int = 100):
    try:
        coll = get_feedback_collection()
        cursor = (
            coll.find({}, {"_id": 0})
            .sort("created_at", -1)
            .limit(int(limit))
        )
        results = await cursor.to_list(length=int(limit))
        return results
    except Exception:
        return list(reversed(feedback_entries))[-int(limit):]


class DeleteChartRequest(BaseModel):
    urls: List[str]


@app.delete("/charts/delete")
async def delete_charts(payload: DeleteChartRequest):
    """Delete chart images from Cloudinary."""
    deleted = []
    failed = []

    for url in payload.urls:
        try:
            result = delete_from_cloudinary(url)
            if result.get("result") == "ok":
                deleted.append(url)
            else:
                failed.append({"url": url, "error": result.get("result", "unknown error")})
        except Exception as e:
            failed.append({"url": url, "error": str(e)})

    return {
        "success": len(failed) == 0,
        "deleted": deleted,
        "failed": failed,
        "total_requested": len(payload.urls),
        "total_deleted": len(deleted),
        "total_failed": len(failed)
    }
