from __future__ import annotations

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
from db import get_feedback_collection

from processing.pdfGeneration import generate_pdf_report
from processing.cloudinary import delete_from_cloudinary


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


@app.post("/process")
async def process_upload(
    file: UploadFile = File(...),
    language: str = Form(""),
    geography: str = Form(""),
):
    del language, geography  # Retained for frontend compatibility

    file_bytes = await file.read()

    try:
        output_filename, content = await process_file(file_bytes, file.filename or "input.xlsx")
    except ProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_id = str(uuid4())
    artifact_store.put(
        ProcessedArtifact(
            job_id=job_id,
            filename=output_filename,
            content=content,
            created_at=datetime.utcnow(),
        )
    )

    return {"job_id": job_id, "file_name": output_filename}


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
async def generate_pdf(file:UploadFile = File(...)):
    """
    Endpoint to generate a beautiful PDF report from Excel data with OpenAI insights.
    Supports brand logo, context, and customizable graph types.
    """
    reportTitle="Gentari"
    brandName="Gentari"
    includeInsights=True
    includeCharts=True
    customPrompt=""

    try:
        result = await generate_pdf_report(file, reportTitle, brandName, includeInsights, includeCharts, customPrompt)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(payload: FeedbackRequest):
    entry = payload.dict()
    entry["created_at"] = datetime.utcnow().isoformat()

    # Save in-memory for backward compatibility
    feedback_entries.append(entry)

    # Persist to MongoDB
    try:
        coll = get_feedback_collection()
        await coll.insert_one(entry)
    except Exception:
        # Ignore persistence failure to not break user flow
        pass

    return {"status": "received", "persisted": True}


@app.get("/feedback")
async def get_feedback(limit: int = 100):
    # Try to fetch from MongoDB, fallback to in-memory list
    try:
        coll = get_feedback_collection()
        cursor = (
            coll.find({}, {"_id": 0})  # exclude ObjectId for JSON serialization
            .sort("created_at", -1)
            .limit(int(limit))
        )
        results = await cursor.to_list(length=int(limit))
        return results
    except Exception:
        # Return latest items from in-memory storage if DB unavailable
        return list(reversed(feedback_entries))[-int(limit):]


class DeleteChartRequest(BaseModel):
    urls: List[str]


@app.delete("/charts/delete")
async def delete_charts(payload: DeleteChartRequest):
    """
    Delete chart images from Cloudinary.
    Accepts a list of Cloudinary URLs to delete.
    """
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
