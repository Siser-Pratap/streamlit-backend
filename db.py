from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection


@lru_cache(maxsize=1)
def get_mongo_client() -> AsyncIOMotorClient:
    """
    Returns a cached AsyncIOMotorClient using the MONGODB_URI environment variable.
    """
    uri: Optional[str] = os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI is not set. Please provide your MongoDB connection string.")
    return AsyncIOMotorClient(uri)


def get_feedback_collection() -> AsyncIOMotorCollection:
    """
    Returns the feedback collection handle.
    Database name defaults to 'rantau', override via MONGODB_DB.
    Collection name is 'feedback'.
    """
    db_name = os.getenv("MONGODB_DB", "rantau")
    client = get_mongo_client()
    return client[db_name]["feedback"]

def get_jobs_collection() -> AsyncIOMotorCollection:
    """
    Returns the jobs collection handle.
    """
    db_name = os.getenv("MONGODB_DB", "rantau")
    client = get_mongo_client()
    return client[db_name]["jobs"]

async def create_job(job_id: str, filename: str):
    jobs_collection = get_jobs_collection()
    job = {
        "_id": job_id,
        "filename": filename,
        "status": "processing",
        "created_at": datetime.utcnow(),
        "completed_at": None,
        "error": None
    }
    await jobs_collection.insert_one(job)


async def update_job_status(job_id: str, status: str, error: str | None = None):
    jobs_collection = get_jobs_collection()
    update_data = {"status": status, "completed_at": datetime.utcnow()}
    if error:
        update_data["error"] = error
    await jobs_collection.update_one({"_id": job_id}, {"$set": update_data})


async def get_job(job_id: str):
    jobs_collection = get_jobs_collection()
    job = await jobs_collection.find_one({"_id": job_id}, {"_id": 0})
    return job

