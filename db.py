from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

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


