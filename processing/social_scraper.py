"""
Helper function to submit scraping jobs to BrightData (Datasets v3 scrape),
poll for results (if job-based), and fetch/parse results for social platforms.

"""

import os
import time
import json
import math
import logging
from typing import List, Dict, Any, Optional, Iterable

import requests

# Basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment-configured API key (do NOT hardcode real keys)
BRIGHTDATA_API_KEY = os.getenv("BRIGHTDATA_API_KEY")
if not BRIGHTDATA_API_KEY:
    logger.warning("BRIGHTDATA_API_KEY not set. Set env var before running scraping.")

# Default BrightData dataset endpoint for v3 scraping (adjust dataset_id when calling)
BRIGHTDATA_BASE = "https://api.brightdata.com"
BRIGHTDATA_SCRAPE_ENDPOINT = "/datasets/v3/scrape"

# Default headers builder
def _headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

# Utility: exponential backoff sleep
def _backoff_sleep(attempt: int, base: float = 0.5, cap: float = 30.0) -> None:
    sleep = min(cap, base * (2 ** attempt))
    jitter = sleep * 0.1 * (0.5 - (time.time() % 1))  # tiny jitter
    time.sleep(max(0, sleep + jitter))

# Submit a scrape job (single HTTP call). Returns raw response JSON or raises.
def submit_scrape_job(
    urls: Iterable[str],
    dataset_id: str,
    notify: bool = False,
    include_errors: bool = True,
    api_key: Optional[str] = None,
    timeout: int = 60,
    extra_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Submit a scraping request to BrightData Datasets v3.

    Parameters:
    - urls: iterable of URLs to scrape
    - dataset_id: your BrightData dataset id (e.g. 'gd_xxx...')
    - notify: whether BrightData should notify (depends on dataset)
    - include_errors: include errors in results
    - extra_payload: any additional keys you want to pass in the request body

    Returns:
    - BrightData response JSON (which may contain job id or immediate results)
    """
    api_key = api_key or BRIGHTDATA_API_KEY
    if not api_key:
        raise RuntimeError("BrightData API key not provided. Set BRIGHTDATA_API_KEY env var or pass api_key.")

    url = f"{BRIGHTDATA_BASE}{BRIGHTDATA_SCRAPE_ENDPOINT}?dataset_id={dataset_id}&notify={'true' if notify else 'false'}&include_errors={'true' if include_errors else 'false'}"
    payload = {"input": [{"url": u} for u in urls]}
    if extra_payload:
        payload.update(extra_payload)

    logger.info("Submitting scrape request to BrightData for %d url(s) ...", len(list(urls)))
    resp = requests.post(url, headers=_headers(api_key), json=payload, timeout=timeout)
    try:
        resp.raise_for_status()
    except Exception:
        logger.error("BrightData submit failed: %s %s", resp.status_code, resp.text)
        raise

    return resp.json()

# Poll job status endpoint (if dataset returns jobId). Not all dataset setups need polling.
def poll_job_status(
    job_id: str,
    dataset_id: str,
    api_key: Optional[str] = None,
    interval: float = 2.0,
    timeout: float = 120.0
) -> Dict[str, Any]:
    """
    Poll job status until completion or timeout.
    Returns the final job status JSON.
    """
    api_key = api_key or BRIGHTDATA_API_KEY
    if not api_key:
        raise RuntimeError("BrightData API key required.")

    status_url = f"{BRIGHTDATA_BASE}/datasets/v3/jobs/{job_id}?dataset_id={dataset_id}"
    start = time.time()
    attempt = 0
    while True:
        resp = requests.get(status_url, headers=_headers(api_key))
        try:
            resp.raise_for_status()
        except Exception:
            logger.error("Job status fetch failed: %s %s", resp.status_code, resp.text)
            raise

        job_json = resp.json()
        state = job_json.get("state") or job_json.get("status")
        logger.debug("Job %s status: %s", job_id, state)
        if state in ("done", "completed", "finished", "success"):
            return job_json
        if state in ("failed", "error"):
            raise RuntimeError(f"BrightData job failed: {job_json}")

        if time.time() - start > timeout:
            raise TimeoutError(f"Job {job_id} did not finish within {timeout} seconds")

        attempt += 1
        _backoff_sleep(attempt, base=interval)

# Simple function that attempts to fetch results from a "results" URL returned by dataset
def fetch_results_from_url(results_url: str, api_key: Optional[str] = None, timeout: int = 60) -> Dict[str, Any]:
    api_key = api_key or BRIGHTDATA_API_KEY
    if not api_key:
        raise RuntimeError("BrightData API key required.")
    resp = requests.get(results_url, headers=_headers(api_key), timeout=timeout)
    resp.raise_for_status()
    return resp.json()

# Helper: high-level scrape-and-get results function
def scrape_and_get_results(
    urls: Iterable[str],
    dataset_id: str,
    api_key: Optional[str] = None,
    wait_for_job: bool = True,
    poll_timeout: float = 120.0,
) -> Dict[str, Any]:
    """
    Submit scrape job and return results. If BrightData returns job info (jobId),
    poll until done and then fetch results. Otherwise, returns immediate response.
    """
    api_key = api_key or BRIGHTDATA_API_KEY
    resp_json = submit_scrape_job(urls, dataset_id, api_key=api_key)

    # The exact keys depend on dataset settings. Handle common shapes:
    #  - immediate 'results' in response
    #  - 'jobId' returned (then poll job status)
    #  - 'data' or 'items' key
    if "results" in resp_json:
        return resp_json["results"]
    if "data" in resp_json:
        return resp_json["data"]
    if "jobId" in resp_json and wait_for_job:
        job_id = resp_json["jobId"]
        job_status = poll_job_status(job_id, dataset_id, api_key=api_key, timeout=poll_timeout)
        # When job completes, the job_status may include a results URL or data
        if "results" in job_status:
            return job_status["results"]
        if "data" in job_status:
            return job_status["data"]
        # Alternatively, job may include a results URL
        if "results_url" in job_status:
            return fetch_results_from_url(job_status["results_url"], api_key=api_key)
        # Fallback: return whole job JSON
        return job_status

    # Fallback: return raw response
    return resp_json

# Platform specific parsing hooks (simple examples)
def parse_facebook_item(item: Dict[str, Any]) -> Dict[str, Any]:
    # item is BrightData-provided JSON per scraped page; fields vary by dataset.
    # Implement your extraction according to the structure BrightData returns.
    # Example pseudo-handling:
    content = item.get("content") or item.get("html") or item.get("body") or ""
    title = item.get("title") or ""
    return {"platform": "facebook", "url": item.get("url"), "title": title, "raw_content": content}

def parse_instagram_item(item: Dict[str, Any]) -> Dict[str, Any]:
    content = item.get("content") or item.get("html") or item.get("body") or ""
    # Example: try to extract caption and images (depends on dataset)
    caption = item.get("caption") or item.get("meta", {}).get("og:description") or ""
    images = item.get("images") or item.get("media") or []
    return {"platform": "instagram", "url": item.get("url"), "caption": caption, "images": images, "raw_content": content}

def parse_x_item(item: Dict[str, Any]) -> Dict[str, Any]:
    content = item.get("content") or item.get("html") or item.get("body") or ""
    tweet_text = item.get("text") or item.get("tweet") or ""
    return {"platform": "x", "url": item.get("url"), "text": tweet_text, "raw_content": content}

# Generic parser dispatcher
def parse_brightdata_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    parsed = []
    for item in results:
        url = item.get("url", "")
        if "facebook.com" in url:
            parsed.append(parse_facebook_item(item))
        elif "instagram.com" in url or "instagr.am" in url:
            parsed.append(parse_instagram_item(item))
        elif "twitter.com" in url or "x.com" in url:
            parsed.append(parse_x_item(item))
        else:
            parsed.append({"platform": "unknown", "url": url, "raw": item})
    return parsed