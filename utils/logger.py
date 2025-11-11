import logging
from datetime import date
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response, JSONResponse

# Formatted Data
formatted_date = date.today().strftime("%Y-%m-%d")

# Create logger
logger = logging.getLogger("api_logger")
logger.setLevel(logging.DEBUG)  # Capture all levels

# Formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Create file handlers
file_handler = logging.FileHandler(f"api-{formatted_date}.log")
file_handler.setLevel(logging.DEBUG)  # Want specific log change with INFO, ERROR, etc
file_handler.setFormatter(formatter)

# Console handler (for all levels)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(formatter)

# Avoid adding handlers multiple times if reloaded
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


# Request/Response Logging Middleware
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        try:
            # Read raw request body
            request_body = await request.body()
            content_type = request.headers.get("content-type", "")

            # Detect and handle different content types
            if (
                "multipart/form-data" in content_type
                or "application/octet-stream" in content_type
            ):
                body_str = "<multipart/form-data or binary content omitted>"
            else:
                try:
                    body_str = request_body.decode("utf-8")
                except UnicodeDecodeError:
                    body_str = "<non-UTF8 data omitted>"

            logger.info(
                f"➡️ Request: {request.method} {request.url.path} | Content-Type: {content_type} | Body: {body_str}"
            )

            # Process request
            response: Response = await call_next(request)

            process_time = (time.time() - start_time) * 1000
            logger.info(
                f"⬅️ Response: {request.method} {request.url.path} | Status: {response.status_code} | Time: {process_time:.2f}ms"
            )

            return response

        except Exception as e:
            logger.error(
                f"❌ Error in {request.method} {request.url.path}: {str(e)}",
                exc_info=True,
            )

            return JSONResponse(
                status_code=500,
                content={"detail": "Internal Server Error"},
            )