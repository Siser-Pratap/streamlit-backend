"""
Configuration settings for the Streamlit app
"""

# API Configuration
API_BASE_URL = "http://localhost:8001"
API_TIMEOUT = 100000

# Upload Configuration
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_FILE_TYPES = ["xlsx", "csv", "xls"]

# Processing Configuration
PROCESSING_TIMEOUT = 10000  # 5 minutes

# Theme colors
DEFAULT_THEME = "dark"

API_KEY=""
