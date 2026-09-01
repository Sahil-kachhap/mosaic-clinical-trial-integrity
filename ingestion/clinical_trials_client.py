import asyncio
import requests
from typing import Any
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from config.settings import settings
from config.logging_config import setup_logging
logger = setup_logging(__name__)

BASE_URL = settings.clinical_trials_base_url
PAGE_SIZE = settings.clinical_trials_page_size
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3

HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


class ClinicalTrialsClient:
    def __init__(self):
        self._session: requests.Session | None = None

    async def __aenter__(self) -> "ClinicalTrialsClient":
        self._session = requests.Session
        self._session.headers.update(HEADERS)
        logger.info("ClinicalTrials client opened")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            self._session.close()
            logger.info("ClinicalTrials client closed")