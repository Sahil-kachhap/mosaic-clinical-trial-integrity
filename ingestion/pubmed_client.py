import asyncio
import httpx
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

BASE_URL = settings.pubmed_base_url
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3
FETCH_BATCH_SIZE = 20
RATE_LIMIT_SLEEP = 0.4


class PubMedClient:
    def __init__(self):
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "PubMedClient":
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(REQUEST_TIMEOUT))
        headers = {
            "Accept": "application/json",
        }
        logger.info("Pubmed client opened")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()
            logger.info("Pubmed client closed")
