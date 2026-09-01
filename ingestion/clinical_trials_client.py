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

    async def search_studies(self, condition: str | None = None,
                             intervention: str | None = None, 
                             sponsor: str | None = None, 
                             status: list[str] | None = None,
                             max_results=100) -> list[dict[str, Any]]:
        all_studies: list[dict[str, Any]] = []
        next_page_token: str | None = None
        page_number = 0

        logger.info(
            f"Searching studies |"
            f"Condition = {condition}"
            f"Intervention = {intervention}"
            f"Sponsor = {sponsor}"
            f"Max Results = {max_results}"
        )

        while len(all_studies) < max_results:
            page_number += 1
            params = self._build_search_params(
                condition=condition,
                intervention=intervention,
                sponsor=sponsor,
                status=status,
                page_token=next_page_token
            )

            response_data = await self._fetch_page(params=params)
            if not response_data:
                break

            page_studies = response_data.get("studies", [])
            if not page_studies:
                logger.info("No more studies available - pagination complete")
                break
            all_studies.extend(page_studies)
            logger.info(
                f"Page: {page_number}"
                f"Fetched = {len(page_studies)}"
                f"Total Fetched so far = {len(all_studies)}"
            )

            next_page_token = response_data.get("nextPageToken")
            if not next_page_token:
                logger.info("Last page reacher - no nextpagetoken in response")
                break

        all_studies = all_studies[:max_results]
        logger.info(
            f"Search Complete |"
            f"Total Studies returned = {len(all_studies)}"
        )

        return all_studies
