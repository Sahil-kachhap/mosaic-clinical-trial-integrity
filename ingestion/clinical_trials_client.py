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
        self._session = requests.Session()
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
            f"Searching studies | "
            f"Condition = {condition} | "
            f"Intervention = {intervention} | "
            f"Sponsor = {sponsor} | "
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

    async def fetch_study(self, nct_id: str) -> dict[str, Any] | None:
        logger.info(f"Fetching study, nct_id = {nct_id}")

        def _get_study():
            return self._session.get(
                f"{BASE_URL}/studies/{nct_id}",
                timeout=REQUEST_TIMEOUT
            )

        try:
            response = asyncio.to_thread(_get_study)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            logger.warning(
                f"Study not found |"
                f"nct_id = {nct_id}"
                f"status = {e.response.status_code}"
            )
            return None
        except Exception as e:
            logger.error(
                f"Failed to fetch study |"
                f"nct_id = {nct_id}"
                f"error = {e}"
            )

    def _build_search_params(
        self,
        condition: str | None,
        intervention: str | None,
        sponsor: str | None,
        status: list[str] | None,
        page_token: str | None,
    ) -> dict[str, Any]:

        params: dict[str, Any] = {
            "pageSize": PAGE_SIZE,
            "format": "json"
        }

        if condition:
            params["query.cond"] = condition
        if intervention:
            params["query.intr"] = intervention
        if sponsor:
            params["query.spons"] = sponsor
        if status:
            params["filter.overallStatus"] = "|".join(status)
        if page_token:
            params["pageToken"] = page_token
        return params

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(
            (requests.exceptions.Timeout, requests.exceptions.ConnectionError)
        )
    )
    async def _fetch_page(self, params: dict[str, Any]) -> dict[str, Any] | None:
        def _get():
            return self._session.get(
                f"{BASE_URL}/studies",
                params=params,
                timeout=REQUEST_TIMEOUT
            )

        try:
            response = await asyncio.to_thread(_get)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            logger.warning(f"request timed out after {REQUEST_TIMEOUT}s - retrying...")
            raise
        except requests.exceptions.ConnectionError:
            logger.warning(f"Connection error - retrying...")
            raise
        except requests.exceptions.HTTPError as e:
            logger.error(f"Http Error from API | status = {e.response.status_code} | url = {e.response.url}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error occured while fetching page | error = {e}")
            return None
