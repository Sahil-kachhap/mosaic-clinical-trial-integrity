import asyncpg
from datetime import datetime
from typing import Any
from config.settings import settings
from config.logging_config import setup_logging
logger = setup_logging(__name__)


class SemanticStore:
    def __init__(self):
        self._pool: asyncpg.Pool | None = None
        logger.info("Semantic Store Initialized")

    async def _ensure_pool(self) -> None:
        if self._pool is not None:
            return

        self._pool = await asyncpg.create_pool(
            host=settings.db_host,
            port=settings.db_port,
            database=settings.db_name,
            user=settings.db_user,
            password=settings.db_password,
            min_size=1,
            max_size=5
        )

        logger.info("Semantic Store pool created")

    async def get_sponsor_profile(self, sponsor: str) -> dict[str, Any] | None:
        await self._ensure_pool()

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT sponsor, credibility_score, total_studies, results_posted, results_missing, broken_promises, avg_delay_days, last_updated
                FROM sponsor_profiles WHERE sponsor = $1
                """,
                sponsor
            )
        
        if row is None:
            logger.info(f"No profile found for sponsor | sponsor = {sponsor}")
            return None
        
        return {
            "sponsor":row["sponsor"],
            "credebility_score":float(row["credebility_score"] or 0.0),
            "total_studies": int(row["total_studies"] or 0),
            "results_posted":int(row["results_posted"] or 0),
            "results_missing":int(row["results_missing"] or 0),
            "broken_promises":int(row["broken_promises"] or 0),
            "avg_delay_days":float(row["avg_delay_days"] or 0.0),
            "last_updated": str(row["last_updated"]),
        }
    
    async def update_sponsor_knowledge(self, sponsor:str, results_posted:bool = False, had_broken_promise:bool = False, delay_days:int = 0) -> None:
        await self._ensure_pool()

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO sponsor_profiles (
                    sponsor,
                    credibility_score,
                    total_studies,
                    results_posted,
                    results_missing,
                    broken_promises,
                    avg_delay_days,
                    last_updated
                )
                VALUES ($1, 0.5, 1, $2, $3, $4, $5, NOW())
                ON CONFLICT (sponsor) DO UPDATE SET
                    total_studies   = sponsor_profiles.total_studies + 1,
                    results_posted  = sponsor_profiles.results_posted + $2,
                    results_missing = sponsor_profiles.results_missing + $3,
                    broken_promises = sponsor_profiles.broken_promises + $4,
                    avg_delay_days  = (
                        (sponsor_profiles.avg_delay_days *
                         sponsor_profiles.total_studies) + $5
                    ) / (sponsor_profiles.total_studies + 1),
                    last_updated    = NOW()
                """,
                sponsor,
                int(results_posted),
                int(not results_posted),
                int(had_broken_promise),
                float(delay_days),
            )

            await conn.execute(
                """
                UPDATE sponsor_profiles
                SET credibility_score = GREATEST(0.0, LEAST(1.0,
                    (
                        CASE
                            WHEN total_studies = 0 THEN 0.5
                            ELSE (results_posted::float / total_studies) * 0.7
                        END
                    ) - (broken_promises * 0.1)
                ))
                WHERE sponsor = $1
                """,
                sponsor
            )

        logger.info(f"Sponsor Knowledge Updated | sponsor = {sponsor} | results_posted = {results_posted} | broken_promise = {had_broken_promise} | delay_days = {delay_days}")

    
    async def get_low_credibility_sponsor(self, threshold:float=0.6, min_studies:int = 3) -> list[dict]:
        await self._ensure_pool()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT sponsor, credibility_score, total_studies, results_posted, results_missing, broken_promises, avg_delay_days, last_updated
                FROM sponsor_profiles WHERE credibility_score < $1 AND total_studies >= $2 
                ORDER BY credibility_score ASC
                """,
                threshold,
                min_studies,
            )

        sponsors = [
            {
                "sponsor": row["sponsor"],
                "credibility_score": float(row["credibility_score"] or 0.0),
                "total_studies": int(row["total_studies"] or 0),
                "results_posted": int(row["results_posted"] or 0),
                "results_missing": int(row["results_missing"] or 0),
                "broken_promises": int(row["broken_promises"] or 0),
                "avg_delay_days": float(row["avg_delay_days"] or 0.0),
                "last_updated": str(row["last_updated"]),
            }
            for row in rows
        ]

        logger.info(f"Low credibility sponsor found | count = {len(sponsors)} | threshold = {threshold} | min_studies = {min_studies}")

        return sponsors
    
    async def sponsor_exists(self, sponsor: str) -> bool:
        await self._ensure_pool()

        async with self._pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM sponsor_profiles WHERE sponsor = $1",
                sponsor,
            )
        
        return (count or 0) > 0
    
    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Semantic Store Pool Closed")