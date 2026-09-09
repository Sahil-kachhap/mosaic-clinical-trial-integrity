import json
import asyncpg
from typing import Any
from processing.embedder import EmbeddedChunk
from config.settings import settings
from config.logging_config import setup_logging

logger = setup_logging(__name__)

POOL_MIN_SIZE = 2
POOL_MAX_SIZE = 10
TOP_K_DEFAULT = 5


class VectorStore:
    def __init__(self):
        self._loop: asyncpg.Pool | None = None

    async def __aenter__(self) -> "VectorStore":
        await self.init()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def init(self) -> None:
        logger.info(
            f"Connecting to Cloud SQL | host = {settings.db_host} | database = {settings.db_name} | {settings.db_user} | {settings.db_password}")
        self._pool = await asyncpg.create_pool(host=settings.db_host, port=int(settings.db_port), database=settings.db_name, user=settings.db_user, password=settings.db_password, min_size=POOL_MIN_SIZE, max_size=POOL_MAX_SIZE, init=self._init_connection)
        logger.info("Connection Pool created successfully")

    async def _init_connection(self, conn: asyncpg.Connection) -> None:
        await conn.set_type_codec(
            "vector",
            encoder=lambda v: json.dumps(v),
            decoder=lambda v: json.loads(v),
            schema="public",
            format="text"
        )

    async def close(self) -> None:
        if self._loop:
            await self._pool.close()
            logger.info("Connection Pool Closed")

    async def save_embedded_chunks(self, chunks: list[EmbeddedChunk]) -> int:
        if not chunks:
            logger.warning("save_embedded_chunks called with empty list")
            return 0

        saved_count = 0

        async with self._pool.acquire() as conn:
            for chunk in chunks:
                try:
                    await conn.execute(
                        """
                            INSERT INTO chunks (nct_id, chunk_text, embedding, chunk_index, source)
                            VALUES ($1, $2, $3, $4, $5) ON CONFLICT DO NOTHING
                        """,
                        chunk.nct_id,
                        chunk.chunk_text,
                        chunk.embedding,
                        chunk.chunk_index,
                        chunk.source
                    )

                    saved_count += 1
                except Exception as e:
                    logger.error(
                        f"Failed to save chunk | chunk_id={chunk.chunk_id} | error = {e}")

        logger.info(
            f"Chunks Saved | saved = {saved_count} | total_input = {len(chunks)} | skipped = {len(chunks) - saved_count}")
        return saved_count

    async def save_study(
        self,
        study_data: dict[str, Any],
    ) -> None:

        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO studies
                    (nct_id, title, sponsor, phase, status,
                     conditions, interventions, primary_outcome,
                     secondary_outcomes, start_date, completion_date,
                     results_posted, enrollment, gcs_path)
                VALUES
                    ($1, $2, $3, $4, $5,
                     $6, $7, $8,
                     $9, $10, $11,
                     $12, $13, $14)
                ON CONFLICT (nct_id) DO UPDATE SET
                    title            = EXCLUDED.title,
                    sponsor          = EXCLUDED.sponsor,
                    phase            = EXCLUDED.phase,
                    status           = EXCLUDED.status,
                    conditions       = EXCLUDED.conditions,
                    interventions    = EXCLUDED.interventions,
                    primary_outcome  = EXCLUDED.primary_outcome,
                    secondary_outcomes = EXCLUDED.secondary_outcomes,
                    start_date       = EXCLUDED.start_date,
                    completion_date  = EXCLUDED.completion_date,
                    results_posted   = EXCLUDED.results_posted,
                    enrollment       = EXCLUDED.enrollment,
                    gcs_path         = EXCLUDED.gcs_path
                """,


                study_data.get("nct_id"),              # $1
                study_data.get("title"),               # $2
                study_data.get("sponsor"),             # $3
                study_data.get("phase"),               # $4
                study_data.get("status"),              # $5
                study_data.get("conditions", []),      # $6
                study_data.get("interventions", []),   # $7
                study_data.get("primary_outcome"),     # $8
                study_data.get("secondary_outcomes", []),  # $9
                study_data.get("start_date"),          # $10
                study_data.get("completion_date"),     # $11
                study_data.get("results_posted"),      # $12
                study_data.get("enrollment"),          # $13
                study_data.get("gcs_path"),            # $14
            )

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = TOP_K_DEFAULT,
        source_filter: str | None = None,
        nct_id_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        conditions = []

        params: list[Any] = [query_embedding]

        param_count = 1

        if source_filter:
            param_count += 1
            conditions.append(f"source = ${param_count}")
            params.append(source_filter)

        if nct_id_filter:
            param_count += 1
            conditions.append(f"nct_id = ${param_count}")
            params.append(nct_id_filter)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        param_count += 1
        params.append(top_k)

        query = f"""
            SELECT
                nct_id,
                chunk_text,
                chunk_index,
                source,
                embedding <=> $1 AS distance
            FROM chunks
            {where_clause}
            ORDER BY distance ASC
            LIMIT ${param_count}
        """

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, *params)

        results = [dict(row) for row in rows]

        logger.info(
            f"Semantic search complete | "
            f"results_found={len(results)} | "
            f"top_k={top_k} | "
            f"source_filter={source_filter} | "
            f"nct_id_filter={nct_id_filter}"
        )

        return results

    async def get_chunk_count(self) -> int:
        async with self._pool.acquire() as conn:
            result = await conn.fetchval("SELECT COUNT(*) FROM chunks")

        logger.info(f"Total chunks in database: {result}")
        return result

    async def study_exists(self, nct_id: str) -> bool:
        async with self._pool.acquire() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM chunks WHERE nct_id = $1",
                nct_id,
            )

        exists = count > 0
        return exists

    async def get_chunks_for_study(self, nct_id: str) -> str:
        async with self._pool.acquire() as conn:
            result = await conn.fetch("SELECT chunk_text FROM chunks WHERE nct_id = $1", nct_id)
        
        if result is None:
            return []
        
        return result