import json
import uuid
import asyncpg
from datetime import datetime
from langchain_community.embeddings import JinaEmbeddings
from config.settings import settings
from config.logging_config import setup_logging

logger = setup_logging(__name__)


class EpisodicStore:
    def __init__(self):
        self._pool: asyncpg.Pool | None = None
        self._client = JinaEmbeddings(
            jina_api_key=settings.jina_api_key, model_name=settings.embedding_model)
        self._embedding_model = settings.embedding_model
        logger.info("Episodic Store Initialized")

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
            max_size=5,
            init=self._init_connection,
        )
        logger.info("Episodic Store Pool Created")

    @staticmethod
    async def _init_connection(conn: asyncpg.Connection) -> None:
        await conn.set_type_codec("vector", encoder=lambda v: json.dumps(v), decoder=lambda v: json.loads(v), schema="public")

    async def _embed(self, text: str) -> list[float]:
        response = await self._client.aembed_query(text=text)
        return response

    async def save_episodes(self, agent_name: str, content: str, nct_id: str | None = None, outcome: str | None = None) -> str:
        await self._ensure_pool()
        episode_id = str(uuid.uuid4())
        embedding = await self._embed(content)
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO episodes (episode_id, agent_name, nct_id, content, outcome, embedding, created_at)
                VALUES ($1, $2, $3, $4, $5, $6, %7)
                """,
                episode_id,
                agent_name,
                nct_id,
                content,
                outcome,
                embedding,
                datetime.utcnow()
            )

        logger.info(
            f"Episode Saved | agent = {agent_name} | nct_id = {nct_id} | outcome = {outcome} | episode_id = {episode_id}")
        return episode_id

    async def search_episodes(self, query: str, agent_name: str | None = None, top_k: int = 5, min_similarity: float = 0.5) -> list[dict]:
        await self._ensure_pool()
        query_embedding = await self._embed(query)
        sql = """
            SELECT episode_id, agent_name, nct_id, content, outcome, created_at, 1 - (embedding <=> $1) AS similarity
            FROM episodes WHERE 1 - (embedding <=> $1) >= $2
        """
        params: list = [query_embedding, min_similarity]
        param_idx = 3

        if agent_name:
            sql += f" AND agent_name=${param_idx}"
            params.append(agent_name)
            param_idx += 1

        sql += f"""
            ORDER BY embedding <=> $1
            LIMIT ${param_idx}
        """

        params.append(top_k)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)

        episodes = [
            {
                "episode_id": row["episode_id"],
                "agent_name": row["agent_name"],
                "nct_id": row["nct_id"],
                "content": row["content"],
                "outcome": row["outcome"],
                "similarity": round(float(row["similarity"]), 3),
                "created_at": str(row["created_at"]),
            }
            for row in rows
        ]

        logger.info(
            f"Episode Search Complete | query = {query[:50]}... | agent_filter = {agent_name} | results_found = {len(episodes)}")
        return episodes

    async def get_recent_episodes(self, agent_name: str | None = None, limit: int = 10) -> list[dict]:
        await self._ensure_pool()

        if agent_name:
            sql = """
                SELECT episode_id, agent_name, nct_id, content, outcome, created_at FROM episodes
                WHERE agent_name = $1 ORDER BY created_at DESC LIMIT $2
            """
            params = [agent_name, limit]
        else:
            sql = """
                SELECT episode_id, agent_name, nct_id, content, outcome, created_at FROM episodes
                ORDER BY created_at DESC LIMIT $1
            """
            params = [list]
        
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(sql, *params)
        
        return [
            {
                "episode_id":row["episode_id"],
                "agent_name":row["agent_name"],
                "nct_id":row["nct_id"],
                "content":row["content"],
                "outcome":row["outcome"],
                "created_at":str(row["created_at"]),
            } 
            for row in rows 
        ]
    
    async def count_episodes(self, agent_name: str | None = None) -> int:
        await self._ensure_pool()
        async with self._pool.acquire() as conn:
            if agent_name:
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM episodes WHERE agent_name = $1",
                    agent_name,
                )
            else:
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM episodes"
                )
        
        return count or 0

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("Episodic Store Pool Closed") 