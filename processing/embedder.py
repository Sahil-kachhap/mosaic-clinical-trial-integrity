import asyncio
from dataclasses import dataclass
from processing.chunker import TextChunk
from langchain_community.embeddings import JinaEmbeddings
from config.settings import settings
from config.logging_config import setup_logging
logger = setup_logging(__name__)

BATCH_SIZE = 50
RETRY_ATTEMPTS = 3
RETRY_SLEEP_SECONDS = 2


@dataclass
class EmbeddedChunk:
    chunk_id: str
    nct_id: str
    chunk_text: str
    chunk_index: int
    source: str
    word_count: int
    embedding: list[float]


class Embedder:
    def __init__(self) -> None:
        self._client = JinaEmbeddings(
            jina_api_key=settings.jina_api_key, model_name=settings.embedding_model)
        self._model = settings.embedding_model
        logger.info(f"embedder initialized with model: {self._model}")

    async def embed_chunks(self, chunks: list[TextChunk]) -> list[EmbeddedChunk]:
        if not chunks:
            logger.warning(
                "No chunks provided to embed. Returning empty list.")
            return []
        logger.warning(
            f"Starting Embedding | Total Chunks: {len(chunks)} | Batch Size: {BATCH_SIZE} | model = {self._model}")
        all_embedded_chunks: list[EmbeddedChunk] = []
        batches = self._create_batches(chunks)

        for batch_num, batch in enumerate(batches):
            logger.info(
                f"Embedding batch {batch_num + 1}/{len(batches)} | chunks in batch={len(batch)}")

            embedded_batch = await self._embed_batch_with_retry(batch=batch, batch_num=batch_num)
            all_embedded_chunks.extend(embedded_batch)

            if batch_num < len(batches) - 1:
                await asyncio.sleep(0.5)

        logger.info(
            f"Embedding complete | total_embedded={len(all_embedded_chunks)} | total_input={len(chunks)} | skipped = {len(chunks) - len(all_embedded_chunks)}")
        return all_embedded_chunks

    def _create_batches(self, chunks: list[TextChunk]) -> list[list[TextChunk]]:
        return [
            chunks[i: i+BATCH_SIZE] for i in range(0, len(chunks), BATCH_SIZE)
        ]

    async def _embed_batch_with_retry(self, batch: list[TextChunk], batch_num: int) -> list[EmbeddedChunk]:
        for attempt in range(1, RETRY_ATTEMPTS+1):
            try:
                return await self._embed_batch(batch=batch)
            except Exception as e:
                if attempt < RETRY_ATTEMPTS:
                    logger.warning(
                        f"Embedding Failed | batch = {batch_num+1} | attempts = {attempt}/{RETRY_ATTEMPTS} | error = {e} | retrying in {RETRY_SLEEP_SECONDS}")
                    await asyncio.sleep(RETRY_SLEEP_SECONDS)
                else:
                    logger.error(
                        f"Embedding failed after {RETRY_ATTEMPTS} attempts | batch = {batch_num + 1} | error = {e} | skipping this batch")
                    return []

        return []

    async def _embed_batch(self, batch: list[TextChunk]) -> list[EmbeddedChunk]:
        texts = [chunk.chunk_text for chunk in batch]
        response: list[list[float]] = await self._client.aembed_documents(texts=texts)

        embedded_chunks: list[EmbeddedChunk] = []

        for i, chunk in enumerate(batch):
            embedding_vector = response[i]
            embedded_chunk = EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                nct_id=chunk.nct_id,
                chunk_text=chunk.chunk_text,
                chunk_index=chunk.chunk_index,
                source=chunk.source,
                word_count=chunk.word_count,
                embedding=embedding_vector
            )

            embedded_chunks.append(embedded_chunk)

        logger.info(f"Batch Embedded Successfully | chunks = {len(embedded_chunks)} | embedding_dims={len(embedded_chunks[0].embedding) if embedded_chunks else 0}")
        return embedded_chunks