from dataclasses import dataclass
from typing import Any
from ingestion.document_parser import ParsedPaper, ParsedStudy
from config.logging_config import setup_logging
logger = setup_logging(__name__)

CHUNK_SIZE = 500
OVERLAP_SIZE = 50


@dataclass
class TextChunk:
    chunk_id: str
    nct_id: str
    chunk_text: str
    chunk_index: int
    source: str
    word_count: int


class Chunker:
    def chunk_study(self, parsed_study: ParsedStudy) -> list[TextChunk]:
        full_text = self._build_full_study_text(parsed_study)
        chunks = self._split_into_chunks(
            text=full_text, nct_id=parsed_study.nct_id, source="study")
        logger.info(
            f"Chunked Study: {parsed_study.nct_id} into {len(chunks)} chunks")
        return chunks

    def chunk_paper(self, parsed_paper: ParsedPaper) -> list[TextChunk]:
        full_text = self._build_full_paper_text(parsed_paper)
        chunks = self._split_into_chunks(
            text=full_text, nct_id=parsed_paper.pmid, source="paper")
        logger.info(
            f"Chunked Paper: {parsed_paper.pmid} into {len(chunks)} chunks")
        return chunks

    def chunk_studies(self, studies: list[ParsedStudy]) -> list[TextChunk]:
        all_chunks: list[TextChunk] = []
        for study in studies:
            chunks = self.chunk_study(study)
            all_chunks.extend(chunks)
        logger.info(
            f"Chunked {len(studies)} studies into {len(all_chunks)} total chunks")
        return all_chunks

    def chunk_papers(self, papers: list[ParsedPaper]) -> list[TextChunk]:
        all_chunks: list[TextChunk] = []
        for paper in papers:
            chunks = self.chunk_paper(paper)
            all_chunks.extend(chunks)
        logger.info(
            f"Chunked {len(papers)} papers into {len(all_chunks)} total chunks")
        return all_chunks

    def _build_full_study_text(self, study: ParsedStudy) -> str:
        sections = []
        sections.append(f"NCT ID: {study.nct_id}")
        sections.append(f"Title: {study.title}")
        sections.append(f"Sponsor: {study.sponsor}")
        sections.append(f"Phase: {study.phase}")
        sections.append(f"Status: {study.status}")

        if study.conditions:
            sections.append(f"Conditions: {', '.join(study.conditions)}")

        if study.interventions:
            sections.append(f"Interventions: {', '.join(study.interventions)}")

        if study.primary_outcome:
            sections.append(
                f"Primary Outcomes: {', '.join(study.primary_outcome)}")

        if study.secondary_outcomes:
            sections.append(
                f"Secondary Outcomes: {', '.join(study.secondary_outcomes)}")

        if study.start_date:
            sections.append(f"Start Date: {study.start_date}")

        if study.completion_date:
            sections.append(f"Completion Date: {study.completion_date}")

        sections.append(
            F"Results Posted: {'Yes' if study.results_posted else 'No'}")

        if study.enrollment:
            sections.append(f"Enrollements: {study.enrollment} participants")

        if study.protocol_amendments:
            sections.append(
                f"Protocol Amendments: {', '.join(study.protocol_amendments)} filed")

        return "\n".join(sections)

    def _build_full_paper_text(self, paper: ParsedPaper) -> str:
        sections = []

        sections.append(f"PMID: {paper.pmid}")
        sections.append(f"Title: {paper.title}")

        if paper.journal:
            sections.append(f"Journal: {paper.journal}")

        if paper.pub_date:
            sections.append(f"Publication Date: {paper.pub_date}")

        if paper.authors:
            sections.append(f"Authors: {', '.join(paper.authors[:5])}")

        if paper.nct_ids_referenced:
            sections.append(
                f"NCT IDS referenced: {', '.join(paper.nct_ids_referenced)}")

        if paper.abstract:
            sections.append(f"Abstract: {paper.abstract}")

        return "\n".join(sections)

    def _split_into_chunks(self, text: str, nct_id: str, source: str) -> list[TextChunk]:
        words = text.split()
        if not words:
            logger.warning(
                f"Empty text for nct_id={nct_id}, source={source}. No chunks produced")
            return []
        chunks: list[TextChunk] = []
        chunk_index = 0
        step = CHUNK_SIZE - OVERLAP_SIZE

        for start in range(0, len(words), step):
            end = start + CHUNK_SIZE
            chunk_words = words[start:end]
            if not chunk_words:
                break
            chunk_text = " ".join(chunk_words)
            chunk_id = f"{nct_id}_chunk_{chunk_index}"
            word_count = len(chunk_words)
            chunk = TextChunk(
                chunk_id=chunk_id,
                nct_id=nct_id,
                chunk_text=chunk_text,
                chunk_index=chunk_index,
                source=source,
                word_count=word_count
            )
            chunks.append(chunk)
            chunk_index += 1
            logger.info(f"Split Completed | nct_id = {nct_id} | source = {source} | chunk_index = {chunk_index} | word_count = {word_count} | chunk_size = {CHUNK_SIZE} | overlap_size = {OVERLAP_SIZE}")
            return chunk