from typing import Any, TypedDict, Annotated
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage

class SignalOutput(TypedDict):
    agent: str
    signal_type: str
    nct_id: str
    summary: str
    evidence: list
    confidence: float

class MosaicState(TypedDict):
    task: str
    nct_ids: list[str]
    max_studies: int
    messages: Annotated[list[BaseMessage], add_messages]
    signals: list[SignalOutput]
    agents_activated: list[str]
    final_brief: str
    run_complete: bool
    run_id: str
    error_log: list

