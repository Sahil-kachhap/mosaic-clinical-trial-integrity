from functools import lru_cache
from graph.hitl import HITLGate
from memory.episodic_store import EpisodicStore
from memory.procedural_store import ProceduralStore
from memory.semantic_store import SemanticStore
from graph.graph_builder import mosaic_graph

@lru_cache(maxsize=1)
def get_hitl_gate() -> HITLGate:
    return HITLGate()

@lru_cache(maxsize=1)
def get_episodic_store() -> EpisodicStore:
    return EpisodicStore()

@lru_cache(maxsize=1)
def get_procedural_store() -> ProceduralStore:
    return ProceduralStore()

@lru_cache(maxsize=1)
def get_semantic_store() -> SemanticStore:
    return SemanticStore()

def get_graph():
    return mosaic_graph