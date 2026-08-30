"""NIMBUS — Non-growing Incremental Memory with Budgeted Utility Splitting.

Constant-footprint memory for LLM agents. The resident working set is a fixed
array of centroids that never grows; original text is never rewritten.
"""

from nimbus.core import CentroidCloud
from nimbus.store import ColdStore, HashEmbedder, SentenceTransformerEmbedder, OpenAIEmbedder
from nimbus.memory import Nimbus, Retrieval, parse_tags, tool_schema

__version__ = "0.1.0"
__all__ = [
    "Nimbus",
    "Retrieval",
    "CentroidCloud",
    "ColdStore",
    "HashEmbedder",
    "SentenceTransformerEmbedder",
    "OpenAIEmbedder",
    "parse_tags",
    "tool_schema",
]