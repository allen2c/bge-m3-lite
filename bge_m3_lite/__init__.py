"""bge-m3-lite: CPU inference for BAAI/bge-m3, onnxruntime is the only dependency."""

from bge_m3_lite.embedder import BGEM3Embedder
from bge_m3_lite.serving import AsyncEmbedder

__version__ = "0.6.1"
__all__ = ["AsyncEmbedder", "BGEM3Embedder", "__version__"]
