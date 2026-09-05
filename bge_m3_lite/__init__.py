"""bge-m3-lite: CPU inference for BAAI/bge-m3, onnxruntime is the only dependency."""

from bge_m3_lite.embedder import BGEM3Embedder

__version__ = "0.1.0"
__all__ = ["BGEM3Embedder", "__version__"]
