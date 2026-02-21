"""ShieldLM: Fast, deployment-aware prompt injection detection."""

__version__ = "0.1.0"


def __getattr__(name):
    if name == "ShieldLMDetector":
        from shieldlm.detector import ShieldLMDetector
        return ShieldLMDetector
    raise AttributeError(f"module 'shieldlm' has no attribute {name!r}")


__all__ = ["ShieldLMDetector", "__version__"]
