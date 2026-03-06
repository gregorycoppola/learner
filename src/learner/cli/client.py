"""HTTP client for the Learner API."""
import httpx

DEFAULT_BASE_URL = "http://localhost:8000"


def get_client(base_url: str = DEFAULT_BASE_URL) -> httpx.Client:
    return httpx.Client(base_url=base_url, timeout=30.0)