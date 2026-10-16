"""Test bootstrap: provide a dummy API key so imports succeed offline."""
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")
