import pytest

from science_rag.config import ConfigError, Settings


def test_defaults_need_no_environment(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    settings = Settings()
    assert settings.text_collection == "docs"
    assert settings.embed_dim == 3072


def test_from_env_reads_and_casts(monkeypatch):
    monkeypatch.setenv("TOP_K", "9")
    monkeypatch.setenv("MIN_IMAGE_SCORE", "0.7")
    monkeypatch.setenv("RENDER_DRAWING_PAGES", "true")
    monkeypatch.setenv("QDRANT_URL", "http://db:6333")
    settings = Settings.from_env()
    assert settings.top_k == 9
    assert settings.min_image_score == pytest.approx(0.7)
    assert settings.render_drawing_pages is True
    assert settings.qdrant_url == "http://db:6333"


def test_missing_keys_fail_only_when_used():
    settings = Settings()
    with pytest.raises(ConfigError, match="GEMINI_API_KEY"):
        settings.require_gemini_key()
    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        settings.require_groq_key()
    assert Settings(gemini_api_key="k").require_gemini_key() == "k"
