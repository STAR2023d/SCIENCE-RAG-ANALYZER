import pytest

from science_rag.retry import with_backoff


def test_retries_then_succeeds():
    calls = {"n": 0}
    delays: list[float] = []

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("temporary")
        return "ok"

    result = with_backoff(
        flaky, retry_on=(ConnectionError,), attempts=5, base_delay=10, sleep=delays.append
    )
    assert result == "ok"
    assert delays == [10, 20]  # linear backoff


def test_gives_up_and_reraises():
    def always_fails():
        raise ConnectionError("down")

    with pytest.raises(ConnectionError):
        with_backoff(always_fails, retry_on=(ConnectionError,), attempts=3, sleep=lambda _: None)


def test_does_not_retry_other_errors():
    calls = {"n": 0}

    def wrong():
        calls["n"] += 1
        raise ValueError("bug")

    with pytest.raises(ValueError):
        with_backoff(wrong, retry_on=(ConnectionError,), sleep=lambda _: None)
    assert calls["n"] == 1
