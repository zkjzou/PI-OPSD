"""AlfworldEnv._post retries a stale pooled connection on /reset ONLY.

The bug: the shared pooled requests.Session (server_pool.py, max_retries=0) can hand back a socket the
env server has already reaped, and the resulting ConnectionError propagated out of agent_loop and killed
three multi-hour training runs. /step must NOT be retried -- it is not idempotent.
"""

import types

import pytest
import requests

from opsd.envs.alfworld.env import AlfworldEnv


class FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.exceptions.HTTPError(f"{self.status_code}")

    def json(self):
        return self._payload


class FakeSession:
    """Fails the first `n_fail` posts with a stale-connection error, then succeeds."""

    def __init__(self, n_fail=0, status=200):
        self.n_fail = n_fail
        self.status = status
        self.calls = 0
        self.closed = 0

    def post(self, url, json=None, timeout=None):
        self.calls += 1
        if self.calls <= self.n_fail:
            raise requests.exceptions.ConnectionError(
                "('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))"
            )
        return FakeResp({"ok": True, "url": url}, status=self.status)

    def close(self):
        self.closed += 1


def _env(session):
    """A bare AlfworldEnv with just the attributes _post touches (no server, no __init__)."""
    e = AlfworldEnv.__new__(AlfworldEnv)
    e._pool = types.SimpleNamespace(session=session)
    e._server_url = "http://fake:8100"
    e.http_timeout = 5
    return e


def test_reset_recovers_from_a_stale_connection():
    s = FakeSession(n_fail=2)  # fails twice, succeeds on the 3rd
    out = _env(s)._post("/reset", {"game_file": "g"})
    assert out["ok"] is True
    assert s.calls == 3, "should have retried twice"
    assert s.closed == 2, "the poisoned pooled socket must be dropped before each retry"


def test_reset_still_raises_once_attempts_are_exhausted():
    s = FakeSession(n_fail=99)
    with pytest.raises(requests.exceptions.ConnectionError):
        _env(s)._post("/reset", {})
    assert s.calls == AlfworldEnv._RETRY_ATTEMPTS, "must not retry forever"


def test_step_is_NOT_retried():
    """/step is not idempotent: a retry after the server processed it would double-apply the action."""
    s = FakeSession(n_fail=1)
    with pytest.raises(requests.exceptions.ConnectionError):
        _env(s)._post("/step", {"action": "go to shelf 1"})
    assert s.calls == 1, "/step must fail fast, not retry"


def test_http_error_status_is_not_retried():
    """A 500 means the request REACHED the app -- that is a real failure, not a stale socket."""
    s = FakeSession(n_fail=0, status=500)
    with pytest.raises(requests.exceptions.HTTPError):
        _env(s)._post("/reset", {})
    assert s.calls == 1, "raise_for_status must sit outside the retry loop"


def test_reset_is_the_only_retryable_path():
    assert AlfworldEnv._RETRYABLE_PATHS == ("/reset",)
