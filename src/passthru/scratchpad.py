"""Fetch notes from Wispr Flow Scratchpad over the Wispr MCP server.

Speaks MCP directly over Streamable HTTP using only the standard library, so the
project gains no MCP SDK dependency. Auth resolves from PASSTHRU_WISPR_TOKEN if set,
otherwise from an mcporter credential store that a previous `mcporter auth wispr`
populated. The token is never logged, printed, or included in any error message.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ENDPOINT = "https://api.wisprflow.ai/connect/mcp"
PROTOCOL_VERSION = "2026-07-28"
CLIENT_NAME = "passthru"
CLIENT_VERSION = "0.1.0"

_MCPORTER_CREDENTIALS = Path.home() / ".mcporter" / "credentials.json"
_SERVER_HINT = "wispr"


class WisprError(RuntimeError):
    """A failure whose message is safe to show a user."""


def resolve_token() -> str:
    """Return a bearer token, or explain precisely why one is unavailable."""
    from_env = os.environ.get("PASSTHRU_WISPR_TOKEN")
    if from_env:
        return from_env.strip()

    if not _MCPORTER_CREDENTIALS.exists():
        raise WisprError(
            f"no credentials at {_MCPORTER_CREDENTIALS} and PASSTHRU_WISPR_TOKEN is unset"
        )

    try:
        store = json.loads(_MCPORTER_CREDENTIALS.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise WisprError(f"credential store at {_MCPORTER_CREDENTIALS} is unreadable: {exc}") from exc

    for entry in store.get("entries", {}).values():
        if not isinstance(entry, dict):
            continue
        if _SERVER_HINT not in str(entry.get("serverUrl", "")):
            continue
        tokens = entry.get("tokens") or {}
        token = tokens.get("access_token")
        if not token:
            continue
        expires_at = tokens.get("expires_at")
        if isinstance(expires_at, (int, float)) and expires_at:
            import time

            if time.time() > float(expires_at):
                raise WisprError(
                    "stored Wispr token has expired; refresh it with: mcporter auth wispr"
                )
        return str(token)

    raise WisprError(
        f"no Wispr token in {_MCPORTER_CREDENTIALS}; run: mcporter auth wispr"
    )


def _post(token: str, payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    """Send one JSON-RPC message and return the decoded result object."""
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        ENDPOINT,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {token}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        if exc.code in (401, 403):
            raise WisprError(
                f"Wispr rejected the access token (HTTP {exc.code}); refresh it with: "
                "mcporter auth wispr"
            ) from exc
        raise WisprError(f"Wispr returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise WisprError(f"could not reach Wispr at {ENDPOINT}: {exc.reason}") from exc

    message = _first_jsonrpc(raw)
    if message is None:
        raise WisprError("Wispr returned a response with no JSON-RPC message in it")
    if "error" in message:
        detail = message["error"].get("message", "no detail given")
        raise WisprError(f"Wispr returned an error: {detail}")
    return message.get("result", {})


def _first_jsonrpc(raw: str) -> dict[str, Any] | None:
    """Pull the JSON-RPC message out of a plain-JSON or SSE-framed response."""
    stripped = raw.strip()
    if stripped.startswith("{"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return None

    message = None
    for line in raw.splitlines():
        if line.startswith("data:"):
            chunk = line[5:].strip()
            if not chunk:
                continue
            try:
                message = json.loads(chunk)
            except json.JSONDecodeError:
                continue
    return message


def _initialize(token: str) -> None:
    _post(
        token,
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": CLIENT_NAME, "version": CLIENT_VERSION},
            },
        },
    )
    # The spec requires this notification before any other call. Servers that have
    # retired sessions ignore it; sending it is harmless and keeps us conformant.
    try:
        _post(
            token,
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
        )
    except WisprError:
        pass


def _call(token: str, tool: str, arguments: dict[str, Any], call_id: int) -> Any:
    result = _post(
        token,
        {
            "jsonrpc": "2.0",
            "id": call_id,
            "method": "tools/call",
            "params": {"name": tool, "arguments": arguments},
        },
    )
    if result.get("isError"):
        text = " ".join(
            block.get("text", "")
            for block in result.get("content", [])
            if isinstance(block, dict)
        )
        raise WisprError(f"{tool} failed: {text or 'no detail given'}")
    return _decode_tool_result(result)


def _decode_tool_result(result: dict[str, Any]) -> Any:
    for block in result.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            try:
                return json.loads(block["text"])
            except (json.JSONDecodeError, KeyError):
                return block["text"]
    return result.get("structuredContent", result)


def list_notes(token: str, query: str = "") -> list[dict[str, Any]]:
    """Return Scratchpad notes, newest first."""
    _initialize(token)
    payload = _call(token, "search_scratchpad_notes", {"query": query}, 2)
    if isinstance(payload, dict):
        return list(payload.get("notes") or [])
    if isinstance(payload, list):
        return payload
    return []


def get_note(token: str, note_id: str) -> dict[str, Any]:
    """Return one Scratchpad note in full."""
    payload = _call(token, "get_scratchpad_note", {"note_id": note_id}, 3)
    return payload if isinstance(payload, dict) else {"id": note_id, "content": str(payload)}


def main(argv: list[str] | None = None) -> int:
    """Print each note's title and body. Returns a process exit code."""
    argv = sys.argv[1:] if argv is None else argv
    query = argv[0] if argv else ""

    try:
        token = resolve_token()
        notes = list_notes(token, query)
    except WisprError as exc:
        print(f"passthru: {exc}", file=sys.stderr)
        return 1

    if not notes:
        where = f" matching {query!r}" if query else ""
        print(f"passthru: Wispr reachable but Scratchpad has no notes{where}", file=sys.stderr)
        return 1

    for note in notes:
        title = note.get("title") or "(untitled)"
        body = (note.get("content") or "").strip()
        if not body and note.get("id"):
            # search returns an excerpt only, so pull the full body per note
            try:
                body = (get_note(token, str(note["id"])).get("content") or "").strip()
            except WisprError as exc:
                print(f"passthru: could not read note {note['id']}: {exc}", file=sys.stderr)
                return 1
        print(f"# {title}\n{body}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def cli(argv: list[str] | None = None) -> int:
    """Entry point for the `passthru-scratchpad` command."""
    import sys

    argv = sys.argv[1:] if argv is None else argv
    query = argv[0] if argv else ""
    try:
        token = resolve_token()
        notes = list_notes(token, query)
    except WisprError as exc:
        print(f"passthru-scratchpad: {exc}", file=sys.stderr)
        return 1

    if not notes:
        where = f" matching {query!r}" if query else ""
        print(
            f"passthru-scratchpad: Wispr reachable but Scratchpad has no notes{where}",
            file=sys.stderr,
        )
        return 1

    for note in notes:
        title = note.get("title") or "(untitled)"
        body = (note.get("content") or "").strip()
        if not body and note.get("id"):
            try:
                body = (get_note(token, str(note["id"])).get("content") or "").strip()
            except WisprError as exc:
                print(
                    f"passthru-scratchpad: could not read note {note['id']}: {exc}",
                    file=sys.stderr,
                )
                return 1
        print(f"# {title}\n{body}\n")
    return 0
