"""Writing cards back through AnkiConnect.

Everything else in this project reads the collection and never touches it. This
module is the exception, so it is deliberately narrow: it adds one note, to one
deck, with one note type, tagged so that anything it created can be found and
undone.

AnkiConnect only answers while Anki is running, which is worth saying clearly
when it is not — a connection refused here means "open Anki", not "something is
broken".
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

ENDPOINT = "http://127.0.0.1:8765"
DECK = "General"
NOTE_TYPE = "code-article"
TAG = "vocab-gen"
TIMEOUT = 10


class AnkiError(RuntimeError):
    """AnkiConnect refused, or is not listening."""


def _call(action: str, **params):
    payload = json.dumps({"action": action, "version": 6, "params": params}).encode()
    request = urllib.request.Request(
        ENDPOINT, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            body = json.loads(response.read())
    except urllib.error.URLError as exc:
        raise AnkiError(
            f"Could not reach AnkiConnect at {ENDPOINT}.\n"
            "Anki has to be running, with the AnkiConnect add-on installed."
        ) from None
    except (TimeoutError, OSError) as exc:
        raise AnkiError(f"AnkiConnect did not respond: {exc}") from None

    if body.get("error"):
        raise AnkiError(str(body["error"]))
    return body.get("result")


def available() -> bool:
    try:
        _call("version")
        return True
    except AnkiError:
        return False


def _note(front: str, back: str, allow_duplicate: bool = False) -> dict:
    return {
        "deckName": DECK,
        "modelName": NOTE_TYPE,
        "fields": {"Front": front, "Back": back},
        "tags": [TAG],
        "options": {
            "allowDuplicate": allow_duplicate,
            # Scope the check to the deck rather than the whole collection.
            "duplicateScope": "deck",
        },
    }


def is_duplicate(front: str, back: str) -> bool:
    """Would Anki reject this as a duplicate of an existing card?"""
    result = _call("canAddNotes", notes=[_note(front, back)])
    return not (result and result[0])


def add_note(front: str, back: str, allow_duplicate: bool = False) -> int:
    """Create the card. Returns Anki's note id.

    A duplicate is reported rather than silently allowed: a second card for a
    different sense of a word is legitimate, so the decision belongs to the
    user, but it should be a decision.
    """
    note_id = _call("addNote", note=_note(front, back, allow_duplicate))
    if note_id is None:
        raise AnkiError("Anki accepted the request but created no note.")
    return int(note_id)
