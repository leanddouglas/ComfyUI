"""Opaque keyset-pagination cursor for /api/assets.

Wire format aligns with the cloud Go implementation in
`common/pagination/cursor.go` so the frontend sees one contract across
runtimes. Payload JSON uses short keys to keep the encoded length small:

    {"s": <sort_field>, "v": <value>, "id": <id>, "o": <order>}

The `o` key binds the cursor to the sort direction it was minted under,
so replaying a `desc` cursor against an `asc` request fails with
``INVALID_CURSOR`` rather than silently walking the wrong direction.
Decoders accept payloads without `o` for backward compatibility with
cursors minted before the binding was introduced (these skip the order
check); new cursors always include it. Cloud has a follow-up to mirror
the field — until then, Python and cloud cursors differ by exactly the
`o` key.

Encoding is base64url with no padding. JSON serialization escapes `<`,
`>`, `&`, U+2028, and U+2029 to match Go's default `json.Marshal`
behavior so asset names containing those characters produce
byte-identical cursors across runtimes.

Time values are serialized as Unix microseconds (UTC) — microsecond
precision matches PostgreSQL's `timestamp` type, so a cursor minted from
a stored timestamp compares back exactly without rounding rows in the
same millisecond bucket.
"""
from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Optional


class InvalidCursorError(ValueError):
    """Raised on a malformed, oversized, or unsupported-sort-field cursor.

    Map to a 400 response with code ``INVALID_CURSOR`` at the handler.
    """


# Wire-format length caps. Cursors are user-controlled, so caps protect the
# decode path from oversized allocations and downstream SQL predicates from
# unbounded strings.
#
# MAX_CURSOR_VALUE_LENGTH is 512 (vs cloud's 256) to fit OSS's
# `AssetReference.name` column max (String(512)) — otherwise a long-named
# asset would mint a cursor the same server then refuses on the next request.
# Cloud's data model has shorter names so its lower cap is fine there;
# cross-runtime byte-identity is unaffected because no real cloud cursor ever
# carries a value > 256.
MAX_ENCODED_CURSOR_LENGTH = 1024
MAX_CURSOR_VALUE_LENGTH = 512
MAX_CURSOR_ID_LENGTH = 128


@dataclass(frozen=True)
class CursorPayload:
    sort_field: str
    value: str
    id: str
    # None means "minted by a producer that did not bind order" (e.g. a cloud
    # cursor from before BE-944's follow-up lands). New cursors always set it.
    order: str | None = None


# Order direction tokens. Mirrored on the cloud follow-up so cursors carrying
# `o` are interchangeable between runtimes once both sides ship the field.
_VALID_ORDERS = ("asc", "desc")


def encode_cursor(sort_field: str, value: str, id: str, order: str = "desc") -> str:
    """Encode a cursor payload as a base64url (no-padding) string.

    `order` binds the cursor to the sort direction it was minted under so a
    later request with a flipped `order` query parameter is rejected with
    ``INVALID_CURSOR`` rather than silently walking the wrong direction.
    """
    if order not in _VALID_ORDERS:
        raise ValueError(f"order must be one of {_VALID_ORDERS}, got {order!r}")
    payload = {"s": sort_field, "v": value, "id": id, "o": order}
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    # Go's default `json.Marshal` escapes these characters in string values; we
    # do the same so an asset name containing one of them produces byte-
    # identical cursors across runtimes. None of these characters appear in
    # JSON structural syntax, so a global replace on the serialized output is
    # safe — it can only touch characters from the encoded values.
    raw = (
        raw.replace("<", "\\u003c")
           .replace(">", "\\u003e")
           .replace("&", "\\u0026")
           .replace(" ", "\\u2028")
           .replace(" ", "\\u2029")
    )
    return base64.urlsafe_b64encode(raw.encode("utf-8")).rstrip(b"=").decode("ascii")


def encode_cursor_from_time(sort_field: str, t: datetime, id: str, order: str = "desc") -> str:
    """Encode a time-typed cursor at Unix microsecond precision.

    Accepts an aware datetime (any timezone) and normalizes to UTC. Naive
    datetimes are rejected so callers can't accidentally encode the local
    wall-clock value of a UTC-stored timestamp.
    """
    if t.tzinfo is None:
        raise ValueError("encode_cursor_from_time requires an aware datetime")
    micros = _datetime_to_unix_micros(t.astimezone(timezone.utc))
    return encode_cursor(sort_field, str(micros), id, order=order)


def decode_cursor(
    cursor: str,
    allowed_sort_fields: Iterable[str],
    expected_order: str | None = None,
) -> CursorPayload:
    """Parse an opaque cursor.

    ``allowed_sort_fields`` is the endpoint's accepted sort-field list — a
    cursor carrying a field outside this set is rejected so a cursor minted
    for one column can't be replayed against another (e.g. a ``created_at``
    timestamp string compared against a ``name`` column).

    ``expected_order`` (``"asc"``/``"desc"``), when supplied, must match the
    payload's ``o`` field. Cursors minted without ``o`` (e.g. by an older
    cloud build) pass the check unconditionally — the binding is best-effort
    until both runtimes ship the field.

    Passing no allowed fields rejects every cursor.
    """
    if len(cursor) > MAX_ENCODED_CURSOR_LENGTH:
        raise InvalidCursorError("cursor exceeds maximum length")

    try:
        # urlsafe_b64decode requires correct padding; we strip on encode, so
        # restore the trailing '=' pad here.
        padding = "=" * (-len(cursor) % 4)
        raw = base64.urlsafe_b64decode(cursor + padding)
    except (ValueError, base64.binascii.Error) as e:
        raise InvalidCursorError(f"encoding: {e}") from e

    try:
        decoded = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise InvalidCursorError(f"payload: {e}") from e

    if not isinstance(decoded, dict):
        raise InvalidCursorError("payload: expected object")

    sort_field = decoded.get("s")
    value = decoded.get("v")
    id = decoded.get("id")
    order = decoded.get("o")  # may be absent on legacy cursors

    if not isinstance(sort_field, str) or not isinstance(value, str) or not isinstance(id, str):
        raise InvalidCursorError("payload: missing or non-string s/v/id")

    if id == "":
        raise InvalidCursorError("missing id")
    if len(id) > MAX_CURSOR_ID_LENGTH:
        raise InvalidCursorError("id exceeds maximum length")
    if len(value) > MAX_CURSOR_VALUE_LENGTH:
        raise InvalidCursorError("value exceeds maximum length")

    if sort_field not in allowed_sort_fields:
        raise InvalidCursorError(f"unsupported sort field {sort_field!r}")

    if order is not None and not isinstance(order, str):
        raise InvalidCursorError("payload: non-string o")
    if order is not None and order not in _VALID_ORDERS:
        raise InvalidCursorError(f"unsupported order {order!r}")
    if expected_order is not None and order is not None and order != expected_order:
        raise InvalidCursorError(
            f"cursor order {order!r} does not match request order {expected_order!r}"
        )

    return CursorPayload(sort_field=sort_field, value=value, id=id, order=order)


def decode_cursor_time(payload: Optional[CursorPayload]) -> datetime:
    """Parse a time-typed cursor value as Unix microseconds, returning UTC."""
    if payload is None:
        raise InvalidCursorError("nil cursor payload")
    try:
        micros = int(payload.value)
    except ValueError as e:
        raise InvalidCursorError(f"value is not a valid timestamp: {e}") from e
    try:
        return _unix_micros_to_datetime(micros)
    except (OverflowError, OSError, ValueError) as e:
        # Crafted out-of-range microseconds (e.g. > datetime.MAX_YEAR) blow up
        # in fromtimestamp / datetime construction. Map to 400, not 500.
        raise InvalidCursorError(f"value is out of representable range: {e}") from e


def decode_cursor_int(payload: Optional[CursorPayload]) -> int:
    """Parse a cursor value as a base-10 integer."""
    if payload is None:
        raise InvalidCursorError("nil cursor payload")
    try:
        return int(payload.value)
    except ValueError as e:
        raise InvalidCursorError(f"value is not a valid integer: {e}") from e


_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _datetime_to_unix_micros(t: datetime) -> int:
    """Convert an aware UTC datetime to Unix microseconds (integer math)."""
    delta = t - _EPOCH
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


def _unix_micros_to_datetime(micros: int) -> datetime:
    """Convert Unix microseconds to a UTC datetime, preserving precision."""
    seconds, micro_remainder = divmod(micros, 1_000_000)
    return datetime.fromtimestamp(seconds, tz=timezone.utc).replace(microsecond=micro_remainder)
