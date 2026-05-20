"""Tests for app.assets.services.cursor.

The wire format must stay byte-identical with the cloud Go implementation
(common/pagination/cursor.go in Comfy-Org/cloud) so the frontend sees one
contract across runtimes. The byte-identity fixture below mirrors the Go
test cases — any drift here means cloud and OSS minted different cursors
for the same triple, which would break FE pagination across backends.
"""
from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone

import pytest

from app.assets.services.cursor import (
    MAX_CURSOR_ID_LENGTH,
    MAX_CURSOR_VALUE_LENGTH,
    MAX_ENCODED_CURSOR_LENGTH,
    CursorPayload,
    InvalidCursorError,
    decode_cursor,
    decode_cursor_int,
    decode_cursor_time,
    encode_cursor,
    encode_cursor_from_time,
)


ALLOWED = ("created_at", "updated_at", "name", "size")


class TestRoundTrip:
    @pytest.mark.parametrize(
        "sort_field, value, id",
        [
            ("created_at", "1716200000000000", "a1b2c3d4-e5f6-7a89-b0c1-d2e3f4a5b6c7"),
            ("size", "1024", "asset-123"),
            ("name", "my-asset.png", "asset-abc"),
            ("name", "résumé.txt", "asset-uni"),
        ],
    )
    def test_encode_decode(self, sort_field, value, id):
        encoded = encode_cursor(sort_field, value, id)
        assert encoded != ""
        payload = decode_cursor(encoded, ALLOWED)
        assert payload.sort_field == sort_field
        assert payload.value == value
        assert payload.id == id


class TestTimeCursor:
    def test_microsecond_precision_preserved(self):
        # Pick a time with non-zero microseconds — encoding at ms would lose the µs.
        ts = datetime(2024, 5, 20, 12, 53, 20, 123456, tzinfo=timezone.utc)
        encoded = encode_cursor_from_time("created_at", ts, "id-1")
        payload = decode_cursor(encoded, ALLOWED)
        # Value must be a microsecond integer string, not a millisecond one.
        assert payload.value == "1716209600123456"
        decoded = decode_cursor_time(payload)
        assert decoded == ts

    def test_decode_returns_utc(self):
        payload = CursorPayload(sort_field="created_at", value="1716200000123456", id="id-1")
        decoded = decode_cursor_time(payload)
        assert decoded.tzinfo == timezone.utc

    def test_naive_datetime_rejected_on_encode(self):
        naive = datetime(2024, 5, 20, 12, 0, 0)
        with pytest.raises(ValueError):
            encode_cursor_from_time("created_at", naive, "id-1")

    def test_non_integer_value_rejected_on_decode(self):
        with pytest.raises(InvalidCursorError):
            decode_cursor_time(CursorPayload("created_at", "not-a-number", "id-1"))

    def test_none_payload_rejected(self):
        with pytest.raises(InvalidCursorError):
            decode_cursor_time(None)

    def test_non_utc_aware_normalized(self):
        # Same instant, different timezone — must encode to the same micros.
        utc_ts = datetime(2024, 5, 20, 12, 0, 0, tzinfo=timezone.utc)
        offset_ts = utc_ts.astimezone(timezone(timedelta(hours=-5)))
        assert encode_cursor_from_time("created_at", utc_ts, "x") == encode_cursor_from_time(
            "created_at", offset_ts, "x"
        )


class TestIntCursor:
    def test_decode_int(self):
        assert decode_cursor_int(CursorPayload("size", "1024", "id-1")) == 1024

    def test_decode_int_rejects_non_int(self):
        with pytest.raises(InvalidCursorError):
            decode_cursor_int(CursorPayload("size", "abc", "id-1"))

    def test_decode_int_rejects_none(self):
        with pytest.raises(InvalidCursorError):
            decode_cursor_int(None)


class TestInvalidInputs:
    def test_oversized_cursor(self):
        oversized = "a" * (MAX_ENCODED_CURSOR_LENGTH + 1)
        with pytest.raises(InvalidCursorError, match="maximum length"):
            decode_cursor(oversized, ALLOWED)

    def test_not_base64(self):
        with pytest.raises(InvalidCursorError):
            decode_cursor("not base64!!!", ALLOWED)

    def test_not_json(self):
        encoded = base64.urlsafe_b64encode(b"definitely not json").rstrip(b"=").decode("ascii")
        with pytest.raises(InvalidCursorError):
            decode_cursor(encoded, ALLOWED)

    def test_empty_id(self):
        encoded = encode_cursor("created_at", "1", "")
        with pytest.raises(InvalidCursorError, match="missing id"):
            decode_cursor(encoded, ALLOWED)

    def test_oversized_id(self):
        encoded = encode_cursor("created_at", "1", "a" * (MAX_CURSOR_ID_LENGTH + 1))
        with pytest.raises(InvalidCursorError, match="id exceeds maximum length"):
            decode_cursor(encoded, ALLOWED)

    def test_oversized_value(self):
        encoded = encode_cursor("created_at", "v" * (MAX_CURSOR_VALUE_LENGTH + 1), "id-1")
        with pytest.raises(InvalidCursorError, match="value exceeds maximum length"):
            decode_cursor(encoded, ALLOWED)

    def test_unsupported_sort_field(self):
        encoded = encode_cursor("execution_time", "1", "id-1")
        with pytest.raises(InvalidCursorError, match="unsupported sort field"):
            decode_cursor(encoded, ALLOWED)

    def test_no_allowed_fields_rejects_everything(self):
        encoded = encode_cursor("created_at", "1", "id-1")
        with pytest.raises(InvalidCursorError):
            decode_cursor(encoded, ())

    def test_non_dict_payload_rejected(self):
        encoded = base64.urlsafe_b64encode(b'["array","not","dict"]').rstrip(b"=").decode("ascii")
        with pytest.raises(InvalidCursorError, match="expected object"):
            decode_cursor(encoded, ALLOWED)


class TestEncodeAtCapsFits:
    def test_max_field_lengths_fit_wire_cap(self):
        # Worst-case payload: value and id at their per-field caps, with a long
        # sort field name. The encoded cursor must fit within MAX_ENCODED_CURSOR_LENGTH
        # so the wire cap cannot reject a cursor the encoder mints at the per-field caps.
        value = "v" * MAX_CURSOR_VALUE_LENGTH
        id = "i" * MAX_CURSOR_ID_LENGTH
        sort_field = "very_long_sort_field_name"

        encoded = encode_cursor(sort_field, value, id)
        assert len(encoded) <= MAX_ENCODED_CURSOR_LENGTH
        payload = decode_cursor(encoded, (sort_field,))
        assert payload.value == value
        assert payload.id == id


class TestDatetimeOverflow:
    """Crafted cursors with extreme micros must map to InvalidCursorError,
    not OverflowError/OSError leaking as 500.
    """

    @pytest.mark.parametrize(
        "micros_str",
        [
            "999999999999999999999",   # 10^21 µs — past datetime.MAX_YEAR by ~14 orders
            "-999999999999999999999",  # symmetric negative — pre-epoch overflow
        ],
    )
    def test_out_of_range_micros_rejected(self, micros_str):
        encoded = encode_cursor("created_at", micros_str, "asset-x")
        payload = decode_cursor(encoded, ALLOWED)
        with pytest.raises(InvalidCursorError):
            decode_cursor_time(payload)


class TestEncoderDecoderSymmetry:
    """The encoder must reject inputs the decoder rejects, or the same server
    will mint a cursor it then 400s on the next request.
    """

    def test_long_name_within_cap_round_trips(self):
        """OSS assets allow names up to 512 chars (`String(512)`); cursor must
        handle that. Cloud's lower cap is acceptable on its side because the
        cloud schema doesn't permit names that long."""
        long_name = "n" * MAX_CURSOR_VALUE_LENGTH
        encoded = encode_cursor("name", long_name, "asset-x")
        payload = decode_cursor(encoded, ALLOWED)
        assert payload.value == long_name


class TestOrderBinding:
    def test_order_baked_into_payload(self):
        encoded = encode_cursor("created_at", "1", "id-1", order="asc")
        payload = decode_cursor(encoded, ALLOWED)
        assert payload.order == "asc"

    def test_mismatched_order_rejected(self):
        encoded = encode_cursor("created_at", "1", "id-1", order="desc")
        with pytest.raises(InvalidCursorError, match="does not match request order"):
            decode_cursor(encoded, ALLOWED, expected_order="asc")

    def test_matching_order_accepted(self):
        encoded = encode_cursor("created_at", "1", "id-1", order="desc")
        payload = decode_cursor(encoded, ALLOWED, expected_order="desc")
        assert payload.order == "desc"

    def test_invalid_order_token_rejected_on_encode(self):
        with pytest.raises(ValueError):
            encode_cursor("created_at", "1", "id-1", order="sideways")

    def test_invalid_order_token_rejected_on_decode(self):
        # Hand-craft a payload with an illegal `o` value.
        raw = b'{"s":"name","v":"x","id":"id-1","o":"sideways"}'
        encoded = base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
        with pytest.raises(InvalidCursorError, match="unsupported order"):
            decode_cursor(encoded, ALLOWED)

    def test_legacy_cursor_without_order_accepted(self):
        """Cursors minted by a producer that didn't include `o` (e.g. an older
        cloud build) must still decode — the order binding is best-effort
        until cloud mirrors the field."""
        raw = b'{"s":"name","v":"x","id":"id-1"}'
        encoded = base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
        payload = decode_cursor(encoded, ALLOWED, expected_order="desc")
        assert payload.order is None  # binding skipped, decode succeeds


class TestGoCompatJsonEscaping:
    """An asset name containing `<`, `>`, `&`, U+2028, or U+2029 must encode
    to the same bytes Go's default `json.Marshal` would produce. The fixtures
    below are generated by Go's encoder; any drift here means cross-runtime
    byte-identity for HTML-significant characters is broken.
    """

    @pytest.mark.parametrize(
        "value, escaped_substring",
        [
            ("foo<bar>.png", "\\u003c"),  # `<` escaped
            ("foo<bar>.png", "\\u003e"),  # `>` escaped
            ("foo&bar.png", "\\u0026"),
            ("foo bar.png", "\\u2028"),  # JS line separator
            ("foo bar.png", "\\u2029"),  # JS paragraph separator
        ],
    )
    def test_html_significant_chars_escaped(self, value, escaped_substring):
        encoded = encode_cursor("name", value, "id-1")
        decoded_bytes = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        assert escaped_substring in decoded_bytes.decode("ascii"), (
            f"Expected {escaped_substring!r} in serialized payload, got: {decoded_bytes!r}"
        )

    def test_value_round_trips_through_escape(self):
        """Encoding then decoding a value with `<>&` should yield the original
        string — the escape only affects the wire form, not the decoded value."""
        original = "foo<&>bar.png"
        encoded = encode_cursor("name", original, "id-1")
        payload = decode_cursor(encoded, ALLOWED)
        assert payload.value == original


class TestByteIdentityFixtures:
    """Pin the wire format so it doesn't drift silently.

    NOTE — these fixtures will need updates on the cloud side once cloud
    mirrors the `o` (order binding) field. Until then, Python cursors and
    cloud cursors differ by exactly that key. The structural format (`s`,
    `v`, `id` plus base64url + Go-compat escaping) remains aligned.
    """

    @pytest.mark.parametrize(
        "sort_field, value, id, order",
        [
            ("created_at", "1716200000000000", "a1b2c3d4-e5f6-7a89-b0c1-d2e3f4a5b6c7", "desc"),
            ("size", "1024", "asset-123", "asc"),
            ("name", "my-asset.png", "asset-abc", "desc"),
            ("name", "foo<bar>&baz.png", "asset-html", "desc"),  # exercises Go-compat escaping
        ],
    )
    def test_encoded_payload_shape_pinned(self, sort_field, value, id, order):
        encoded = encode_cursor(sort_field, value, id, order=order)
        decoded_bytes = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        decoded_str = decoded_bytes.decode("ascii")
        # Keys appear in the documented order; binding `o` is present.
        for needle in (f'"s":"{sort_field}"', f'"id":"{id}"', f'"o":"{order}"'):
            assert needle in decoded_str, (
                f"Expected {needle!r} in payload {decoded_str!r}"
            )
