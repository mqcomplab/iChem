"""Reversible uint64 encoding for canonical ZINC22 identifiers.

ZINC22 commercial identifiers have the form ``ZINCtt00nnnnnnnn``, where
``t`` and ``n`` are radix-62 characters.  The fixed ``00`` source field means
the remaining ten characters fit in an unsigned 64-bit integer.
"""
from numbers import Integral

_RADIX62_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
_RADIX62_INDEX = {character: index for index, character in enumerate(_RADIX62_ALPHABET)}
_RADIX = len(_RADIX62_ALPHABET)
_PAYLOAD_LENGTH = 12
_ENCODED_LENGTH = 10
_MAX_PACKED_ZINC22_ID = _RADIX ** _ENCODED_LENGTH - 1


def pack_zinc22_uint64(zinc_id: str) -> int:
    """Pack a canonical commercial ZINC22 ID into a uint64-compatible int.

    Only IDs with the canonical ``ZINCtt00nnnnnnnn`` layout can be encoded.
    Rejecting other IDs is intentional: retaining their source field would
    require more than 64 bits and could otherwise create collisions.
    """
    zinc_id = str(zinc_id).strip()
    if not zinc_id.startswith("ZINC"):
        raise ValueError(f"Invalid ZINC22 ID {zinc_id!r}: expected a 'ZINC' prefix")

    payload = zinc_id[4:]
    if len(payload) != _PAYLOAD_LENGTH or payload[2:4] != "00":
        raise ValueError(
            f"Invalid ZINC22 ID {zinc_id!r}: expected 'ZINCtt00nnnnnnnn'"
        )

    packed = 0
    for character in payload[:2] + payload[4:]:
        try:
            digit = _RADIX62_INDEX[character]
        except KeyError as error:
            raise ValueError(
                f"Invalid ZINC22 ID {zinc_id!r}: {character!r} is not radix-62"
            ) from error
        packed = packed * _RADIX + digit
    return packed


def unpack_zinc22_uint64(packed_id: int) -> str:
    """Unpack a uint64-compatible int into a canonical commercial ZINC22 ID."""
    if not isinstance(packed_id, Integral):
        raise TypeError("packed_id must be an integer")
    packed_id = int(packed_id)
    if not 0 <= packed_id <= _MAX_PACKED_ZINC22_ID:
        raise ValueError(
            f"packed_id must be between 0 and {_MAX_PACKED_ZINC22_ID}, inclusive"
        )

    characters = ["0"] * _ENCODED_LENGTH
    for position in range(_ENCODED_LENGTH - 1, -1, -1):
        packed_id, digit = divmod(packed_id, _RADIX)
        characters[position] = _RADIX62_ALPHABET[digit]
    return f"ZINC{''.join(characters[:2])}00{''.join(characters[2:])}"
