import pytest

from iChem.bitbirch.zinc22_ids import pack_zinc22_uint64, unpack_zinc22_uint64


@pytest.mark.parametrize(
    "zinc_id",
    ["ZINCaa0000000000", "ZINCzz00ZZZZZZZZ", "ZINC000000000000"],
)
def test_zinc22_uint64_round_trip(zinc_id):
    assert unpack_zinc22_uint64(pack_zinc22_uint64(zinc_id)) == zinc_id


@pytest.mark.parametrize(
    "zinc_id",
    ["ZINCaa0100000000", "ZINCaa00not-valid", "not-a-zinc-id"],
)
def test_pack_zinc22_uint64_rejects_noncanonical_ids(zinc_id):
    with pytest.raises(ValueError):
        pack_zinc22_uint64(zinc_id)
