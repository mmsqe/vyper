import itertools

import pytest


def test_bytes_literal_code(get_contract):
    bytes_literal_code = """
@external
def foo() -> Bytes[5]:
    return b"horse"

@external
def bar() -> Bytes[10]:
    return concat(b"b", b"a", b"d", b"m", b"i", b"", b"nton")

@external
def baz() -> Bytes[40]:
    return concat(b"0123456789012345678901234567890", b"12")

@external
def baz2() -> Bytes[40]:
    return concat(b"01234567890123456789012345678901", b"12")

@external
def baz3() -> Bytes[40]:
    return concat(b"0123456789012345678901234567890", b"1")

@external
def baz4() -> Bytes[100]:
    return concat(b"01234567890123456789012345678901234567890123456789",
                  b"01234567890123456789012345678901234567890123456789")
    """

    c = get_contract(bytes_literal_code)
    assert c.foo() == b"horse"
    assert c.bar() == b"badminton"
    assert c.baz() == b"012345678901234567890123456789012"
    assert c.baz2() == b"0123456789012345678901234567890112"
    assert c.baz3() == b"01234567890123456789012345678901"
    assert c.baz4() == b"0123456789" * 10

    print("Passed string literal test")


@pytest.mark.parametrize("i,e,_s", itertools.product([95, 96, 97], [63, 64, 65], [31, 32, 33]))
def test_bytes_literal_splicing_fuzz(get_contract, i, e, _s):
    kode = f"""
moo: Bytes[100]

@external
def foo(s: uint256, L: uint256) -> Bytes[100]:
    x: int128 = 27
    r: Bytes[100] = slice(b"{("c" * i)}", s, L)
    y: int128 = 37
    if x * y == 999:
        return r
    return b"3434346667777"

@external
def bar(s: uint256, L: uint256) -> Bytes[100]:
    self.moo = b"{("c" * i)}"
    x: int128 = 27
    r: Bytes[100] = slice(self.moo, s, L)
    y: int128  = 37
    if x * y == 999:
        return r
    return b"3434346667777"

@external
def baz(s: uint256, L: uint256) -> Bytes[100]:
    x: int128 = 27
    self.moo = slice(b"{("c" * i)}", s, L)
    y: int128 = 37
    if x * y == 999:
        return self.moo
    return b"3434346667777"
    """

    c = get_contract(kode)
    o1 = c.foo(_s, e - _s)
    o2 = c.bar(_s, e - _s)
    o3 = c.baz(_s, e - _s)
    assert o1 == o2 == o3 == b"c" * (e - _s), (i, _s, e - _s, o1, o2, o3)

    print("Passed string literal splicing fuzz-test")


def test_large_bytes_constant_roundtrip(get_contract):
    # 256 bytes of dense (high-entropy) content, well above the content-aware
    # CODECOPY lowering threshold. Verifies that a source-level constant
    # Bytes literal served via CODECOPY matches what we would get from an
    # inline MSTORE chain. (cf. issue #2505)
    import hashlib

    buf = b""
    seed = b"vyper#2505-roundtrip"
    while len(buf) < 256:
        seed = hashlib.sha256(seed).digest()
        buf += seed
    expected = buf[:256]
    blob_hex = expected.hex()
    code = f"""
BLOB: constant(Bytes[256]) = x"{blob_hex}"

@external
@view
def get_blob() -> Bytes[256]:
    return BLOB

@external
@view
def blob_len() -> uint256:
    return len(BLOB)
"""
    c = get_contract(code)
    assert c.get_blob() == expected
    assert c.blob_len() == 256


def test_large_bytes_constant_is_emitted_via_codecopy():
    # Verifies the CODECOPY lowering happens at the IR layer for a dense
    # literal, and confirms that (a) a small constant and (b) a sparse
    # ABI-encoded constant both stay inline (MSTORE chain is at least as
    # compact as CODECOPY for those cases, due to Vyper's push optimizer).
    import hashlib

    import vyper

    # Dense 256-byte literal: heuristic fires.
    buf = b""
    seed = b"vyper#2505-ir"
    while len(buf) < 256:
        seed = hashlib.sha256(seed).digest()
        buf += seed
    dense_hex = buf[:256].hex()
    dense_src = f"""
BLOB: constant(Bytes[256]) = x"{dense_hex}"

@external
@view
def get_blob() -> Bytes[256]:
    return BLOB
"""
    ir_text = str(vyper.compile_code(dense_src, output_formats=["ir_runtime"])["ir_runtime"])
    assert "_const_bytestring_" in ir_text

    # Small literal (16 bytes): stays inline.
    small_src = """
BLOB: constant(Bytes[16]) = x"aabbccddeeff00112233445566778899"

@external
@view
def get_blob() -> Bytes[16]:
    return BLOB
"""
    ir_text = str(vyper.compile_code(small_src, output_formats=["ir_runtime"])["ir_runtime"])
    assert "_const_bytestring_" not in ir_text

    # Sparse ABI-encoded literal (204 bytes of zero-padded fields): inline
    # MSTORE chain compresses below raw bytes, so heuristic declines.
    abi_hex = (
        "a9059cbb"
        "00000000000000000000000060bc267d1242494ca0ba2a944ec0ba223c68b0e2"
        "0000000000000000000000000000000000000000000000000000000000002710"
    ) * 3
    abi_src = f"""
BLOB: constant(Bytes[204]) = x"{abi_hex}"

@external
@view
def get_blob() -> Bytes[204]:
    return BLOB
"""
    ir_text = str(vyper.compile_code(abi_src, output_formats=["ir_runtime"])["ir_runtime"])
    assert "_const_bytestring_" not in ir_text


def test_duplicate_large_bytes_constants_are_deduped(get_contract):
    # Two identical source-level constant bytestrings that exceed the
    # content-aware lowering threshold should share a single data section
    # entry. Use a 256-byte deterministic dense blob so the heuristic fires.
    import hashlib

    import vyper

    buf = b""
    seed = b"vyper#2505-dedup"
    while len(buf) < 256:
        seed = hashlib.sha256(seed).digest()
        buf += seed
    blob_hex = buf[:256].hex()
    code = f"""
BLOB_A: constant(Bytes[256]) = x"{blob_hex}"
BLOB_B: constant(Bytes[256]) = x"{blob_hex}"

@external
@view
def get_a() -> Bytes[256]:
    return BLOB_A

@external
@view
def get_b() -> Bytes[256]:
    return BLOB_B
"""
    out = vyper.compile_code(code, output_formats=["ir_runtime"])
    ir_text = str(out["ir_runtime"])
    # Exactly one data section entry for the shared blob.
    assert ir_text.count("[data,\n    _const_bytestring_") == 1

    c = get_contract(code)
    expected = bytes.fromhex(blob_hex)
    assert c.get_a() == expected
    assert c.get_b() == expected


def test_constructor_large_bytes_constant_stays_inline(get_contract):
    # Constants used in init code cannot reach the runtime data section via
    # CODECOPY, so init-time usage must stay inline. Use the constant in the
    # constructor and ensure compilation still succeeds and the runtime
    # behavior is correct.
    import hashlib

    buf = b""
    seed = b"vyper#2505-ctor"
    while len(buf) < 256:
        seed = hashlib.sha256(seed).digest()
        buf += seed
    blob = buf[:256]
    blob_hex = blob.hex()
    code = f"""
BLOB: constant(Bytes[256]) = x"{blob_hex}"

stored: public(Bytes[256])

@deploy
def __init__():
    self.stored = BLOB

@external
@view
def get_stored() -> Bytes[256]:
    return self.stored
"""
    c = get_contract(code)
    assert c.get_stored() == blob
