from airc_tools.limits import cap_head, clip, head_tail


def test_head_tail_under_limit():
    assert head_tail("small", 100) == "small"


def test_head_tail_elides_middle():
    text = "".join(f"line{i}\n" for i in range(10000))
    out = head_tail(text, 1000)
    assert "elided" in out
    assert out.startswith("line0\n")
    assert out.rstrip().endswith("line9999")  # tail preserved
    assert len(out) < 1500


def test_cap_head():
    kept, dropped = cap_head("abcdef", 3)
    assert kept == "abc"
    assert dropped == 3


def test_cap_head_under():
    kept, dropped = cap_head("ab", 10)
    assert kept == "ab"
    assert dropped == 0


def test_clip():
    out = clip("x" * 5000, 100)
    assert "more bytes" in out
    assert len(out) < 200
