"""Tests for ``skynet_matrix.markdown.to_matrix_html``.

The renderer is the single source of truth for converting LLM /
bot replies into the HTML subset Element accepts. We assert on
structural tags (``<table>``, ``<h3>``, ``<pre>``, …) rather than
exact whitespace so a mistune patch release doesn't break us.
"""

from __future__ import annotations

from skynet_matrix import to_matrix_html


def test_empty_input_returns_empty():
    assert to_matrix_html("") == ""
    assert to_matrix_html(None) == ""  # type: ignore[arg-type]


def test_header_renders_as_h_tag():
    out = to_matrix_html("### Рекомендації")
    assert "<h3>" in out
    assert "Рекомендації" in out


def test_gfm_table_renders_as_table_tag():
    src = "| A | B |\n|---|---|\n| 1 | 2 |\n"
    out = to_matrix_html(src)
    assert "<table>" in out
    assert "<thead>" in out
    assert "<tbody>" in out
    assert "<th>A</th>" in out.replace(" ", "").replace("\n", "")
    assert "<td>1</td>" in out.replace(" ", "").replace("\n", "")


def test_unordered_list():
    out = to_matrix_html("- one\n- two\n")
    assert "<ul>" in out
    assert "<li>one</li>" in out


def test_ordered_list():
    out = to_matrix_html("1. first\n2. second\n")
    assert "<ol>" in out
    assert "<li>first</li>" in out


def test_fenced_code_block():
    src = "```python\nprint('hi')\n```\n"
    out = to_matrix_html(src)
    assert "<pre>" in out
    assert "<code" in out
    # Content escaped (HTML safe)
    assert "&#39;hi&#39;" in out or "&apos;hi&apos;" in out or "'hi'" in out


def test_inline_code_and_bold_and_emphasis():
    out = to_matrix_html("**bold** _em_ `code`")
    assert "<strong>bold</strong>" in out
    assert "<em>em</em>" in out
    assert "<code>code</code>" in out


def test_link_renders_as_anchor():
    out = to_matrix_html("[label](https://example.com)")
    assert '<a href="https://example.com">label</a>' in out


def test_blockquote():
    out = to_matrix_html("> важливо")
    assert "<blockquote>" in out
    assert "важливо" in out


def test_html_in_input_is_escaped():
    """Raw HTML in the markdown source must NOT pass through — Element
    sanitises server-side, but escaping at our layer means the literal
    text shows up instead of being silently stripped."""
    out = to_matrix_html("hello <script>alert(1)</script>")
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_real_world_binance_reply():
    """Regression sample from the user's screenshot: a header + table +
    list + fenced code in one reply must all render to HTML, not pass
    through as raw markdown text."""
    src = (
        "### Рекомендація\n"
        "\n"
        "| Куди | Коли | Переваги |\n"
        "|------|------|----------|\n"
        "| Spot | DCA  | контроль |\n"
        "\n"
        "1. Спочатку — Spot.\n"
        "2. За потребою — Funding.\n"
        "\n"
        "```bash\n"
        "curl -X POST https://api.binance.com\n"
        "```\n"
    )
    out = to_matrix_html(src)
    assert "<h3>" in out
    assert "<table>" in out
    assert "<ol>" in out
    assert "<pre>" in out
    # Pipes from raw markdown table must NOT survive in the output
    assert "| Spot |" not in out
    # Header markers must NOT survive
    assert "### " not in out
