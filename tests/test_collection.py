"""Extractor tests.

Every HTML shape here was observed in the real collection.
"""

import pytest

from vocab_gen.collection import terms_in_html


@pytest.mark.parametrize(
    "html, expected",
    [
        # Both nesting orders occur in the collection.
        ("The <i><u>inexorable</u></i> path.", ["inexorable"]),
        ("A <u><i>hilt</i></u> of glass.", ["hilt"]),
        # Two separate words on one note must not merge into one term.
        (
            "the fawning <u><i>supplicants</i></u> and the hateful <i><u>pomp</u></i>",
            ["supplicants", "pomp"],
        ),
        # A genuine phrase sits in one element and must stay whole.
        ("It was <i><u>sui generis</u></i>.", ["sui generis"]),
        ("Grind it with a <i><u>pumice stone</u></i>.", ["pumice stone"]),
        # Underline/italic via style attributes rather than tags.
        (
            '<span style="text-decoration: underline"><span style="font-style: italic">'
            "bucolic</span></span>",
            ["bucolic"],
        ),
        # Mixed: a styled tag outside, a style attribute inside.
        ('<u><span style="font-style:italic">nexus</span></u>', ["nexus"]),
        # Anki's non-breaking spaces and whitespace runs.
        ("a <i><u>cinéma&nbsp;&nbsp;vérité</u></i> feel", ["cinéma vérité"]),
        # Accents and hyphens survive.
        ("his <i><u>pied-à-terre</u></i>", ["pied-à-terre"]),
        # Italic-only emphasis on non-vocab cards must NOT be extracted.
        ("What does <i>brachii</i> mean?", []),
        ("the number of <i>significant digits</i>", []),
        # Underline-only is likewise not enough.
        ("a <u>heading</u>", []),
        # Bold adds nothing on its own.
        ("<b>bold</b> and <i>italic</i>", []),
        # Void tags inside a styled run must not terminate the tag stack.
        ("<i><u>death<br>knell</u></i>", ["death knell"]),
        # Stray punctuation Anki leaves inside the span.
        ('<i><u>reprieve,</u></i>', ["reprieve"]),
        ('<i><u>“glade”</u></i>', ["glade"]),
        # Empty / no markup.
        ("", []),
        ("plain sentence", []),
    ],
)
def test_terms_in_html(html, expected):
    assert terms_in_html(html) == expected


def test_unclosed_tags_do_not_swallow_the_rest():
    """Anki HTML is not always well-formed; a stray <i> must not style everything."""
    out = terms_in_html("<i><u>ford</u> and then unstyled trailing text")
    assert out == ["ford"]


def test_nested_same_tag_closes_innermost_only():
    assert terms_in_html("<i><u><i>butte</i></u></i>") == ["butte"]
