"""Tests for the text cleaner."""

from __future__ import annotations

import pytest

from src.preprocessing import (
    CLEANING_STEPS,
    TextCleaner,
    build_char_vectorizer,
    build_union_vectorizer,
    build_word_vectorizer,
    cleaning_ablation_cleaners,
)


@pytest.fixture
def cleaner():
    return TextCleaner()


def test_lowercases_and_collapses_whitespace(cleaner):
    assert cleaner.clean("  HELLO   World \n\n ") == "hello world"


def test_removes_urls_and_ip_addresses(cleaner):
    cleaned = cleaner.clean("see https://example.com/page and 192.168.0.1")
    assert "example.com" not in cleaned
    assert "192" not in cleaned
    assert "url" in cleaned and "ipaddress" in cleaned


def test_strips_wiki_signature(cleaner):
    assert "user" not in cleaner.clean("nonsense [[User:Someone|Someone]] talk")


def test_expands_contractions_keeping_negation(cleaner):
    assert "not" in cleaner.clean("you can't be serious")


def test_normalises_leet_obfuscation(cleaner):
    assert cleaner.clean("st@pid") == "stapid"
    assert cleaner.clean("id!ot") == "idiot"
    assert cleaner.clean("b1tch") == "bitch"


def test_leet_normalisation_leaves_plain_numbers_alone(cleaner):
    """Rewriting every digit would turn the year 2013 into "2oie"."""
    sentence = "the 2013 revision of section 4"
    assert cleaner.clean(sentence) == sentence


def test_collapses_repeated_characters(cleaner):
    assert cleaner.clean("stuuuuupid") == "stuupid"


def test_handles_empty_and_non_string_input(cleaner):
    assert cleaner.clean("") == ""
    assert cleaner.clean("   ") == ""
    assert cleaner.clean(12345) == "12345"
    assert cleaner.clean(None) == "none"


def test_transform_returns_one_output_per_input(cleaner):
    documents = ["First COMMENT", "second one", ""]
    assert len(cleaner.transform(documents)) == len(documents)


def test_flags_can_be_switched_off():
    keep_everything = TextCleaner(
        lowercase=False, strip_punctuation=False, normalise_leet=False
    )
    assert keep_everything.clean("HELLO, World!") == "HELLO, World!"


def test_cleaning_ablation_cleaners_covers_every_flag():
    cleaners = cleaning_ablation_cleaners()
    expected = {"full", "none"} | {f"no_{step}" for step in CLEANING_STEPS}
    assert set(cleaners) == expected

    for step in CLEANING_STEPS:
        assert getattr(cleaners["full"], step) is True
        assert getattr(cleaners["none"], step) is False
        variant = cleaners[f"no_{step}"]
        assert getattr(variant, step) is False
        for other in CLEANING_STEPS:
            if other != step:
                assert getattr(variant, other) is True


def test_each_ablation_flag_isolates_its_normalisation_step():
    cleaners = cleaning_ablation_cleaners()
    assert cleaners["no_lowercase"].clean("HELLO") == "hello".upper()
    assert "example" in cleaners["no_strip_urls"].clean("https://example.com")
    assert "user" in cleaners["no_strip_wiki_markup"].clean("[[User:Alice]]")
    assert "can't" in cleaners["no_expand_contractions"].clean("you can't")
    assert "b1tch" in cleaners["no_normalise_leet"].clean("b1tch")
    assert "stuuuuupid" in cleaners["no_collapse_repeats"].clean("stuuuuupid")
    assert "?" in cleaners["no_strip_punctuation"].clean("why?")


def test_vectorizer_produces_word_and_char_features():
    documents = [
        "you are an idiot",
        "you are an idiot and a fool",
        "thanks for the helpful edit",
        "thanks for the edit",
    ]
    word_matrix = build_word_vectorizer(max_features=100, min_df=1).fit_transform(
        documents
    )
    char_matrix = build_char_vectorizer(max_features=100, min_df=1).fit_transform(
        documents
    )
    matrix = build_union_vectorizer(word_features=100, char_features=100).fit_transform(
        documents
    )
    assert word_matrix.shape[0] == len(documents)
    assert char_matrix.shape[0] == len(documents)
    assert matrix.shape[0] == len(documents)
    assert matrix.shape[1] > word_matrix.shape[1]

