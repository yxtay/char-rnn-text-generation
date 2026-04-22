# -*- coding: utf-8 -*-
"""Build a character alphabet from NLTK WordNet lemma strings (no gloss text)."""
import json
import os

_CACHE_FILENAME = "wordnet_char_inventory.json"


def _data_path(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", name)


def ensure_nltk_wordnet():
    import nltk

    try:
        from nltk.corpus import wordnet as wn
        wn.synsets("test")
    except LookupError:
        nltk.download("wordnet", quiet=True)


def _chars_from_wordnet():
    from nltk.corpus import wordnet as wn

    chars = set()
    for syn in wn.all_synsets():
        for lem in syn.lemmas():
            for ch in lem.name().replace("_", " "):
                chars.add(ch)
    return chars


def lemma_character_inventory():
    """
    Characters appearing in WordNet lemma names. Cached under
    ``data/wordnet_char_inventory.json``. Adds digits and common whitespace.
    """
    cache_path = _data_path(_CACHE_FILENAME)
    if os.path.isfile(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            payload = json.load(f)
        return set(payload["chars"])

    ensure_nltk_wordnet()
    chars = _chars_from_wordnet()
    chars.update(" \t\n0123456789")
    chars.update("-'")
    chars.discard("\r")
    chars.discard("\x0b")
    chars.discard("\x0c")
    ordered = sorted(chars)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"chars": ordered, "source": "nltk.corpus.wordnet lemmas"}, f, ensure_ascii=False)
    return set(ordered)
