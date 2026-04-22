# -*- coding: utf-8 -*-
"""
WordNet-backed lexicon utilities.

This project is *character-level*: labels are characters, not words. WordNet is
used only to derive which *characters* appear in English lemma strings (glosses
and synset definitions are not used). That yields a smaller softmax than the
default ``string.printable`` inventory, which can speed training slightly.
"""
import json
import os

_CACHE_FILENAME = "wordnet_char_inventory.json"


def _data_path(name):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", name)


def ensure_nltk_wordnet():
    """Download the WordNet corpus via NLTK if missing."""
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
    Return a set of characters to use as the training alphabet.

    Built from every character that appears in any WordNet lemma name (with
    underscores treated as spaces). Digits, common punctuation used in lemmas,
    and ASCII whitespace are always included. Results are cached under
    ``data/wordnet_char_inventory.json`` after the first successful NLTK scan.
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


def wordnet_lemma_words():
    """
    Iterator over distinct WordNet lemma strings (underscores as spaces).
    Useful if you later add word-level models; not used by the char-RNN trainers.
    """
    ensure_nltk_wordnet()
    from nltk.corpus import wordnet as wn

    seen = set()
    for syn in wn.all_synsets():
        for lem in syn.lemmas():
            w = lem.name().replace("_", " ")
            if w not in seen:
                seen.add(w)
                yield w
