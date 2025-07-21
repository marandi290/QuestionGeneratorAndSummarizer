import string
import re
from typing import List
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def remove_duplicates(items: list[str]) -> list[str]:
    """
    Remove semantically normalized duplicates from a list of strings.
    """
    unique_items = []
    normalized_set = set()

    for item in items:
        norm = _normalize_item(item)
        if norm not in normalized_set:
            unique_items.append(item)
            normalized_set.add(norm)

    return unique_items


def remove_distractors_duplicate_with_correct_answer(correct: str, distractors: list[str]) -> list[str]:
    """
    Remove distractors that are the same as the correct answer (normalized).
    """
    norm_correct = _normalize_item(correct)
    return [d for d in distractors if _normalize_item(d) != norm_correct]


def _get_most_distinct_from_key(key: str, items: list[str]) -> list[str]:
    """
    Placeholder: Can be used to rank distractors by distinctiveness from the key.
    """
    # TODO: Consider implementing semantic distance or cosine similarity filtering
    return items


def _get_most_distinct_from_each_other():
    """
    TODO: Calculate BLEU scores between all pairs of items.
    Remove the most similar pair until you reach the desired number.
    """
    pass


def _normalize_item(item: str) -> str:
    """
    Normalize text by lowering case, removing punctuation, articles, and extra whitespace.
    """
    def remove_articles(text: str) -> str:
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())

    def remove_punc(text: str) -> str:
        return ''.join(ch for ch in text if ch not in string.punctuation)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(item))))


def _calculate_nltk_bleu(references: list[str], hypothesis: str, bleu_n: int = 1) -> float:
    """
    Calculate BLEU score between hypothesis and references.
    """
    if not hypothesis.strip():
        return 0.0

    refs_tokenized = [word_tokenize(ref) for ref in references]
    hyp_tokenized = word_tokenize(hypothesis)

    chencherry = SmoothingFunction()
    weights = {
        1: (1, 0, 0, 0),
        2: (0.5, 0.5, 0, 0),
        3: (0.33, 0.33, 0.33, 0),
        4: (0.25, 0.25, 0.25, 0.25),
    }

    weight = weights.get(bleu_n, weights[1])
    return sentence_bleu(refs_tokenized, hyp_tokenized, weights=weight, smoothing_function=chencherry.method2)
