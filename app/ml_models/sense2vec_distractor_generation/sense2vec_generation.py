from sense2vec import Sense2Vec
from collections import OrderedDict
from typing import List
from pathlib import Path
import os

class Sense2VecDistractorGeneration:
    def __init__(self):
        model_path = Path("app/ml_models/sense2vec_distractor_generation/data/s2v_old")
        if not model_path.exists():
            raise FileNotFoundError(f"Sense2Vec model not found at: {model_path}")

        self.s2v = Sense2Vec().from_disk(str(model_path))

    def generate(self, answer: str, desired_count: int) -> list[str]:
        distractors = []
        normalized_answer = answer.lower().replace(" ", "_")

        sense = self.s2v.get_best_sense(normalized_answer)
        if not sense:
            return []

        most_similar = self.s2v.most_similar(sense, n=desired_count)

        for phrase, _ in most_similar:
            clean_phrase = phrase.split("|")[0].replace("_", " ").lower()

            if clean_phrase != normalized_answer:
                distractors.append(clean_phrase.capitalize())

        # Remove duplicates while preserving order
        return list(OrderedDict.fromkeys(distractors))
