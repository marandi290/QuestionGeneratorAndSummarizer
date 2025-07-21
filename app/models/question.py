from dataclasses import dataclass, field

@dataclass
class Question:
    answerText: str
    questionText: str = ''
    distractors: list[str] = field(default_factory=list)
