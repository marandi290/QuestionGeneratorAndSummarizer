from typing import List
from nltk.tokenize import sent_tokenize
import toolz
import time

from app.modules.duplicate_removal import remove_distractors_duplicate_with_correct_answer, remove_duplicates
from app.modules.text_cleaning import clean_text
from app.ml_models.answer_generation.answer_generator import AnswerGenerator
from app.ml_models.distractor_generation.distractor_generator import DistractorGenerator
from app.ml_models.question_generation.question_generator import QuestionGenerator
from app.ml_models.sense2vec_distractor_generation.sense2vec_generation import Sense2VecDistractorGeneration
from app.models.question import Question


class MCQGenerator:
    def __init__(self, is_verbose: bool = False):
        """
        Initialize and load the ML models used for MCQ generation.
        """
        start_time = time.perf_counter()
        print("Loading ML Models...")

        # AnswerGenerator is not used in current flow
        # self.answer_generator = AnswerGenerator()
        # if is_verbose:
        #     print(f"Loaded AnswerGenerator in {round(time.perf_counter() - start_time, 2)} seconds.")

        self.question_generator = QuestionGenerator()
        if is_verbose:
            print(f"Loaded QuestionGenerator in {round(time.perf_counter() - start_time, 2)} seconds.")

        self.distractor_generator = DistractorGenerator()
        if is_verbose:
            print(f"Loaded DistractorGenerator in {round(time.perf_counter() - start_time, 2)} seconds.")

        self.sense2vec_distractor_generator = Sense2VecDistractorGeneration()
        if is_verbose:
            print(f"Loaded Sense2VecDistractorGenerator in {round(time.perf_counter() - start_time, 2)} seconds.")

    def generate_mcq_questions(self, context: str, desired_count: int) -> List[Question]:
        """
        Main function to generate MCQ questions from input context.

        Args:
            context (str): The input paragraph or text.
            desired_count (int): Desired number of MCQs.

        Returns:
            List[Question]: List of MCQ questions with answers and distractors.
        """
        cleaned_text = clean_text(context)
        questions = self._generate_question_answer_pairs(cleaned_text, desired_count)
        questions = self._generate_distractors(cleaned_text, questions)

        for question in questions:
            print("-------------------")
            print(f"Answer: {question.answerText}")
            print(f"Question: {question.questionText}")
            print(f"Distractors: {question.distractors}")

        return questions

    def _generate_question_answer_pairs(self, context: str, desired_count: int) -> List[Question]:
        """
        Generate question-answer pairs from the cleaned context.

        Args:
            context (str): Cleaned input text.
            desired_count (int): Number of question-answer pairs.

        Returns:
            List[Question]: List of `Question` objects with answers and questions.
        """
        context_splits = self._split_context(context, desired_count)
        questions = []

        for split in context_splits:
            answer, question = self.question_generator.generate_qna(split)
            questions.append(Question(answer.capitalize(), question))

        return list(toolz.unique(questions, key=lambda q: q.answerText))

    def _generate_distractors(self, context: str, questions: List[Question]) -> List[Question]:
        """
        Generate distractors for each question.

        Args:
            context (str): Original context.
            questions (List[Question]): List of questions with answers.

        Returns:
            List[Question]: Questions with filled `distractors` field.
        """
        for question in questions:
            t5_distractors = self.distractor_generator.generate(
                generate_count=5,
                correct=question.answerText,
                question=question.questionText,
                context=context
            )

            if len(t5_distractors) < 3:
                s2v_distractors = self.sense2vec_distractor_generator.generate(
                    word=question.answerText,
                    top_n=3
                )
                distractors = t5_distractors + s2v_distractors
            else:
                distractors = t5_distractors

            distractors = remove_duplicates(distractors)
            distractors = remove_distractors_duplicate_with_correct_answer(question.answerText, distractors)

            # TODO: Filter distractors by BLEU score similarity if needed
            question.distractors = distractors

        return questions

    def _split_context(self, context: str, desired_count: int) -> List[str]:
        """
        Split the context into multiple parts based on the desired number of MCQs.

        Args:
            context (str): Input paragraph.
            desired_count (int): Number of desired splits.

        Returns:
            List[str]: List of sub-contexts.
        """
        sents = sent_tokenize(context)
        sent_ratio = len(sents) / desired_count

        if sent_ratio < 1:
            return sents

        take_count = int(sent_ratio + 1)
        context_splits = []

        for i in range(0, len(sents), take_count - 1):
            chunk = ' '.join(sents[i:i + take_count])
            context_splits.append(chunk)

        return context_splits

    # Unused methods (if needed, you can uncomment and use them later)
    # def _generate_answers(self, context: str, desired_count: int) -> List[Question]:
    #     answers = self._generate_multiple_answers_according_to_desired_count(context, desired_count)
    #     unique_answers = remove_duplicates(answers)
    #     return [Question(answer) for answer in unique_answers]

    # def _generate_questions(self, context: str, questions: List[Question]) -> List[Question]:
    #     for question in questions:
    #         question.questionText = self.question_generator.generate(question.answerText, context)
    #     return questions

    # def _generate_answer_for_each_sentence(self, context: str) -> List[str]:
    #     sents = sent_tokenize(context)
    #     return [self.answer_generator.generate(sent, 1)[0] for sent in sents]
