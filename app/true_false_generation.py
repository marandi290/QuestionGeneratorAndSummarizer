import nltk
import random
import spacy
from nltk.tokenize import sent_tokenize

# Download required NLTK data
nltk.download('punkt')

# Load spaCy English model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


def get_sentences(text):
    """Split text into individual sentences."""
    return sent_tokenize(text)


def negate_sentence(sentence):
    """
    Negate a sentence by inserting 'not' after the auxiliary verb or main verb.
    Simple heuristic-based approach.
    """
    doc = nlp(sentence)
    new_tokens = []
    negated = False

    for token in doc:
        # Insert 'not' after first auxiliary or modal verb
        if not negated and token.pos_ in ['AUX', 'VERB']:
            if token.text.lower() not in ['not', 'never']:
                new_tokens.append(token.text)
                new_tokens.append('not')
                negated = True
            else:
                new_tokens.append(token.text)
        else:
            new_tokens.append(token.text)
    return ' '.join(new_tokens)


def generate_true_false_questions(text, num_questions=5):
    sentences = get_sentences(text)
    random.shuffle(sentences)

    tf_questions = []

    for i, sentence in enumerate(sentences[:num_questions]):
        if random.random() > 0.5:
            # Create True statement
            tf_questions.append({"question": sentence, "answer": "True"})
        else:
            # Create False statement by negating
            negated = negate_sentence(sentence)
            tf_questions.append({"question": negated, "answer": "False"})

    return tf_questions


if __name__ == "__main__":
    sample_text = """
    The sun is the star at the center of the Solar System. It is a nearly perfect sphere of hot plasma, 
    heated to incandescence by nuclear fusion reactions in its core. Earth orbits the Sun at an average distance 
    of about 93 million miles (150 million kilometers). The Sun provides the energy necessary for life on Earth.
    """

    print("Generated True/False Questions:\n")
    questions = generate_true_false_questions(sample_text, num_questions=5)
    for idx, qa in enumerate(questions, 1):
        print(f"Q{idx}: {qa['question']}")
        print(f"Answer: {qa['answer']}\n")
