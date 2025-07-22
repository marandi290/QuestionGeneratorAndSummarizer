import nltk
import random
import spacy
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

# Load spaCy English model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")


def get_sentences(text):
    return sent_tokenize(text)


def negate_sentence(sentence):
    """Insert 'not' after the first verb (AUX/VERB) as a simple negation."""
    doc = nlp(sentence)
    new_tokens = []
    negated = False

    for token in doc:
        if not negated and token.pos_ in ['AUX', 'VERB'] and token.text.lower() not in ['not', 'never']:
            new_tokens.append(token.text)
            new_tokens.append('not')
            negated = True
        else:
            new_tokens.append(token.text)
    return ' '.join(new_tokens)


def mask_sentence(sentence):
    """
    Mask a key word (Named Entity or NOUN) in the sentence.
    Return masked sentence, correct answer, and a fake (random) answer.
    """
    doc = nlp(sentence)
    # Prioritize named entities
    targets = [ent for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "DATE", "EVENT", "WORK_OF_ART"]]
    if not targets:
        # Fallback to nouns
        targets = [token for token in doc if token.pos_ == "NOUN" and token.is_alpha]

    if not targets:
        return None, None, None

    target = random.choice(targets)
    correct_answer = target.text

    # Replace with a blank
    masked = sentence.replace(correct_answer, "_____")

    # Generate fake answer (just shuffling some nouns from a generic pool)
    fake_words = ["India", "Microsoft", "Einstein", "Mount Everest", "1999", "Python"]
    fake_answer = random.choice([word for word in fake_words if word != correct_answer])

    return masked, correct_answer, fake_answer


def generate_true_false_questions(text, num_questions=5, method="both"):
    sentences = get_sentences(text)
    random.shuffle(sentences)

    tf_questions = []

    for sentence in sentences:
        if len(tf_questions) >= num_questions:
            break

        if method == "negation" or (method == "both" and random.random() < 0.5):
            # Use negation
            if len(sentence.split()) < 4:
                continue
            if random.random() > 0.5:
                tf_questions.append({"question": sentence, "answer": "True"})
            else:
                tf_questions.append({"question": negate_sentence(sentence), "answer": "False"})
        else:
            # Use masking
            masked, correct, fake = mask_sentence(sentence)
            if masked and correct and fake:
                if random.random() > 0.5:
                    tf_questions.append({"question": f"{masked} (Answer: {correct})", "answer": "True"})
                else:
                    tf_questions.append({"question": f"{masked} (Answer: {fake})", "answer": "False"})

    return tf_questions


if __name__ == "__main__":
    sample_text = """
    Isaac Newton was an English mathematician, physicist, and astronomer who is widely recognized as one of the most influential scientists of all time. 
He formulated the laws of motion and universal gravitation. 
Newton was born in 1643 in Woolsthorpe, England. 
He was elected President of the Royal Society in 1703 and was knighted by Queen Anne in 1705. 
The Royal Society is one of the oldest scientific institutions in the world. 
Cambridge University was where Newton studied and later taught. 
He published his book "Philosophiæ Naturalis Principia Mathematica" in 1687, which laid the foundations of classical mechanics.
    """

    print("Generated True/False Questions:\n")
    questions = generate_true_false_questions(sample_text, num_questions=7, method="both")

    for idx, qa in enumerate(questions, 1):
        print(f"Q{idx}: {qa['question']}")
        print(f"Answer: {qa['answer']}\n")
