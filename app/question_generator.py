from transformers import pipeline

qg_pipeline = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")

def generate_mcqs(text, num_questions=10):
    sentences = text.split(". ")
    questions = []

    for sentence in sentences:
        if len(questions) >= num_questions:
            break
        prompt = "generate question: " + sentence.strip()
        try:
            result = qg_pipeline(prompt, max_length=64, do_sample=False)
            questions.append(result[0]['generated_text'])
        except Exception:
            continue

    return questions
