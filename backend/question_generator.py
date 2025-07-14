from transformers import pipeline

qg_pipeline = pipeline("text2text-generation", model="valhalla/t5-base-qg-hl")

def generate_mcqs(text):
    input_text = "generate questions: " + text[:1000]
    results = qg_pipeline(input_text, max_length=64, do_sample=False)
    return [results[0]["generated_text"]]