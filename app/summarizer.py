from transformers import pipeline

summarizer = pipeline("summarization", model="t5-base", tokenizer="t5-base")

def generate_summary(text, max_length=150, min_length=40):
    summary = summarizer(text[:1000], max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']