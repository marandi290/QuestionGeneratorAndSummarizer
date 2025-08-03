from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load pre-trained T5 model and tokenizer
model_name = "t5-small"  # You can use "t5-base" or "t5-large" for better performance
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def summarize_text(text: str, max_input_length=512, max_output_length=150):
    # Prepend "summarize:" as T5 expects this prefix for summarization tasks
    input_text = "summarize: " + text.strip().replace("\n", " ")

    # Tokenize input text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=max_input_length, truncation=True)

    # Generate summary
    summary_ids = model.generate(
        inputs,
        max_length=max_output_length,
        num_beams=4,
        early_stopping=True,
        length_penalty=2.0
    )

    # Decode and return summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
def split_text_into_chunks(text, max_tokens=450):
    # Approximate 1 token ≈ 0.75 words
    # So 450 tokens ≈ 600 words max
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len((current_chunk + sentence).split()) <= max_tokens:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def generate_summary_from_long_text(long_text):
    chunks = split_text_into_chunks(long_text)
    all_summary = []

    for i, chunk in enumerate(chunks):
        #print(f"\n--- Generating questions for chunk {i+1} ---")
        try:
            summary = summarize_text(chunk)
            #print(f"Chunk {i+1} questions: {questions}")
            all_summary.append(summary)
        except Exception as e:
            print(f"Error generating for chunk {i+1}: {e}")

    return all_summary