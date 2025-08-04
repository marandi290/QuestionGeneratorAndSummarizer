from transformers import BartTokenizer, BartForConditionalGeneration
import torch

tokenizer = BartTokenizer.from_pretrained("app/ml_models/summary_generation/bart_model", 
                                          bos_token="<s>", eos_token="</s>", unk_token="<unk>", pad_token="<pad>")
model = BartForConditionalGeneration.from_pretrained("app/ml_models/summary_generation/bart_model")

def chunk_text(text, max_tokens=1024):
    tokens = tokenizer.encode(text, truncation=False)
    chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

def summarize_long_text(text):
    chunks = chunk_text(text)
    summaries = []

    for chunk in chunks:
        inputs = tokenizer([chunk], max_length=1024, return_tensors='pt', truncation=True)
        summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=150, min_length=30, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    # Optional: summarize the combined summaries
    combined_summary = " ".join(summaries)
    if len(tokenizer.encode(combined_summary)) > 1024:
        return summarize_long_text(combined_summary)  # Recursive summarization
    return combined_summary

# 🧪 Example usage
if __name__ == "__main__":
    text = """
    Climate change refers to long-term shifts in temperatures and weather patterns. These shifts may be natural, 
    such as through variations in the solar cycle. But since the 1800s, human activities have been the main driver 
    of climate change, primarily due to burning fossil fuels like coal, oil, and gas. Burning these materials releases 
    what are called greenhouse gases into Earth’s atmosphere. These emissions act like a blanket wrapped around the Earth, 
    trapping the sun’s heat and raising temperatures.
    
    Examples of greenhouse gases include carbon dioxide and methane. These come from using gasoline for driving a car or coal 
    for heating a building, for example. Clearing land and forests can also release carbon dioxide. Landfills for garbage are 
    a major source of methane emissions. Energy, industry, transport, buildings, agriculture and land use are among the main 
    emitters. And climate change can affect our health, ability to grow food, housing, safety and work. Some of us are already 
    more vulnerable to climate impacts, such as people living in small island nations and other developing countries. Conditions 
    like sea-level rise and saltwater intrusion have advanced to the point where whole communities have had to relocate, and 
    prolonged droughts are putting people at risk of famine.
    """
    result = summarize_long_text(text)
    print("Summary:\n", result)
