import os
from app.utils.video_processing import video_to_text
from flask import Flask, render_template, request
from flask_cors import CORS
from app.mcq_generator import MCQGenerator
from app.true_false_generation import generate_true_false_questions
from app.utils.extraction import extract_and_clean_uploaded_file
from app.ml_models.summary_generation.summarizer import generate_summary_from_long_text
from app.ml_models.summary_generation.bart_summarizer import summarize_long_text as bart_generate_summary
from werkzeug.utils import secure_filename
from app.ml_models.descriptive_question_generation.descriptive_long_qg import generate_questions_from_long_text as generate_long_answer_questions
from app.ml_models.descriptive_question_generation.descriptive_short_qg import generate_questions_from_long_text as generate_short_answer_questions

app = Flask(__name__)
CORS(app)

# Folder to save uploaded video files
UPLOAD_FOLDER = os.path.join('app', 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def home():
    results = {
        "summary": "",
        "mcq": "",
        "true_false": "",
        "short_answer": "",
        "long_answer": "",
        "fill_blank": "",
        "match_following": ""
    }
    typed_text = request.form.get("text", "")
    input_text = ""

    if request.method == "POST":
        task = request.form.get("task")  # Safely get task

        #input_text = ""
        uploaded_file = request.files.get("file")
        uploaded_video = request.files.get("video")
        #typed_text = request.form.get("text")

        # File upload
        if uploaded_file:
            input_text = extract_and_clean_uploaded_file(uploaded_file)

        # Video upload
        elif uploaded_video:
            filename = secure_filename(uploaded_video.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_video.save(video_path)

            # Convert video to text
            input_text = video_to_text(video_path)
        elif typed_text:
            input_text = typed_text.strip()
            
        if not input_text:
            results[task] = "❗ No input text provided."
        else:
            if task == "mcq":
                mcq_generator = MCQGenerator(is_verbose=True)
                questions = mcq_generator.generate_mcq_questions(input_text, desired_count=5)
                results["mcq"] = '\n\n'.join([
                    f"Q: {q.questionText}\nAns: {q.answerText}\nOptions: {', '.join(q.distractors)}"
                    for q in questions
                ])
            elif task == "true_false":
                questions = generate_true_false_questions(input_text, num_questions=5, method="both")
                results["true_false"] = '\n\n'.join([
                    f"Q: {q['question']}\nAns: {q['answer']}" for q in questions
                ])
            elif task == "summary":
                summaries = bart_generate_summary(input_text)
                results["summary"] = summaries if summaries else "❗ No summary generated."
            elif task == "short_answer":
                questions = generate_short_answer_questions(input_text)
                results["short_answer"] = '\n\n'.join([
                    f"{q}" for q in questions
                ])
            elif task == "long_answer":
                questions = generate_long_answer_questions(input_text)
                results["long_answer"] = '\n\n'.join([
                    f"{q}" for q in questions
                ])
            elif task == "fill_blank":
                results["fill_blank"] = "🔧 Fill in the blanks generation not implemented yet."
            elif task == "match_following":
                results["match_following"] = "🔧 Match the following generation not implemented yet."

    return render_template(
        "index.html",
        results=results,
        text=input_text
    )


if __name__ == '__main__':
    app.run(debug=True)
