from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import torch
import warnings
# from transformers import MBartForConditionalGeneration, MBart50Tokenizer

# ================== Config ==================
load_dotenv()
app = Flask(__name__)

# ================== Suppress warnings ==================
warnings.filterwarnings("ignore", category=FutureWarning)

# ================== Model ==================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model_name = "facebook/mbart-large-50"
# tokenizer = MBart50Tokenizer.from_pretrained(model_name, src_lang="km_KH", tgt_lang="km_KH")

# model_path = r"D:\ITC File\Year4\Internship document\Text_Generation\Web_database\Project\model\best_model.pth"
# model = MBartForConditionalGeneration.from_pretrained(model_name)
# model.load_state_dict(torch.load(model_path, map_location=device))
# model.to(device)
# model.eval()
# print(f"Model loaded successfully on {device}.")

# # ================== Summarization ==================
# def summarize_text(input_text):
#     if not input_text.strip():
#         return ""
#     inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1012)
#     inputs = {k: v.to(device) for k, v in inputs.items()}

#     with torch.no_grad():
#         summary_ids = model.generate(inputs["input_ids"], max_length=350, num_beams=4, early_stopping=True)
#         summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

#     return summary.strip() or "មិនអាចសង្ខេបបានទេ។"

# ================== Routes ==================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/how_to_use')
def how_to_use():
    return render_template('how_to_use.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

# @app.route('/summarize', methods=['POST'])
# def summarize():
#     data = request.get_json()
#     input_text = data.get("text", "").strip()

#     if not input_text:
#         return jsonify({"summary": "សូមបញ្ចូលអត្ថបទសិន។"}), 400

#     output_text = summarize_text(input_text)
#     return jsonify({"summary": output_text})

# ================== Run server ==================
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)

