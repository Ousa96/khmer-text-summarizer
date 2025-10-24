from flask import Flask, render_template, request, jsonify
import warnings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# ================== Config ==================
app = Flask(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

# ================== Device ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ================== Load Model ==================
MODEL_NAME = "angkor96/khmer-news-summarization"
print("Loading tokenizer and model... this may take some time.")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
model.eval()  # set model to evaluation mode
print("Model loaded successfully!")

# ================== Summarization Function ==================
def text_summarize(text, max_length=150):
    try:
        input_text = f"summarize: {text}"
        # Tokenize and truncate long inputs
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        # Move inputs to device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Disable gradient calculation for inference
        with torch.no_grad():
            summary_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
                num_beams=5,
                length_penalty=2.0,
                early_stopping=True
            )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary

    except Exception as e:
        print("Error during summarization:", e)
        return "មិនអាចសង្ខេបបានទេ។"

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

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    input_text = data.get("text", "").strip()

    if not input_text:
        return jsonify({"summary": "សូមបញ្ចូលអត្ថបទសិន។"}), 400

    summary = text_summarize(input_text)
    return jsonify({"summary": summary})

# ================== Run Server ==================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
