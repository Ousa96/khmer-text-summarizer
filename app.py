from flask import Flask, render_template, request, jsonify
import os
import requests
import warnings
# from transformers import MBartForConditionalGeneration, MBart50Tokenizer

# ================== Config ==================
# load_dotenv()
app = Flask(__name__)

# ================== Suppress warnings ==================
warnings.filterwarnings("ignore", category=FutureWarning)

# Hugging Face model setup
HF_API_TOKEN = os.environ.get("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") # replace with your actual token
MODEL_ID = "angkor96/khmer-news-summarization"  # change if needed
HF_API_URL = f"https://api-inference.huggingface.co/models/angkor96/khmer-news-summarization" 

headers = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Accept": "application/json"
}

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

    payload = {"inputs": input_text}
    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()

        # Extract summary text from response (depends on model format)
        summary = None
        if isinstance(result, list) and len(result) > 0 and "summary_text" in result[0]:
            summary = result[0]["summary_text"]
        elif isinstance(result, list) and len(result) > 0 and "generated_text" in result[0]:
            summary = result[0]["generated_text"]
        else:
            summary = "មិនអាចសង្ខេបបានទេ។"

        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ================== Run server ==================
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

