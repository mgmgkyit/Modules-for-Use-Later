from flask import Flask, request, render_template, jsonify, send_file
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import pickle
import datetime
import csv
import os
import numpy as np
from fpdf import FPDF
from io import BytesIO

# -----------------------------
# FLASK SETUP
# -----------------------------
app = Flask(__name__)
app.secret_key = "secret-key"

# Global log store for session
chat_log_data = []

# -----------------------------
# DEVICE SETUP
# -----------------------------
os.chdir(os.path.dirname(os.path.abspath(__file__)))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# LOAD MAIN CHATBOT MODEL
# -----------------------------
ADAPTER_PATH = 'model/gpt-neo-lora-checkpoint-final'
peft_config = PeftConfig.from_pretrained(ADAPTER_PATH)
base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval().to(device)

# -----------------------------
# LOAD INTENT CLASSIFIER
# -----------------------------
with open("model/intent_classifier.pkl", "rb") as f:
    intent_vectorizer, intent_model = pickle.load(f)

def is_in_domain_input(user_input, confidence_threshold=0.3):
    vec = intent_vectorizer.transform([user_input])
    probs = intent_model.predict_proba(vec)
    max_prob = np.max(probs)
    if max_prob >= confidence_threshold:
        predicted_intent = intent_model.classes_[np.argmax(probs)]
        return True, predicted_intent, max_prob
    else:
        return False, None, max_prob

# -----------------------------
# CHATBOT RESPONSE
# -----------------------------
def chat(user_input, temperature=0.7):
    input_text = f"### Prompt:\n{user_input}\n### Response:\n"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )
    decoded = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    return decoded.split("### Response:")[-1].strip() if "### Response:" in decoded else decoded.strip()

# -----------------------------
# CSV LOGGING SETUP
# -----------------------------
LOG_FILE = "case_logs.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['timestamp', 'user_input', 'response', 'confidence', 'status'])

# -----------------------------
# MAIN CHAT ROUTE
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    global chat_log_data
    response = ""
    user_input = ""
    confidence = ""
    status = "Accepted"

    if request.method == "POST":
        user_input = request.form["user_input"]
        is_in, intent, confidence = is_in_domain_input(user_input)

        if not is_in:
            response = f"Sorry, I can only assist with IT-related queries. (confidence: {confidence:.2f})"
            status = "Rejected"
        else:
            response = chat(user_input)
            if "i don't know" in response.lower():
                response += "\n\nCase escalated due to unclear answer."
                status = "Escalated"

        # Log to CSV
        with open(LOG_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.datetime.now(), user_input, response, f"{confidence:.2f}", status
            ])

        # Store in global session list
        chat_log_data.append({
            "question": user_input,
            "response": response
        })

    return render_template("index.html", response=response, request=request)

# -----------------------------
# DOWNLOAD PDF
# -----------------------------
@app.route("/download_pdf")
def download_pdf():
    global chat_log_data
    if not chat_log_data:
        return "No session history found.", 400

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for i, qa in enumerate(chat_log_data, start=1):
        pdf.multi_cell(0, 10, f"{i}. Q: {qa['question']}\nA: {qa['response']}\n\n")

    pdf_bytes = pdf.output(dest='S').encode('latin1')
    pdf_buffer = BytesIO(pdf_bytes)

    return send_file(
        pdf_buffer,
        as_attachment=True,
        download_name=f"chat_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mimetype='application/pdf'
    )

# -----------------------------
# EXIT HANDLERS
# -----------------------------
@app.route("/exit", methods=["POST"])
def exit_program():
    data = request.get_json()
    save_pdf = data.get("save", False)

    if save_pdf:
        return jsonify({"redirect": "/download_pdf"})

    os._exit(0)

@app.route("/shutdown", methods=["POST"])
def shutdown_server():
    os._exit(0)

# -----------------------------
# START SERVER
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
