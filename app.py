import gradio as gr
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from datetime import datetime
import csv
import os

# Load model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained("debojit01/course-review-sentiment")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

labels = ['negative', 'neutral', 'positive']

# Setup log files
log_path = "logs.csv"
corrections_path = "corrections.csv"

for path, headers in [(log_path, ["timestamp", "input_text", "predicted_label"]),
                      (corrections_path, ["timestamp", "input_text", "predicted_label", "user_correction"])]:
    if not os.path.exists(path):
        with open(path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)

def classify_review(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    prob_dict = {label: float(prob) for label, prob in zip(labels, probs)}
    predicted_label = labels[torch.argmax(probs).item()]

    # Logging
    with open(log_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([datetime.now().isoformat(), text, predicted_label])
    
    return prob_dict, text, predicted_label

def save_correction(text, predicted_label, user_correction):
    if user_correction != predicted_label:
        with open(corrections_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([datetime.now().isoformat(), text, predicted_label, user_correction])
    return f"✅ Thanks! Correction recorded: {user_correction}"

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 📘 Course Review Sentiment Classifier")
    gr.Markdown("Enter a course review and get the sentiment prediction. You can correct the result if needed.")

    input_text = gr.Textbox(lines=4, placeholder="Enter course review here...")
    output_label = gr.Label(num_top_classes=3, label="Predicted Sentiment")
    predict_btn = gr.Button("Classify")

    with gr.Row(visible=False) as correction_row:
        gr.Markdown("### ❓ Is the prediction wrong?")
        correction_dropdown = gr.Dropdown(choices=labels, label="Correct Sentiment")
        submit_btn = gr.Button("Submit Correction")
        correction_status = gr.Textbox(interactive=False)

    hidden_text = gr.Textbox(visible=False)
    hidden_pred = gr.Textbox(visible=False)

    def show_correction_ui(_, text, pred):
        return gr.update(visible=True), text, pred

    predict_btn.click(classify_review, inputs=input_text, outputs=[output_label, hidden_text, hidden_pred]) \
               .then(show_correction_ui, inputs=[output_label, hidden_text, hidden_pred],
                     outputs=[correction_row, hidden_text, hidden_pred])

    submit_btn.click(save_correction, inputs=[hidden_text, hidden_pred, correction_dropdown], outputs=correction_status)

if __name__ == "__main__":
    demo.launch()
