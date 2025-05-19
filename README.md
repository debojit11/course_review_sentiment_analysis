# 📘 Course Review Sentiment Classifier

A Gradio-powered NLP app that classifies course reviews into **positive**, **neutral**, or **negative** sentiments using a fine-tuned DistilBERT model. Built for user-friendly interaction with correction feedback and logging features to support continual improvement.

🔗 **Live Demo**: [Hugging Face Space](https://huggingface.co/spaces/debojit01/course-review-sentiment)

---

## 📝 Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [App Structure](#app-structure)
- [Logging and Feedback](#logging-and-feedback)
- [How to Run](#how-to-run)
- [Model Details](#model-details)
- [License](#license)

---

## 🚀 Features

- 🔍 Classifies sentiment of textual course reviews.
- 🤖 Powered by fine-tuned `DistilBERT` from Hugging Face.
- 🧠 Supports **3 sentiment classes**: `positive`, `neutral`, and `negative`.
- 📊 Displays class probabilities using Gradio’s `Label` component.
- ✅ Lets users correct model predictions post-inference.
- 🗂️ Logs predictions and user corrections to local CSV files.
- 📈 Designed to support feedback-driven model retraining in the future.

---

## 🛠️ Tech Stack

- **Language**: Python 3.8+
- **Model**: [DistilBERT](https://huggingface.co/distilbert-base-uncased) fine-tuned on sentiment classification
- **Framework**: [Hugging Face Transformers](https://huggingface.co/transformers/)
- **Frontend UI**: [Gradio](https://gradio.app/)
- **Logging**: CSV-based persistent storage

---

## 📦 Installation

```bash
git clone https://github.com/debojit01/course-review-sentiment.git
cd course-review-sentiment

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

**requirements.txt**
```text
torch
transformers
gradio
```

---

## 💻 Usage

```bash
python app.py
```

Then open your browser to the URL shown in the terminal (usually http://127.0.0.1:7860/).

---

## 🗂️ App Structure

```
course-review-sentiment/
├── app.py                   # Main Gradio app
├── logs.csv                 # Prediction logs (auto-created)
├── corrections.csv          # User correction logs (auto-created)
├── requirements.txt         # Python dependencies
└── README.md
```

---

## 📊 Logging and Feedback

- **logs.csv**: Records every prediction with a timestamp.
  - Format: `timestamp, input_text, predicted_label`
- **corrections.csv**: Records user corrections when sentiment was misclassified.
  - Format: `timestamp, input_text, predicted_label, user_correction`

This design enables feedback loops for future model retraining and evaluation.

---

## 🧠 Model Details

- Hugging Face Model: [`debojit01/course-review-sentiment`](https://huggingface.co/debojit01/course-review-sentiment)
- Base Model: `distilbert-base-uncased`
- Task: Sentiment classification
- Classes: `positive`, `neutral`, `negative`

---

## 🧪 How to Run Locally

```bash
# Run the app
python app.py
```

If running inside a Jupyter/Colab environment, use:
```python
demo.launch(share=True)
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙌 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Gradio UI Framework](https://gradio.app/)
- All course reviewers who made this dataset possible for fine-tuning!

---
