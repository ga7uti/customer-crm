# ML Intent Analysis

A single-repo project for **training** and **demoing** a customer-support intent
classifier. The model maps a free-text support message to one of **77 banking
support intents** using classic NLP (TF-IDF) plus Logistic Regression, trained
on the [PolyAI banking support dataset](https://github.com/PolyAI-LDN/task-specific-datasets).

Training and the demo live in one place: `intent_analysis.ipynb`.

## What's in this repo

```
.
├── intent_analysis.ipynb      # Full workflow + demo (this is the main artifact)
├── train.csv                  # Training dataset
├── test.csv                   # Test dataset
├── ticket_intent_model.joblib # Trained model (created by section 9)
├── validation_errors.csv      # Error analysis output (created by section 4)
├── requirements.txt
└── README.md
```

`ticket_intent_model.joblib` and `validation_errors.csv` are generated when you
run the notebook.

## Features

- End-to-end notebook: loading, analysis, training, tuning and evaluation.
- Text pipeline: `TfidfVectorizer` (n-grams) → `LogisticRegression`.
- Model comparison (Logistic Regression vs. Decision Tree).
- Feature (n-gram) and regularization (`C`) experiments.
- Error analysis with saved mis-classifications.
- Trained model persisted with `joblib`.
- Interactive demo: predict an intent and explain which words drive it.

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

The data comes from the PolyAI banking support dataset (77 intents). The
notebook downloads it automatically and saves `train.csv` / `test.csv` locally
in section 1, so it only needs to run once.

## Model

| Stage      | Component                                       |
| ---------- | ----------------------------------------------- |
| Features   | `TfidfVectorizer(lowercase=True, ngram=(1,1))`  |
| Classifier | `LogisticRegression(C=10.0, max_iter=1000)`     |
| Task       | Multi-class text classification (77 intents)    |

Evaluation uses **accuracy** and **macro-F1** to account for class imbalance.

## Usage

Open the notebook and run the cells top to bottom:

```bash
jupyter notebook intent_analysis.ipynb
```

- Sections 1–9 train and save the model to `ticket_intent_model.joblib`.
- Section 10 is the interactive demo: set the `text` variable to any customer
  message to see the predicted intent plus the words that drove the decision.

## License

For educational use. Dataset © PolyAI-LDN.
