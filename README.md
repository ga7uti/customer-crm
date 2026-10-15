# ML Projects

A single repo hosting two classic-NLP classification projects, each self-contained
in its own folder:

- **`intent-anlaysis/`** — customer-support **intent** classifier (77 banking intents).
- **`customer-review/`** — product-review **sentiment** classifier (positive / negative).

Each project trains a model end-to-end and ships an interactive demo inside a
single Jupyter notebook.

## What's in this repo

```
.
├── intent-anlaysis/
│   ├── intent_analysis.ipynb       # Intent workflow + demo (main artifact)
│   ├── train.csv                   # Training dataset
│   ├── test.csv                    # Test dataset
│   ├── ticket_intent_model.joblib  # Trained model (generated)
│   └── validation_errors.csv       # Error-analysis output (generated)
├── customer-review/
│   ├── customer_review.ipynb           # Review workflow + demo (main artifact)
│   ├── flipkart_data.csv               # Raw reviews
│   ├── flipkart_data_with_sentiment.csv
│   ├── flipkart_data_binary.csv
│   ├── customer_review_model.joblib    # Trained model (generated)
│   └── requirements.txt                # Project-specific dependencies
├── requirements.txt
└── README.md
```

## Setup

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Project: Intent analysis (`intent-anlaysis/`)

Maps a free-text support message to one of **77 banking support intents** using
classic NLP (TF-IDF) plus Logistic Regression, trained on the
[PolyAI banking support dataset](https://github.com/PolyAI-LDN/task-specific-datasets).

Training and the demo live in `intent_analysis.ipynb`.

`ticket_intent_model.joblib` and `validation_errors.csv` are generated when you
run the notebook.

### Features

- End-to-end notebook: loading, analysis, training, tuning and evaluation.
- Text pipeline: `TfidfVectorizer` (n-grams) → `LogisticRegression`.
- Model comparison (Logistic Regression vs. Decision Tree).
- Feature (n-gram) and regularization (`C`) experiments.
- Error analysis with saved mis-classifications.
- Trained model persisted with `joblib`.
- Interactive demo: predict an intent and explain which words drive it.

### Model

| Stage      | Component                                       |
| ---------- | ----------------------------------------------- |
| Features   | `TfidfVectorizer(lowercase=True, ngram=(1,1))`  |
| Classifier | `LogisticRegression(C=10.0, max_iter=1000)`     |
| Task       | Multi-class text classification (77 intents)    |

Evaluation uses **accuracy** and **macro-F1** to account for class imbalance.

### Usage

Open the notebook and run the cells top to bottom:

```bash
jupyter notebook intent-anlaysis/intent_analysis.ipynb
```

- Sections 1–9 train and save the model to `ticket_intent_model.joblib`.
- Section 10 is the interactive demo: set the `text` variable to any customer
  message to see the predicted intent plus the words that drove the decision.

---

## Project: Customer review (`customer-review/`)

Maps the free-text of a product review to a **positive** or **negative**
sentiment using classic NLP (TF-IDF) plus Logistic Regression, trained on the
Flipkart product reviews dataset. A three-way (positive / neutral / negative)
label is derived from the star rating, then neutral reviews are dropped to build
the binary training set.

Training and the demo live in `customer_review.ipynb`.

`flipkart_data_with_sentiment.csv`, `flipkart_data_binary.csv` and
`customer_review_model.joblib` are generated when you run the notebook.

### Features

- End-to-end notebook: loading, sentiment labelling, training and evaluation.
- Text pipeline: `TfidfVectorizer` (n-grams) → `LogisticRegression`.
- Model comparison (Logistic Regression vs. Naive Bayes vs. Decision Tree).
- Regularization (`C`), smoothing (`alpha`) and n-gram experiments.
- Threshold analysis on the negative class.
- Trained model persisted with `joblib`.
- Interactive demo: predict the sentiment of a single review.

### Model

| Stage      | Component                                                  |
| ---------- | ---------------------------------------------------------- |
| Features   | `TfidfVectorizer(lowercase=True, ngram=(1,1))`             |
| Classifier | `LogisticRegression(C=10.0, max_iter=1000, class_weight='balanced')` |
| Task       | Binary text classification (positive / negative)           |

Evaluation uses **accuracy** and **macro-F1** to account for class imbalance.

### Usage

Open the notebook and run the cells top to bottom:

```bash
jupyter notebook customer-review/customer_review.ipynb
```

- Sections 1–8 explore the data and compare models.
- Section 9 trains and saves the model to `customer_review_model.joblib`.
- Section 10 is the interactive demo: set the `review` variable to any review to
  see the predicted sentiment and its confidence.

## License

For educational use. Datasets © PolyAI-LDN and their respective owners.
