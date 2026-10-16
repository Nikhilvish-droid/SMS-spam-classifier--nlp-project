# Email Spam Classifier

An NLP project that classifies text messages as **spam** or **ham** (legitimate) using TF-IDF features
and a classical machine learning model.

## Overview

Given the text of a message, the model predicts whether it is spam. Messages are cleaned and stemmed,
converted to TF-IDF vectors, and classified. The notebook uses Multinomial Naive Bayes, and the
script `src/train.py` packages the whole flow as a single scikit-learn `Pipeline` with a linear SVM.

## Project structure

```
Email-spam-classifier--nlp-project/
├── data/
│   ├── SMSSpamCollection      # raw dataset (5,574 labelled SMS messages, tab-separated)
│   └── readme                 # original description of the dataset
├── notebooks/
│   └── model_train.ipynb         # EDA, text cleaning, TF-IDF + Naive Bayes experiments
├── src/
│   ├── preprocess.py          # text cleaning / stemming functions used by the pipeline
│   └── train.py               # pipeline version: preprocessing + TF-IDF + linear SVM
├── models/
│   └── SmsSpamClassifier.pkl  # trained pipeline saved by src/train.py
└── README.md
```

## Dataset

The [SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection): each line is a
label (`ham` or `spam`) and the message text, separated by a tab.

- 5,574 messages in total, 403 of them exact duplicates
- After dropping duplicates: 5,169 messages — 4,516 ham (87.4%) and 653 spam (12.6%)
- No missing values

The classes are imbalanced, so accuracy alone is misleading; look at spam precision and recall too.

## Approach

1. **EDA** (`notebooks/Untitled.ipynb`) — checked nulls, duplicates and the class balance.
2. **Text cleaning** — lower-case, strip URLs, remove everything except letters, digits and spaces.
3. **Stop words** — removed using scikit-learn's `ENGLISH_STOP_WORDS` (notebook only).
4. **Stemming and lemmatization** — NLTK `PorterStemmer` followed by `WordNetLemmatizer`.
5. **Features** — `TfidfVectorizer` on the cleaned text.
6. **Model**
   - Notebook: `MultinomialNB(alpha=0.1)`
   - `src/train.py`: `SVC(kernel='linear', probability=True)` inside a `Pipeline`, so the raw message
     goes in and the prediction comes out, with cleaning done by a `FunctionTransformer`
7. **Evaluation** — 80/20 train/test split with `random_state=42`; classification report and
   confusion matrix are printed.

## Getting started

```bash
pip install pandas scikit-learn nltk seaborn matplotlib

# one-time download of the NLTK data used for lemmatization
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"

# train the pipeline and print the classification report and confusion matrix
python src/train.py
```

`train.py` reads the dataset from `data/SMSSpamCollection`, so run it from anywhere — paths are
resolved relative to the script.

## Model evaluation

Multinomial Naive Bayes (notebook), on the held-out 20% test set (1,034 messages):

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| ham   | 0.98      | 1.00   | 0.99     | 894     |
| spam  | 0.97      | 0.90   | 0.93     | 140     |

Overall accuracy: **0.98**.

Confusion matrix (rows = actual, columns = predicted):

|             | Predicted ham | Predicted spam |
|-------------|---------------|----------------|
| Actual ham  | 890           | 4              |
| Actual spam | 14            | 126            |

Only 4 legitimate messages were flagged as spam, while 14 spam messages slipped through. Running
`src/train.py` prints the same report for the SVM pipeline.

## Notes / what's next

- `train.py` saves the trained pipeline to `models/SmsSpamClassifier.pkl`. Load it with `joblib.load`
  (with `src/` on the Python path, because the pipeline uses `preprocess_batch` from `src/preprocess.py`).
- There is no `predict.py` or app yet; a small script or Streamlit app that loads the saved pipeline
  and classifies a pasted message is the natural next step.
- Add a `requirements.txt` with exact versions (`pip freeze` inside your environment).
- The dataset is SMS, not email, so performance on real emails (longer, with headers and HTML) will
  likely be lower — try training on an email corpus such as Enron or SpamAssassin.
- Ideas to try: compare Naive Bayes, SVM and Logistic Regression side by side, tune hyper-parameters
  with `GridSearchCV`, use word n-grams, and handle the class imbalance (e.g. `class_weight='balanced'`).
