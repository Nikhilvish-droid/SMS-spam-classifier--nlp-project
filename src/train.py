import joblib
from pathlib import Path
import pandas as pd

import nltk

# nltk.download("wordnet")
# nltk.download("omw-1.4")

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from preprocess import preprocess_batch

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

data_dir = Path(__file__).parent.parent/'data'
model_dir = Path(__file__).parent.parent/'models'

df = pd.read_csv(data_dir/'SMSSpamCollection',
            sep='\t',
            names=['labels', 'messages'],
            header = None
            )
print('data loaded successfully')

df = df.drop_duplicates()

# df['messages'] = df['messages'].map(lambda x : nlp_processing(x))

X = df['messages']
y = df['labels']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=42)


model_pipeline = Pipeline([

    ('preprocessing', FunctionTransformer(preprocess_batch, validate = False)),

    ('tfidf', TfidfVectorizer()),
    ('model', SVC(kernel = 'linear'))
])
print('model_pipeline created')


model_pipeline.fit(X_train, y_train)
print("trained model successfully")

y_pred = model_pipeline.predict(X_test)

report = classification_report(y_pred= y_pred, y_true= y_test)
matrix = confusion_matrix(y_pred=y_pred, y_true=y_test, labels=['ham', 'spam'])

print("classification report -->")
print(report)

print('matrix -->')
print(matrix)


joblib.dump(model_pipeline, model_dir/'SmsSpamClassifier.pkl')
print('model saved')
