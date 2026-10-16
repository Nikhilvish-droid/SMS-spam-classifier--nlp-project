import re

from nltk.stem import PorterStemmer, WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def nlp_processing(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = text.strip()

    words = text.split()
    words = [stemmer.stem(word) for word in words]
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)


def preprocess_batch(texts):
    # lives in its own module (not a lambda) so the saved pipeline can be unpickled elsewhere
    return [nlp_processing(text) for text in texts]
