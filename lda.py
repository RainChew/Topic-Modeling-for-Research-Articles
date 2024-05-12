import pickle
from gensim.models import LdaModel
from gensim.corpora import Dictionary

# Load the LDA model and dictionary
def load_lda_model(filename):
    with open(filename, 'rb') as f:
        lda_model = pickle.load(f)
    return lda_model

def preprocess_text_lda(text):
    processed_text = text.lower().split()  # Example preprocessing: converting to lowercase and splitting
    return processed_text

def get_topic_similarity_lda(text, lda_model, dictionary):
    processed_text = preprocess_text_lda(text)
    bow_vector = dictionary.doc2bow(processed_text)
    topic_distribution = lda_model[bow_vector]
    return topic_distribution