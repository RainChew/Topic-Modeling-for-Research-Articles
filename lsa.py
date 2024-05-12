import pickle
from gensim.models import LsiModel
from gensim.corpora import Dictionary

# Load the LSA model and dictionary
def load_lsa_model(filename):
    with open(filename, 'rb') as f:
        lsa_model_data = pickle.load(f)
    return lsa_model_data

# Preprocess text for LSA
def preprocess_text_lsa(text):
    processed_text = text.lower().split()  # Example preprocessing: converting to lowercase and splitting
    return processed_text

# Get topic similarity using LSA
def get_topic_similarity_lsa(text, lsa_model_data):
    lsa_model_optimal = lsa_model_data['lsa_model_optimal']
    dictionary = lsa_model_data['dictionary']
    processed_text = preprocess_text_lsa(text)
    bow_vector = dictionary.doc2bow(processed_text)
    topic_distribution = lsa_model_optimal[bow_vector]
    return topic_distribution
