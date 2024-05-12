import pickle
import streamlit as st
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
import matplotlib.pyplot as plt

# Load the dictionary used for LDA model
with open('dictionary.pkl', 'rb') as f:
    dictionary = pickle.load(f)

# Load the best LDA model
with open('best_lda_model.pkl', 'rb') as f:
    saved_lda_model = pickle.load(f)

# Function to preprocess text
def preprocess_text(text):
    # Implement your preprocessing steps here
    return text.lower().split()

# Function to get topic similarity for new text
def get_topic_similarity(text):
    processed_text = preprocess_text(text)
    bow_vector = dictionary.doc2bow(processed_text)
    topic_distribution = saved_lda_model[bow_vector]
    return topic_distribution

# Mapping of topic indices to topic labels
topic_labels = {
    0: 'Statistics and Probability',
    1: 'Miscellaneous',
    2: 'Algorithmic Problem Solving',
    3: 'Machine Learning and Neural Networks',
    4: 'Graph Theory',
    5: 'Astronomy and Astrophysics',
    6: 'Social Networks and Systems',
    7: 'Mathematics and Proof',
    8: 'Dynamics and Systems Theory',
    9: 'Physics - Quantum Mechanics'
}

# Streamlit app
st.title('LDA Topic Similarity App')

user_input = st.text_input('Enter your text:')
if user_input:
    topic_distribution = get_topic_similarity(user_input)
    topics, scores = zip(*topic_distribution)
    
    # Plot the topic distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(range(len(topics)), scores, tick_label=[topic_labels[i] for i in topics], color='skyblue')
    ax.set_xlabel('Topics')
    ax.set_ylabel('Probability')
    ax.set_title('Topic Distribution')
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

    st.write('Topic Distribution:', [(topic_labels[i], score) for i, score in topic_distribution])
