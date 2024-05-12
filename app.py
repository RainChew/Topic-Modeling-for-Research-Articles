import pickle
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

with open('dictionary.pkl', 'rb') as f:
    dictionary = pickle.load(f)

with open('best_lda_model.pkl', 'rb') as f:
    saved_lda_model = pickle.load(f)

df = pd.read_csv("data_with_topics.csv")

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

def preprocess_text(text):
    return text.lower().split()

def get_topic_similarity(text):
    processed_text = preprocess_text(text)
    bow_vector = dictionary.doc2bow(processed_text)
    topic_distribution = saved_lda_model[bow_vector]
    return topic_distribution

def get_similar_abstracts(topic, df, num_abstracts=5):
    filtered_df = df[df['Topic'] == topic]
    random_abstracts = filtered_df.sample(n=num_abstracts)[['TITLE', 'ABSTRACT']].values.tolist()
    return random_abstracts

st.title('LDA Topic Similarity App')

st.write(
    f"""
    <style>
        .reportview-container .main .block-container{{
            max-width: 800px;
            padding: 2rem;
        }}
        .streamlit-text-area {{
            width: 100%;
            height: 150px;
            padding: 12px 20px;
            margin: 8px 0;
            box-sizing: border-box;
            border: 1px solid #ccc;
            border-radius: 4px;
            resize: none;
        }}
        .streamlit-button {{
            background-color: #4CAF50;
            color: white;
            padding: 12px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            float: right;
        }}
        .streamlit-button:hover {{
            background-color: #45a049;
        }}
        .topic-container {{
            margin-top: 2rem;
        }}
        .similar-container {{
            margin-top: 2rem;
            border-top: 1px solid #ccc;
            padding-top: 2rem;
        }}
        .similar-topic {{
            font-weight: bold;
            margin-top: 1rem;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

if st.button('Generate Topic Label'):
    st.write('Topic Label Generated Successfully!')

user_input = st.text_area('Enter your abstract:', '')

if user_input:
    st.markdown('## User Input:')
    st.write(user_input)
    st.markdown('---')
    
    topic_distribution = get_topic_similarity(user_input)
    topics, scores = zip(*topic_distribution)
    
    highest_prob_topic_idx = np.argmax(scores)
    highest_prob_topic_label = topic_labels[topics[highest_prob_topic_idx]]
    
    st.markdown(f"The abstract is most similar to the topic: **{highest_prob_topic_label}**")
    st.markdown('---')
    
    topic_df = pd.DataFrame({'Topic': [topic_labels[i] for i in topics], 'Probability': scores})
    
    topic_distribution_sorted = sorted(topic_distribution, key=lambda x: x[1], reverse=True)
    
    sorted_topics, sorted_scores = zip(*topic_distribution_sorted)
    sorted_topic_df = pd.DataFrame({'Topic': [topic_labels[i] for i in sorted_topics], 'Probability': sorted_scores})
    st.markdown('## Topic Distribution:')
    fig = px.bar(sorted_topic_df, x='Topic', y='Probability', title='Topic Distribution', color='Probability')
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig)
    st.markdown('---')
    
    st.markdown('## Similar Topics, Titles, and Abstracts:')
    for topic, score in topic_distribution_sorted:
        st.markdown(f'<div class="similar-topic">Similar to {topic_labels[topic]} (Probability: {score:.2f}):</div>', unsafe_allow_html=True)
        similar_abstracts = get_similar_abstracts(topic, df)
        for idx, (title, abstract) in enumerate(similar_abstracts, start=1):
            st.markdown(f'**Title {idx}:** {title}')
            st.markdown(f'**Abstract {idx}:** {abstract}')
        st.markdown('---')
