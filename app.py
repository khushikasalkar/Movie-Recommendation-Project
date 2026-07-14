import streamlit as st
import pandas as pd
import pickle
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Title
st.title("Movie Recommendation System")

# Load data
df = pd.read_csv("cleaned_data1.csv")

# Load or generate similarity matrix
if os.path.exists("similarity.pkl"):
    with open("similarity.pkl", "rb") as f:
        similarity = pickle.load(f)
else:
    st.info("Generating similarity matrix for the first time. Please wait...")

    cv = CountVectorizer(max_features=10000, stop_words="english")
    vectors = cv.fit_transform(df["tags"].fillna("")).toarray()

    similarity = cosine_similarity(vectors)

    with open("similarity.pkl", "wb") as f:
        pickle.dump(similarity, f)

    st.success("Similarity matrix generated successfully!")

# Movie names
names = sorted(df["title"].unique())

# Function to get movie index
def get_movie_index(name):
    try:
        return df[df["title"] == name].index[0]
    except:
        return -1

# Dropdown
name = st.selectbox("Select Movie You Watched", names)

# Recommend
if st.button("Recommend"):

    index = get_movie_index(name)

    if index == -1:
        st.error("Movie not found")
    else:
        distances = similarity[index]

        movies = sorted(
            list(enumerate(distances)),
            key=lambda x: x[1],
            reverse=True
        )

        st.subheader("Recommended Movies")

        for movie in movies[1:6]:
            st.write(df.iloc[movie[0]].title)
