import streamlit as st
import pandas as pd
import pickle

# Title
st.title("Movie Recommendation System")

# Load data
df = pd.read_csv("cleaned_data1.csv")
similarity = pickle.load(open("similarity.pkl", "rb"))

# Movie names for dropdown
names = sorted(df["title"].unique())

# Function to get movie index
def get_movie_index(name):
    for i in df.index:
        if name == df.loc[i, "title"]:
            return i
    return -1

# Function to get movie name
def get_movie_name(i):
    if i >= len(df):
        return ""
    return df.loc[i, "title"]

# Dropdown
name = st.selectbox("Select Movie You Watched", names)

# Recommend button
if st.button("Recommend"):

    index = get_movie_index(name)

    if index == -1:
        st.error("Movie not found")
    else:
        st.write("### Predicted Next 5 Movies")

        distances = similarity[index]

        movies = sorted(
            list(enumerate(distances)),
            reverse=True,
            key=lambda x: x[1]
        )

        # Skip the selected movie itself
        for movie in movies[1:6]:
            st.write(get_movie_name(movie[0]))