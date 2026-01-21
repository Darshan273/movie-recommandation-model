import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Movie Recommender", page_icon="🎬")

movies = pickle.load(open('movies.pkl','rb'))
similarity = pickle.load(open('similarity.pkl','rb'))

st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox(
    "Select a movie",
    movies['title'].values
)

def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]

    movies_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    return [movies.iloc[i[0]].title for i in movies_list]

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    for movie in recommendations:
        st.write("✅", movie)
    