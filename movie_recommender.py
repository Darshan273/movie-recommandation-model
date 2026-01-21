
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
import numpy as np

# Load and prepare the dataset
df = pd.read_csv('https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Movies%20Recommendation.csv')
df_feature = df[['Movie_Genre','Movie_Keywords','Movie_Tagline','Movie_Cast','Movie_Director']].fillna('')
combined = df_feature['Movie_Genre'] + ' ' + df_feature['Movie_Keywords'] + ' ' + df_feature['Movie_Tagline'] + ' ' + df_feature['Movie_Cast'] + ' ' + df_feature['Movie_Director']

# Vectorize and compute similarity
tfidf = TfidfVectorizer()
vectors = tfidf.fit_transform(combined)
similarity = cosine_similarity(vectors)

# Flask app
app = Flask(__name__)

@app.route("/recommend", methods=["GET"])
def recommend():
    movie = request.args.get('movie')
    if movie not in df['Movie_Title'].values:
        return jsonify({'error': 'Movie not found'}), 404

    index = df[df['Movie_Title'] == movie].index[0]
    distances = list(enumerate(similarity[index]))
    sorted_movies = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
    recommended = [df.iloc[i[0]]['Movie_Title'] for i in sorted_movies]

    return jsonify({'recommended_movies': recommended})

if __name__ == '__main__':
    app.run(debug=True)