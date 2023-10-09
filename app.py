from flask import Flask, render_template, request
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__, template_folder='templates')

# Load data and define functions
movies = pd.read_csv("movies.csv")

def clean_title(title):
    title = re.sub("[^a-zA-Z0-9 ]", "", title)
    return title

movies["clean_title"] = movies["title"].apply(clean_title)

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
tfidf = vectorizer.fit_transform(movies["clean_title"])

def search(title):
    title = clean_title(title)
    query_vec = vectorizer.transform([title])
    similarity = cosine_similarity(query_vec, tfidf).flatten()
    indices = np.argpartition(similarity, -5)[-5:]
    results = movies.iloc[indices].iloc[::-1]
    
    return results

ratings = pd.read_csv("ratings.csv")

def find_similar_movies(movie_id):
    similar_users = ratings[(ratings["movieId"] == movie_id) & (ratings["rating"] > 4)]["userId"].unique()
    similar_user_recs = ratings[(ratings["userId"].isin(similar_users)) & (ratings["rating"] > 4)]["movieId"]
    similar_user_recs = similar_user_recs.value_counts() / len(similar_users)

    similar_user_recs = similar_user_recs[similar_user_recs > 0.10]
    all_users = ratings[(ratings["movieId"].isin(similar_user_recs.index)) & (ratings["rating"] > 4)]
    all_user_recs = all_users["movieId"].value_counts() / len(all_users["userId"].unique())
    rec_percentages = pd.concat([similar_user_recs, all_user_recs], axis=1)
    rec_percentages.columns = ["similar", "all"]
    
    rec_percentages["score"] = rec_percentages["similar"] / rec_percentages["all"]
    rec_percentages = rec_percentages.sort_values("score", ascending=False)
    return rec_percentages.head(10).merge(movies, left_index=True, right_on="movieId")[["score", "title", "genres"]]

@app.route('/')
def home():
    return render_template('index.html', results=[], recommendations=[])

@app.route('/recommend', methods=['POST'])
def recommend():
    title = request.form.get('title')
    results = search(title)
    if len(results) > 0:
        movie_id = results.iloc[0]["movieId"]
        recommendations = find_similar_movies(movie_id)
        # Convert DataFrames to a list of dictionaries
        results_data = results.to_dict(orient='records')
        recommendations_data = recommendations.to_dict(orient='records')
        return render_template('index.html', results=results_data, recommendations=recommendations_data)
    else:
        return render_template('index.html', error="Movie not found!")

if __name__ == "__main__":
    app.run(debug=True)