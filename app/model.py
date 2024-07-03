import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# Load datasets
movies = pd.read_csv('../data/movies.csv')
ratings = pd.read_csv('../data/ratings.csv')

# Merge datasets
data = pd.merge(ratings, movies, on='movieId')

# Handle missing values
data.dropna(inplace=True)

# Pivot table
movie_matrix = data.pivot_table(index='userId', columns='title', values='rating').fillna(0)

# Collaborative filtering with SVD
svd = TruncatedSVD(n_components=20)
matrix_svd = svd.fit_transform(movie_matrix)

# Content-based filtering
movie_features = movies.set_index('movieId').loc[movie_matrix.columns].reset_index()
genres = movie_features['genres'].str.get_dummies(sep='|')
movie_features = movie_features.join(genres)

# Cosine similarity
similarity_matrix = cosine_similarity(movie_features.drop(['movieId', 'title', 'genres'], axis=1))

# Combine collaborative and content-based filtering
combined_similarity = (cosine_similarity(matrix_svd) + similarity_matrix) / 2

# Recommendation function
def recommend(movie_title, n_recommendations=5):
    movie_idx = movie_matrix.columns.get_loc(movie_title)
    similar_movies = combined_similarity[movie_idx]
    movie_indices = similar_movies.argsort()[::-1][1:n_recommendations+1]
    return movie_matrix.columns[movie_indices]

# Save the model
joblib.dump(combined_similarity, 'model.pkl')
