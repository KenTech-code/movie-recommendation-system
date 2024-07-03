from flask import Flask, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model and data
data = pd.read_csv('../data/cleaned_data.csv')
model = joblib.load('model.pkl')

# Load data
movies = pd.read_csv('../data/movies.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_title = request.form['movie']
    recommendations = model.recommend(movie_title)
    return render_template('index.html', recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
