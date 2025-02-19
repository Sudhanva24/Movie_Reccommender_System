from fastapi import FastAPI, HTTPException
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import ast

# Initialize FastAPI app
app = FastAPI()

# Load the model and data (wrapped in try-except for better error handling)
try:
    model = SentenceTransformer("sbert_model")
    movie_data = pd.read_csv('./Data/vectors.csv')
    
    # Convert string representation of embeddings to numpy arrays
    # Assuming embeddings are stored as strings in the CSV
    movie_data['embeddings'] = movie_data['embeddings'].apply(lambda x: np.array(ast.literal_eval(x)))
except Exception as e:
    print(f"Error during initialization: {str(e)}")
    raise

@app.get("/")
async def home():
    return {"message": "Welcome to the Movie Recommender API"}

@app.get("/recommend/{movie_id}")
async def recommend_movies(movie_id: int):
    try:
        # Check if movie_id exists in DataFrame
        if movie_id not in movie_data["id"].values:
            raise HTTPException(status_code=404, detail="Movie ID not found in dataset")

        # Get the row for the given movie_id
        target_movie = movie_data[movie_data["id"] == movie_id].iloc[0]

        # Extract embedding for the given movie
        target_embedding = target_movie["embeddings"].reshape(1, -1)

        # Create matrix of all embeddings
        all_embeddings = np.vstack(movie_data["embeddings"].values)

        # Compute similarity with all movies
        similarities = cosine_similarity(target_embedding, all_embeddings)[0]

        # Get top 5 similar movies (excluding itself)
        movie_data_copy = movie_data.copy()
        movie_data_copy["similarity"] = similarities
        similar_movies = (
            movie_data_copy[movie_data_copy["id"] != movie_id]
            .nlargest(5, "similarity")
        )

        # Format the recommendations
        recommendations = similar_movies[["title", "similarity"]].to_dict(orient="records")
        xs
        # Round similarity scores for better readability
        for rec in recommendations:
            rec["similarity"] = round(float(rec["similarity"]), 3)

        return {
            "movie": target_movie["title"],
            "recommendations": recommendations
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

# Optional: Add endpoint to get all available movies
@app.get("/movies")
async def get_movies():
    try:
        return {
            "movies": movie_data[["id", "title"]].to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching movies: {str(e)}")