from fastapi import FastAPI
import pickle
import numpy as np
import pandas as pd

app = FastAPI()

model_knn = None
pivot_table = None
book_isbns = None

@app.on_event("startup")
def load_model():
    global model_knn, pivot_table, book_isbns
    try:
        print("Loading Recommender Models...")
        with open('../../models/model_knn.pkl', 'rb') as f:
            model_knn = pickle.load(f)
        with open('../../models/pivot_table.pkl', 'rb') as f:
            pivot_table = pickle.load(f)
        with open('../../models/book_isbns.pkl', 'rb') as f:
            book_isbns = pickle.load(f)
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading models: {e}")

@app.get("/recommend")
def recommend(isbn: str):
    if model_knn is None:
        return {"error": "Model not loaded"}
    
    if isbn not in book_isbns:
        return {"error": "Book not found in popular index", "recommendations": []}

    # Find the index of the book in the pivot table
    query_index = book_isbns.index(isbn)
    
    # Get recommendations (reshape to 2D array)
    distances, indices = model_knn.kneighbors(
        pivot_table.iloc[query_index, :].values.reshape(1, -1), 
        n_neighbors=6
    )
    
    # Helper to get ISBNs
    recommended_isbns = []
    for i in range(1, len(distances.flatten())):
        recommended_isbns.append(pivot_table.index[indices.flatten()[i]])

    # We return just the ISBNs; the frontend will fetch details from Content Service
    return {"isbn": isbn, "recommendations": recommended_isbns}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5002)