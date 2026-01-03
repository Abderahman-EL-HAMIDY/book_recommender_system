from fastapi import FastAPI, HTTPException
import pandas as pd
import os

app = FastAPI()

# Global variable to store data
books_df = None

@app.on_event("startup")
def load_data():
    global books_df
    try:
        print("📚 Loading Books dataset...")
        # The dataset often uses different encodings or delimiters. 
        # We use error_bad_lines=False to skip messy rows.
        file_path = "../../data/Books.csv"
        
        # Try loading with standard settings for the Book-Crossing dataset
        books_df = pd.read_csv(
            file_path, 
            dtype={'ISBN': str},
            on_bad_lines='skip',
            encoding="latin-1" 
        )
        
        # Clean up column names (strip spaces)
        books_df.columns = books_df.columns.str.strip()
        
        # Ensure we have the images (Handle missing values)
        books_df['Image-URL-L'] = books_df['Image-URL-L'].fillna("https://via.placeholder.com/150")
        
        print(f"✅ Loaded {len(books_df)} books.")
    except Exception as e:
        print(f"❌ Error loading books: {e}")
        books_df = pd.DataFrame()

@app.get("/books")
def get_books(limit: int = 20):
    if books_df is None or books_df.empty:
        return {"error": "Books data not loaded"}
    
    # Return random sample or top N
    sample = books_df.head(limit).fillna("")
    return {"books": sample.to_dict(orient="records")}

@app.get("/books/{isbn}")
def get_book_details(isbn: str):
    if books_df is None:
        return {"error": "Data not loaded"}
    
    # Filter by ISBN
    book = books_df[books_df['ISBN'] == isbn]
    
    if book.empty:
        raise HTTPException(status_code=404, detail="Book not found")
    
    return book.iloc[0].fillna("").to_dict()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)