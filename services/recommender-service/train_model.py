import pandas as pd
import pickle
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import os

# Create models directory
os.makedirs('../../models', exist_ok=True)

print("⏳ Loading data for training...")

try:
    # 1. Load Data
    books = pd.read_csv("../../data/Books.csv", encoding="latin-1", on_bad_lines='skip', dtype={'ISBN': str})
    ratings = pd.read_csv("../../data/Ratings.csv", encoding="latin-1", on_bad_lines='skip', dtype={'ISBN': str, 'User-ID': int})

    # 2. Filter Data (To avoid memory crash)
    # Only keep users who rated > 200 books
    x = ratings['User-ID'].value_counts() > 200
    y = x[x].index
    ratings = ratings[ratings['User-ID'].isin(y)]

    # Only keep books with > 50 ratings
    rating_counts = ratings.groupby('ISBN')['Book-Rating'].count()
    popular_books = rating_counts[rating_counts >= 50].index
    final_ratings = ratings[ratings['ISBN'].isin(popular_books)]

    # 3. Create Pivot Table (Matrix)
    print("🔄 Creating Pivot Table...")
    pt = final_ratings.pivot_table(index='ISBN', columns='User-ID', values='Book-Rating')
    pt.fillna(0, inplace=True)

    # 4. Train Model (Nearest Neighbors)
    print("🧠 Training Model...")
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(pt.values)

    # 5. Save Artifacts
    print("💾 Saving models...")
    with open('../../models/model_knn.pkl', 'wb') as f:
        pickle.dump(model_knn, f)
    
    with open('../../models/pivot_table.pkl', 'wb') as f:
        pickle.dump(pt, f)
        
    # Save a map of ISBNs for lookup
    with open('../../models/book_isbns.pkl', 'wb') as f:
        pickle.dump(list(pt.index), f)

    print(f"✅ Training complete! Model trained on {pt.shape[0]} books.")

except Exception as e:
    print(f"❌ Error: {e}")