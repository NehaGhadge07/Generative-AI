# build_faiss.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os

# === CONFIG ===s
CSV_FILE = "13k-recipes.csv"
TEXT_COLUMN = "Cleaned_Ingredients"  # change to match your dataset column
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FAISS_INDEX_FILE = "recipe_faiss.index"
DOCS_FILE = "docs.pkl"

# === LOAD DATA ===
df = pd.read_csv(CSV_FILE)
assert TEXT_COLUMN in df.columns, f"Column '{TEXT_COLUMN}' not found in dataset"
texts = df[TEXT_COLUMN].dropna().tolist()

# === CHUNKING (optional) ===
# If your texts are very long, you can chunk them here
# For now, we'll use full texts as-is

# === EMBEDDING ===
model = SentenceTransformer(EMBEDDING_MODEL)
embeddings = model.encode(texts, show_progress_bar=True)

# === SAVE DOCS (to match FAISS results later) ===
with open(DOCS_FILE, "wb") as f:
    pickle.dump(texts, f)

# === BUILD FAISS INDEX ===
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

faiss.write_index(index, FAISS_INDEX_FILE)

print(f"FAISS index built and saved to: {FAISS_INDEX_FILE}")
print(f"Document texts saved to: {DOCS_FILE}")
