import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


CSV_PATH = "./data/anime_recommendation_dataset.csv"
QDRANT_URL = "http://localhost:6333"
COLLECTION = "anime_rag"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2" 

def make_doc(row: pd.Series) -> str:
    title = str(row.get("title", "")).strip()
    genres = str(row.get("genres", "")).strip()
    synopsis = str(row.get("synopsis", "")).strip()
    episodes = row.get("episodes", None)
    score = row.get("score", None)

    return (
        f"Title: {title}\n"
        f"Genres: {genres}\n"
        f"Episodes: {episodes}\n"
        f"Score: {score}\n"
        f"Synopsis: {synopsis}\n"
    )

if __name__ == "__main__":
    df = pd.read_csv(CSV_PATH)

    for c in ["title", "genres", "synopsis"]:
        if c in df.columns:
            df[c] = df[c].fillna("")
    
    docs = [make_doc(df.iloc[i]) for i in range(len(df))]
    model = SentenceTransformer(EMBED_MODEL)
    dim = model.get_sentence_embedding_dimension()

    client = QdrantClient(url=QDRANT_URL)

    # Recreate collection
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    batch_size = 256
    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        batch_docs = docs[start:end]
        batch_df = df.iloc[start:end]

        vecs = model.encode(batch_docs, batch_size=64, show_progress_bar=False, normalize_embeddings=True)

        points = []
        for i, (vec, row, doc) in enumerate(zip(vecs, batch_df.to_dict(orient="records"), batch_docs)):
            episodes_val = None
            if pd.notna(row['episodes']):
                episodes_val = int(row['episodes'])
            pid = start + i
            payload = {
                "title": str(row.get("title", "")),
                "genres": str(row.get("genres", "")),
                "episodes": episodes_val,
                "episodes_known": episodes_val is not None,
                "score": float(row["score"]) if "score" in row and pd.notna(row["score"]) else None,
                "synopsis": str(row.get("synopsis", "")),
                "doc": doc,  # store full doc for RAG context
            }
            points.append(PointStruct(id=pid, vector=vec.tolist(), payload=payload))
        
        client.upsert(collection_name=COLLECTION, points=points)
        print(f"Upserted {start}..{end-1}")
    
    print(f"Done. Qdrant collection: {COLLECTION}")