# 🎌 Anime Recommendation System using RAG (Qdrant + Phi-3)

This project implements an anime recommendation system using a Retrieval-Augmented Generation (RAG) pipeline.
It combines vector similarity search with a local Large Language Model (LLM) to provide grounded, explainable anime recommendations based on user queries.

The system is fully local, runs on Windows, and does not require OpenAI APIs or cloud services.

## 🚀 What This Project Does

Given a natural-language query such as:

> "Recommend dark psychological mystery anime under 26 episodes"

The system:

1. Retrieves the most relevant anime from a vector database (Qdrant)

2. Applies hard constraints (e.g., episode count)

3. Sends the retrieved context to a local LLM

4. Returns ranked recommendations with explanations

## 🧠 Architecture Overview

```
User Query
   ↓
Sentence Embedding (SentenceTransformers)
   ↓
Qdrant Vector Database (Docker)
   ↓
Top-K Relevant Anime (with metadata)
   ↓
Local LLM (Phi-3 Mini via LM Studio)
   ↓
Ranked Recommendations + Explanations
```

## 🛠️ Tools & Technologies Used

- Vector Database

    - Qdrant

        - Runs locally using Docker

        - Stores embeddings + metadata (title, genres, episodes, synopsis, etc.)

- Embeddings

    - SentenceTransformers

        - Model: all-MiniLM-L6-v2

        - Used for encoding anime descriptions and user queries

- Large Language Model (LLM)

    - Phi-3 Mini (4k Instruct)

        - Model: microsoft/Phi-3-mini-4k-instruct-gguf

        - Runs locally using LM Studio

        - Accessed via OpenAI-compatible API

## 📂 Project Structure

```
anime_recommendation/
│
├── danime_recommendation_dataset.csv   # Original anime dataset
│
├── build_qdrant_index.py               # Builds embeddings and indexes data into Qdrant
├── anime_rag.py                        # Main RAG pipeline (retrieval + LLM generation)
│
├── docker-compose.yml                  # Docker Compose config for Qdrant
│
├── env/                                # Python virtual environment (not committed)
├── requirements.txt                   # Python dependencies
│
└── README.md                           # Project documentation
```

## Dataset
The dataset used in this project is available on Kaggle:  
[Anime Recommendation Dataset](https://www.kaggle.com/datasets/ylmzasel/anime-recommendation-dataset)

## 📌 TODO
1. UMAP/t-SNE visualization of anime clusters
