import re
from typing import Optional, Tuple
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range, MatchValue

# ------------ CONFIG ------------
QDRANT_URL = "http://localhost:6333"
COLLECTION = "anime_rag" 

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"
LMSTUDIO_API_KEY = "lm-studio"
LMSTUDIO_MODEL = "phi-3-mini-4k-instruct"
# --------------------------------

def parse_episode_constraint(query: str) -> Optional[Tuple[str, int]]:
    # Parses episode constraints from the query.
    # Returns a tuple of (operator, value) or None if no constraint found.
    # For including more information for recommendations based on episode count.
    q = query.lower()

    m = re.search(r"(?:under|less than)\s+(\d+)\s+episodes", q)
    if m:
        return ("lt", int(m.group(1)))

    m = re.search(r"(?:at most|no more than|<=)\s*(\d+)\s+episodes", q)
    if m:
        return ("lte", int(m.group(1)))

    m = re.search(r"(?:at least|>=)\s*(\d+)\s+episodes", q)
    if m:
        return ("gte", int(m.group(1)))

    m = re.search(r"(?:exactly)\s+(\d+)\s+episodes", q)
    if m:
        return ("eq", int(m.group(1)))

    return None

def build_qdrant_filter(query: str) -> Optional[Filter]:
    ep = parse_episode_constraint(query)
    if not ep:
        return None

    op, val = ep
    if op == "eq":
        return Filter(must=[FieldCondition(key="episodes", match=MatchValue(value=val))])
    if op == "lt":
        return Filter(must=[FieldCondition(key="episodes", range=Range(lt=val))])
    if op == "lte":
        return Filter(must=[FieldCondition(key="episodes", range=Range(lte=val))])
    if op == "gte":
        return Filter(must=[FieldCondition(key="episodes", range=Range(gte=val))])
    return None

def build_context(hits, max_items: int = 10) -> str:
    blocks = []
    for i, h in enumerate(hits[:max_items], start=1):
        p = h.payload or {}
        title = p.get("title", "")
        genres = p.get("genres", "")
        episodes = p.get("episodes", None)
        score = p.get("score", None)
        synopsis = p.get("synopsis", "")

        blocks.append(
            f"[{i}] Title: {title}\n"
            f"Genres: {genres}\n"
            f"Episodes: {episodes}\n"
            f"Score: {score}\n"
            f"Synopsis: {synopsis}\n"
        )
    return "\n".join(blocks)



class AnimeRAG:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.qdrant = QdrantClient(url=QDRANT_URL)
        self.llm = OpenAI(base_url=LMSTUDIO_BASE_URL, api_key=LMSTUDIO_API_KEY, timeout=60)

    def recommend(self, query: str, top_k_retrieve: int = 30, max_context_items: int = 6, n_recs: int = 5) -> str:
        # 1) Embed query
        qvec = self.embedder.encode([query], normalize_embeddings=True)[0]

        # 2) Retrieve from Qdrant
        qfilter = build_qdrant_filter(query)

        result = self.qdrant.query_points(
            collection_name=COLLECTION,
            query=qvec.tolist(),
            limit=top_k_retrieve,
            with_payload=True,
            query_filter=qfilter,
        )
        hits = result.points

        print("Retrieved:", len(hits))
        print("First:", hits[0].payload.get("title"))

        if not hits:
            return "No candidates were retrieved from the vector database (possibly due to strict filters)."

        # 3) Build RAG context
        context = build_context(hits, max_items=max_context_items)

        # 4) Ask Phi-3 to rank + explain
        system = (
            "You are an anime recommendation assistant.\n"
            "Rules:\n"
            "- You MUST ONLY recommend anime from the provided candidates.\n"
            "- Do NOT invent titles.\n"
            "- Respect hard constraints like episode count if mentioned.\n"
            "- If episode count is unknown, say it's unknown instead of guessing.\n"
        )

        user = (
            f"User request:\n{query}\n\n"
            f"Candidates (retrieved from database):\n{context}\n\n"
            f"Task:\n"
            f"- Pick the best {n_recs} candidates.\n"
            f"- Rank them from best to worst.\n"
            f"- For each, give ONE short reason grounded in synopsis/genres.\n"
            f"Output format:\n"
            f"1) Title — reason\n2) Title — reason\n..."
        )

        resp = self.llm.chat.completions.create(
            model=LMSTUDIO_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_tokens=300, 
        )
        return resp.choices[0].message.content
    
if __name__ == "__main__":
    rag = AnimeRAG()

    q = "Recommend dark psychological mystery anime under 26 episodes, minimal comedy."
    print(rag.recommend(q))