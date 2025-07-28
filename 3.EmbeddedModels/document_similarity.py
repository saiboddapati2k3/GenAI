from langchain_huggingface import HuggingFaceEndpointEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from dotenv import load_dotenv
load_dotenv()


embedding_model = HuggingFaceEndpointEmbeddings(
    repo_id ="sentence-transformers/all-MiniLM-L6-v2"
)



documents = [
    "The capital of France is Paris.",
    "Python is a popular programming language.",
    "Mount Everest is the tallest mountain.",
    "The Mona Lisa was painted by Leonardo da Vinci.",
    "The Taj Mahal is located in Agra, India.",
    "RishabPant scored a half century in the ongoing test match vs England despite being injured."
]

query = "fitness"

doc_embeddings = embedding_model.embed_documents(documents)


query_embeddings = embedding_model.embed_query(query)


similarities = cosine_similarity([query_embeddings], doc_embeddings)[0]


most_similar_index = np.argmax(similarities)
most_similar_doc = documents[most_similar_index]


print(f"\n🔍 Query: {query}")
print(f"✅ Most Similar Document: {most_similar_doc}")
print(f"📈 Similarity Score: {similarities[most_similar_index]:.4f}")
