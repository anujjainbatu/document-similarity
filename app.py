from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

documents = [
    #Information about indian cricketers
    "Sachin Tendulkar is a former Indian cricketer and captain, widely regarded as one of the greatest batsmen in the history of cricket.",
    "Virat Kohli is an Indian cricketer and the current captain of the India national team in all formats.",
    "MS Dhoni is a former Indian cricketer and captain, known for his calm demeanor and excellent leadership skills.",
    "Rohit Sharma is an Indian cricketer and the current vice-captain of the India national team in limited-overs formats.",
    "Kapil Dev is a former Indian cricketer and captain, who led India to its first World Cup victory in 1983."
]

query = "Tell me about Sachin Tendulkar."

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)), key=lambda x: x[1])[-1]

print(query)
print(documents[index])
print("similarity score is",score)