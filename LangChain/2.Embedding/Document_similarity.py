from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

chess_documents = [
    "Magnus Carlsen is a Norwegian chess grandmaster known for his deep understanding and dominance in modern chess.",
    "Viswanathan Anand, a former world champion from India, is known for his rapid calculation and sportsmanship.",
    "Garry Kasparov is a legendary Russian grandmaster famous for his intense rivalry with computers and his aggressive play.",
    "Hikaru Nakamura is an American grandmaster known for his blitz skills and strong online presence.",
    "Judit Polgár is considered the greatest female chess player in history, known for defeating several world champions."
]


query = 'tell me about Magnus Carlsen '

doc_embeddings = embedding.embed_documents(chess_documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]
index, score = sorted(list(enumerate(scores)), key = lambda x : x[1])[-1]

print(query)
print(chess_documents[index])
print("similarity score is:", score)