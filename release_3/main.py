import numpy as np
import faiss
import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch
import joblib
import pandas as pd

# Загрузка сохраненных данных и индекса
text_embeddings = joblib.load('release_3/text_embeddings.joblib')

# Создание FAISS индекса после определения text_embeddings
dimension = text_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(text_embeddings.astype('float32'))

# Датасет
df = pd.read_csv('release_3/movies_filtered.csv')
titles = df['movie_title'].tolist()
images = df['image_url'].tolist()
descr = df['description'].tolist()
links = df['page_url'].tolist()


# Загрузка модели и токенизатора
tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")
tokenizer.model_max_length

# Функция для векторизации текста
def embed_bert_cls(text, model, tokenizer):
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(model.device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].cpu().numpy()    

# Streamlit интерфейс
st.title("Умный поиск фильмов")

user_input = st.text_area("Введите описание фильма:")
num_recs = st.selectbox("Количество рекомендаций:", [1, 3, 5, 10])

if st.button("Найти"):
    if user_input:
        user_embedding = embed_bert_cls(user_input, model, tokenizer).astype('float32').reshape(1, -1)
        _, top_indices = index.search(user_embedding, num_recs)
        
        st.write(f"Рекомендованные фильмы (Топ-{num_recs}):")
        
        for index in top_indices[0]:
            col1, col2 = st.columns([1, 4])
            
            with col1:
                st.image(images[index])
            
            with col2:
                st.write(titles[index])
                st.write(descr[index])