import faiss
import streamlit as st
from transformers import AutoTokenizer, AutoModel
import torch
import joblib
import pandas as pd

# Загрузка сохраненных данных и индекса
text_embeddings = joblib.load('release_3/mail_embeddings.joblib')
index = faiss.read_index('release_3/mail_faiss_index.index')

# Датасет
df = pd.read_csv('release_3/clean_mail_movie.csv')
titles = df['movie_title'].tolist()
images = df['image_url'].tolist()
descr = df['description'].tolist()
links = df['page_url'].tolist()

# Загрузка модели и токенизатора
tokenizer = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny2")
model = AutoModel.from_pretrained("cointegrated/rubert-tiny2")

# Функция для векторизации текста
def embed_bert_cls(text, model, tokenizer):
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt', max_length=1024)
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
        distances, top_indices = index.search(user_embedding, num_recs)  # Здесь добавляем переменную distances
        
        st.write(f"Рекомендованные фильмы (Топ-{num_recs}):")
        
        for i, index in enumerate(top_indices[0]):
            col1, col2, col3 = st.columns([1, 4, 1])  # Добавляем ещё одну колонку для уверенности
            
            with col1:
                try:
                    st.image(images[index])  # Загружаем обложку фильма
                except Exception as e:
                    st.write(f"Could not display image at index {index}. Error: {e}")  # Это на случай отсутствия обложки

            with col2:
                st.markdown(f"[{titles[index]}]({links[index]})")  # Название фильма сделано кликабельным
                st.write(descr[index])  # Выводим описание фильма
            
            with col3:
                st.write(f"Уверенность: {(1/(1-distances[0][i]))*100:.1f}%")  # Выводим уверенность