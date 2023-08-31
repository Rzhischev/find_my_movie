import streamlit as st
import pandas as pd
import random

# Заголовок приложения
st.title("Random Movie Descriptions")

# Чтение данных из CSV-файла
@st.cache_data
def load_data():
    return pd.read_csv("ithinker_movies.csv")

data = load_data()

# Получение случайных 10 записей
random_indices = random.sample(range(0, len(data)), 10)
random_rows = data.iloc[random_indices]

# Отображение данных
for index, row in random_rows.iterrows():
    st.write(f"**{row['movie_title']}** - {row['description']}")
