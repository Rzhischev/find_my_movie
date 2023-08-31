import requests
from bs4 import BeautifulSoup
import csv
import time  # Для задержки между запросами
from tqdm import tqdm

# Открываем CSV файл для записи
with open('movies.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['page_url', 'image_url', 'movie_title', 'description', 'review']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Проходим по всем страницам
    for p in tqdm(range(1, 250+1)):  
        # URL страницы
        url = f'https://kino.mail.ru/cinema/all/?page={p}'

        # Отправляем HTTP запрос и получаем ответ
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')

        # Находим все элементы, содержащие информацию о фильмах
        for movie_data in soup.findAll('div', 'cols__column cols__column_small_percent-100 cols__column_medium_percent-50 cols__column_large_percent-50'):
            # Извлекаем базовые данные
            page_url = 'https://kino.mail.ru' + movie_data.a['href']
            image_url = movie_data.img['src']
            movie_title = movie_data.div.span.a.span.text

            # Получаем описание с индивидуальной страницы фильма
            r_movie = requests.get(page_url)
            movie_page_data = BeautifulSoup(r_movie.text, 'html.parser')
            description_element = movie_page_data.find('span', 'text text_inline text_light_medium text_fixed valign_baseline')
            review_element = movie_page_data.find('div', 'review__item__afisha__descr')

            # Проверяем, найден ли элемент
            if description_element is not None:
                description = description_element.text
            else:
                description = "No description"

            if review_element is not None:
                review = review_element.text
            else:
                review = "No review"

            # Записываем данные в CSV файл
            writer.writerow({'page_url': page_url, 'image_url': image_url, 'movie_title': movie_title, 'description': description, 'review': review})

            # # Добавляем задержку, чтобы избежать блокировки
            # time.sleep(1)

print("Scraping completed.")
