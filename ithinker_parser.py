import requests
from bs4 import BeautifulSoup
import csv
import time  # Для задержки между запросами

# Открываем CSV файл для записи
with open('movies.csv', 'w', newline='', encoding='utf-8') as csvfile:
    fieldnames = ['page_url', 'image_url', 'movie_title', 'description']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Проходим по всем страницам
    for page in range(1, 39):  # 39 — это последняя страница + 1
        # URL страницы
        url = f'https://ithinker.ru/film/page{page}/'

        # Отправляем HTTP запрос и получаем ответ
        r = requests.get(url)
        data = BeautifulSoup(r.text, 'html.parser')

        # Находим все элементы, содержащие информацию о фильмах
        for movie_data in data.find_all('div', class_='uiStandartListElementInside'):
            # Извлекаем базовые данные
            page_url = 'https://ithinker.ru' + movie_data.a['href']
            image_url = "https://ithinker.ru" + movie_data.a.img['src']
            movie_title = movie_data.a.img['alt']

            # Получаем описание с индивидуальной страницы фильма
            r_movie = requests.get(page_url)
            movie_page_data = BeautifulSoup(r_movie.text, 'html.parser')
            description_element = movie_page_data.select_one('body > div.uiStandartTemplateWrapper > div.uiStandartTemplateMain > div > div:nth-child(1) > div.uiFilmMainWrapper > div.uiFilmContent > div:nth-child(5)')

            # Проверяем, найден ли элемент
            if description_element is not None:
                description = description_element.text
            else:
                description = "Description not found"

            # Записываем данные в CSV файл
            writer.writerow({'page_url': page_url, 'image_url': image_url, 'movie_title': movie_title, 'description': description})

            # Опционально: добавляем задержку, чтобы избежать блокировки
            # time.sleep(1)

        print(f"Completed page {page}")  # Опционально: выводим номер обработанной страницы

print("Scraping completed.")
