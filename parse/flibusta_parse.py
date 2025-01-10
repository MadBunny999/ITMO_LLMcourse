import mysql.connector
from datetime import timedelta
import csv
import os
import time
import json
from bs4 import BeautifulSoup
import pymysql
import tqdm

import requests
from bs4 import BeautifulSoup
import os


# Подключение к базе данных MySQL
mysql_config = {
    "host": "localhost",
    "user": "root",
    "password": "root",
    "database": "library",
    "charset": "utf8mb4"
}

conn = pymysql.connect(**mysql_config)
cursor = conn.cursor()

# Директория для сохранения файлов
output_dir = 'books/flibusta'
os.makedirs(output_dir, exist_ok=True)

# Функция для очистки аннотаций
def clean_annotation(annotation):
    soup = BeautifulSoup(annotation, 'html.parser')
    cleaned_text = soup.get_text(separator=' ', strip=True)
    cleaned_text = ' '.join(cleaned_text.split())
    return cleaned_text

# Функция для предзагрузки жанров
def preload_genres(cursor):
    genre_map = {}
    try:
        query = """
            SELECT lg.BookId, gl.GenreDesc
            FROM libgenre lg
            JOIN libgenrelist gl ON lg.GenreId = gl.GenreId
        """
        cursor.execute(query)
        results = cursor.fetchall()

        for book_id, genre_desc in results:
            if book_id not in genre_map:
                genre_map[book_id] = []
            genre_map[book_id].append(genre_desc)

        return genre_map
    except Exception as e:
        print(f"Ошибка при загрузке жанров: {e}")
        return {}

genre_map = preload_genres(cursor)

# Функция для обработки пакета книг
def process_batch(book_ids, meta_data_dict, current_batch, total_batches, start_time, total_books):
    try:
        # Формируем запрос для получения аннотаций для всех книг в батче
        format_strings = ','.join(['%s'] * len(book_ids))
        query = f"SELECT BookId, Body FROM libbannotations WHERE BookId IN ({format_strings})"
        cursor.execute(query, book_ids)
        results = cursor.fetchall()

        processed_count = 0

        for book_id, body in results:
            if not body:
                print(f"Книга ID {book_id}: Аннотация отсутствует.")
                continue

            cleaned_annotation = clean_annotation(body)
            meta_data = meta_data_dict[str(book_id)]

            # Добавляем жанры в метаинформацию
            genres = genre_map.get(int(book_id), ["Жанр не указан"])
            meta_data["Genres"] = genres

            # Сохраняем аннотацию в текстовый файл
            txt_file_path = os.path.join(output_dir, f"{book_id}.txt")
            with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(cleaned_annotation)

            # Сохраняем метаинформацию в JSON-файл
            json_file_path = os.path.join(output_dir, f"{book_id}.json")
            with open(json_file_path, 'w', encoding='utf-8') as json_file:
                json.dump(meta_data, json_file, ensure_ascii=False, indent=4)

            processed_count += 1

        # Оценка времени
        elapsed_time = time.time() - start_time
        avg_time_per_book = elapsed_time / (current_batch * len(book_ids))
        remaining_batches = total_batches - current_batch
        estimated_remaining_time = avg_time_per_book * remaining_batches * len(book_ids)

        print(f"Пакет {current_batch}/{total_batches} обработан. "
              f"Обработано книг: {processed_count}/{total_books}. "
              f"Прошло: {str(timedelta(seconds=int(elapsed_time)))}, "
              f"Осталось: {str(timedelta(seconds=int(estimated_remaining_time)))}")

    except Exception as e:
        print(f"Ошибка при обработке пакета: {e}")

# Основная функция для пакетной обработки
def parse(batch_size=500):
    catalog_file = 'books/catalog/catalog.txt'

    # Счётчики для статистики
    start_time = time.time()
    
    with open(catalog_file, 'r', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter=';')
        rows = [row for row in reader if len(row) >= 9 and row[8].isdigit()]
        total_books = len(rows)

        # Создаём словарь метаданных для всех книг
        meta_data_dict = {}
        book_ids = []

        for row in rows:
            book_id = row[8].strip()
            book_ids.append(book_id)

            meta_data_dict[book_id] = {
                "Last Name": row[0].strip(),
                "First Name": row[1].strip(),
                "Middle Name": row[2].strip(),
                "Title": row[3].strip(),
                "Subtitle": row[4].strip(),
                "Language": row[5].strip(),
                "Year": row[6].strip(),
                "Series": row[7].strip(),
                "ID": book_id
            }

        # Обрабатываем книги пакетами
        total_batches = (total_books + batch_size - 1) // batch_size

        for current_batch, i in enumerate(range(0, total_books, batch_size), start=1):
            batch_ids = book_ids[i:i + batch_size]
            process_batch(batch_ids, meta_data_dict, current_batch, total_batches, start_time, total_books)

    conn.close()
    elapsed_time = time.time() - start_time
    print(f"\nОбработка завершена. Всего обработано книг: {total_books}. Время: {str(timedelta(seconds=int(elapsed_time)))}")


def parse_covers():
        directory = "books\flibusta"
        txt_files = [f for f in os.listdir(directory) if f.endswith(".txt")]

        start_time = time.time()
        
        for filename in tqdm(txt_files, desc="Загрузка аннотаций и метаданных", unit="файл"):
            book_id = filename.replace(".txt", "")
            get_book_cover(book_id, save_directory='books/covers/')
        
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"\nЗагрузка завершена. Всего загружено {len(texts)} аннотаций.")
        print(f"Затраченное время: {elapsed_time:.2f} секунд.")





def get_book_cover(book_id, save_directory="covers"):
    """
    Парсит URL обложки книги с Flibusta по её ID и сохраняет обложку в файл.
    
    :param book_id: ID книги (например, 571627)
    :param save_directory: Директория для сохранения обложек
    :return: Путь к сохранённому файлу или None, если обложка не найдена
    """
    base_url = f"https://flibusta.site/b/{book_id}"
    
    try:
        # Запрос страницы книги
        response = requests.get(base_url)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при запросе страницы: {e}")
        return None

    # Парсим HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Ищем тег <img> с атрибутом alt="Cover image"
    cover_img = soup.find('img', alt="Cover image")
    if not cover_img:
        print(f"Обложка для книги с ID {book_id} не найдена.")
        return None

    # Формируем полный URL обложки
    cover_url = cover_img.get('src')
    if cover_url and not cover_url.startswith("http"):
        cover_url = f"https://flibusta.site{cover_url}"

    try:
        # Запрос обложки
        cover_response = requests.get(cover_url)
        cover_response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при загрузке обложки: {e}")
        return None

    # Создаём директорию для сохранения, если её нет
    os.makedirs(save_directory, exist_ok=True)

    # Определяем имя файла и сохраняем обложку
    filename = os.path.join(save_directory, f"{book_id}.jpg")
    with open(filename, 'wb') as file:
        file.write(cover_response.content)
    
    print(f"Обложка для книги с ID {book_id} сохранена в '{filename}'.")
    return filename


# # Пример использования:
# book_id = 571627  # ID книги
# saved_file = get_book_cover(book_id)

# if saved_file:
#     print(f"Обложка сохранена: {saved_file}")
# else:
#     print("Обложка не найдена.")



# Запуск основной функции
#parse(batch_size=500)
parse_covers()

