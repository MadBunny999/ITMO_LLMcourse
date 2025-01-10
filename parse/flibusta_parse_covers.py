import os
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time


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
        response = requests.get(base_url, timeout=5)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        #print(f"Ошибка при запросе страницы для книги {book_id}: {e}")
        return None

    # Парсим HTML
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Ищем тег <img> с атрибутом alt="Cover image"
    cover_img = soup.find('img', alt="Cover image")
    if not cover_img:
        #print(f"Обложка для книги с ID {book_id} не найдена.")
        return None

    # Формируем полный URL обложки
    cover_url = cover_img.get('src')
    if cover_url and not cover_url.startswith("http"):
        cover_url = f"https://flibusta.site{cover_url}"

    try:
        # Запрос обложки
        cover_response = requests.get(cover_url, timeout=5)
        cover_response.raise_for_status()
    except requests.exceptions.RequestException as e:
        #print(f"Ошибка при загрузке обложки для книги {book_id}: {e}")
        return None

    # Создаём директорию для сохранения, если её нет
    os.makedirs(save_directory, exist_ok=True)

    # Определяем имя файла и сохраняем обложку
    filename = os.path.join(save_directory, f"{book_id}.jpg")
    with open(filename, 'wb') as file:
        file.write(cover_response.content)
    
    #print(f"Обложка для книги с ID {book_id} сохранена в '{filename}'.")
    return filename


def parse_covers(directory="books/flibusta", save_directory="books/covers/", max_workers=20):
    """
    Парсит обложки для книг в указанной директории с использованием многопоточности.
    
    :param directory: Директория с текстовыми файлами книг
    :param save_directory: Директория для сохранения обложек
    :param max_workers: Количество потоков для загрузки
    """
    txt_files = [f for f in os.listdir(directory) if f.endswith(".txt")]
    book_ids = [filename.replace(".txt", "") for filename in txt_files]

    start_time = time.time()

    # Используем многопоточность для ускорения загрузки
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_book = {executor.submit(get_book_cover, book_id, save_directory): book_id for book_id in book_ids}
        
        # Прогресс-бар с учетом выполнения задач
        for future in tqdm(as_completed(future_to_book), total=len(future_to_book), desc="Загрузка обложек"):
            book_id = future_to_book[future]
            try:
                future.result()
            except Exception as e:
                #print(f"Ошибка при обработке книги {book_id}: {e}")
                pass

    end_time = time.time()
    elapsed_time = end_time - start_time

    #print(f"\nЗагрузка завершена. Всего обработано {len(book_ids)} книг.")
    #print(f"Общее время выполнения: {elapsed_time:.2f} секунд.")


# Пример вызова функции
parse_covers(directory="books/flibusta", save_directory="books/covers/", max_workers=20)
