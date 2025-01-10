import streamlit as st
from dotenv import load_dotenv
import os
import requests
from urllib.parse import quote

import re

from book_assist import BookAssistant


# Загружаем ключ API из переменной окружения
load_dotenv(".env")
#GOOGLE_BOOKS_API_KEY = os.environ.get("GOOGLE_BOOKS_API_KEY")

NO_COVER_PLACEHOLDER_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ac/No_image_available.svg/120px-No_image_available.svg.png"


import requests

# URL для доступа к обложкам в S3-бакете
S3_BUCKET_URL = "https://storage.yandexcloud.net/book-covers"

def get_book_cover(book_id):
    """
    Получает URL обложки книги из S3-бакета.
    :param book_id: ID книги
    :return: URL обложки или URL заглушки, если обложка не найдена
    """
    # Формируем URL обложки
    cover_url = f"{S3_BUCKET_URL}/{book_id}.jpg"
    
    # Проверяем существование обложки
    if check_cover_exists(cover_url):
        return cover_url

    # Если обложка не найдена, возвращаем заглушку
    return NO_COVER_PLACEHOLDER_URL

def check_cover_exists(cover_url):
    """
    Проверяет существование обложки в S3-бакете.
    :param cover_url: URL обложки
    :return: True, если обложка существует, иначе False
    """
    try:
        response = requests.head(cover_url)
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"Ошибка проверки обложки: {e}")
        return False




# Инициализация BookAssistant
assistant = BookAssistant()


# Настройка заголовка страницы
st.set_page_config(page_title="Скромный книжный червь", page_icon="🪱")


st.markdown(
    """
    <style>
    /* Стили для карточек */
    .card {
        background-color: #15161f; /* Темный фон карточек */
        color: #f5f5f5; /* Цвет текста */
        border: 1px solid #444; /* Рамка */
        border-radius: 10px; /* Радиус углов */
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.3); /* Тень */
        min-height: 450px; /* Минимальная высота карточки */
    }

    /* Полное скрытие меню в правом верхнем углу */
    [data-testid="stToolbar"] {
        display: none !important;
    }

    /* Скрытие футера Streamlit */
    footer {
        visibility: hidden !important;
        height: 0;
    }

    /* Скрытие "Made with Streamlit" */
    .viewerBadge_container__1QSob {
        visibility: hidden !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)









# Заголовок
#st.title("Книжный червь 🪱")

# Заголовок с поддержкой шрифта
st.markdown(
    """
    <h1 style="font-family: 'Noto Color Emoji', sans-serif; text-align: left;">
        Скромный книжный червь <img src="https://symbl-world.akamaized.net/i/webp/97/257dc253624102208f2f0d6c0059c6.webp" width="36">
    </h1>
    """,
    unsafe_allow_html=True,
)

# Описание системы
st.write("""
Добро пожаловать, здесь обитает тотемное животное мира книг! Книжный червь знает всё, что вам нужно о книгах и даже больше...
""")

# Поле для ввода запроса
user_query = st.text_input("Спросите червя и погружайтесь в мир книг...", "")

# Кнопка для выполнения поиска
if st.button("Получить рекомендацию"):
    if not user_query.strip():
        st.warning("Пожалуйста, введите запрос.")
    else:
        with st.spinner("Ищем книги..."):
            # Получаем рекомендации
            response = assistant.recommend_book(user_query)
        
        # Отображаем результат
        if isinstance(response, str):
            st.write(response)
        else:
            for i, book in enumerate(response, start=1):
                annotation_match = re.search(r"Аннотация: (.*?)(?:\. Ключевые слова:|$)", book['annotation'])
                annotation = annotation_match.group(1).strip() if annotation_match else ""

                st.markdown(
                    f"""
                    <div class="card">
                        <h3>{i}. {book['title']}</h3>
                        <p><b>Автор:</b> {book['author']}</p>
                        <p><b>Жанр:</b> {book['genres']}</p>
                        <p><b>Год издания:</b> {book['year']}</p>
                        <img src="{get_book_cover(book['id'])}" width="150" style="float: left; margin-right: 15px;">
                        <p><b>Аннотация:</b> {annotation}</p>
                        <a href="{book['link']}" target="_blank">Читать книгу</a>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# Кнопка для нового запроса
if st.button("Новый запрос"):
    st.rerun()