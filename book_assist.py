import os
import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import GigaChat
import numpy as np
from tqdm import tqdm
import time
import re
from keybert import KeyBERT
from langdetect import detect, DetectorFactory, LangDetectException
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore")

DetectorFactory.seed = 0

# Путь к каталогу с книгами
BOOKS_DIR = "books/flibusta"
INDEX_PATH = "faiss_index"


def detect_language(text):
    """Определяет язык текста. Возвращает код языка (например, 'ru', 'en')."""
    try:
        return detect(text)
    except LangDetectException:
        return "en"  # По умолчанию английский, если язык не удалось определить


class BookAssistant:
    def __init__(self, books_dir=BOOKS_DIR, index_path=INDEX_PATH):
        load_dotenv(".env")
        self.books_dir = books_dir
        self.index_path = index_path
        self.giga_key = os.environ.get("SB_AUTH_DATA")
        self.llm = GigaChat(credentials=self.giga_key, model="GigaChat-Pro", timeout=30, verify_ssl_certs=False)
        self.embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        if not self.load_index():
            self.vectorstore = self.create_faiss_index()
    

    def load_index(self):
        if os.path.exists(self.index_path):
            self.vectorstore = FAISS.load_local(self.index_path, self.embedding_model, allow_dangerous_deserialization=True)
            #print(f"FAISS индекс содержит {len(self.vectorstore.index)} векторов")
            return True
        return None

    def extract_keywords_with_keybert(self, kw_model, text, top_n=5):
        """Извлекает ключевые слова из текста с помощью KeyBERT."""
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=top_n)
        return [kw[0] for kw in keywords]

    def batch_extract_keywords(self, texts, kw_model, top_n=5, max_workers=4):
        """Извлекает ключевые слова из списка текстов параллельно."""
        keywords_list = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.extract_keywords_with_keybert, kw_model, text, top_n): text for text in texts}

            for future in tqdm(as_completed(futures), total=len(futures), desc="Извлечение ключевых слов"):
                try:
                    keywords = future.result()
                except Exception as e:
                    keywords = []
                keywords_list.append(", ".join(keywords))

        return keywords_list

    def create_faiss_index(self):
        # Загрузка аннотаций и метаданных
        texts, metadatas = self.load_annotations_and_metadata(self.books_dir)
        # Инициализация модели KeyBERT
        kw_model = KeyBERT(model="paraphrase-multilingual-MiniLM-L12-v2")
        vectorstore = self.create_faiss_index_in_batches(texts, metadatas, kw_model, batch_size=10000, index_save_path=self.index_path)
        vectorstore.save_local(self.index_path)
        print("Создан новый индекс и сохранён.")        

    def create_faiss_index_in_batches(self, texts, metadatas, kw_model, batch_size=10000, index_save_path="faiss_index"):
        """Создание FAISS индекса по частям и сохранение промежуточных результатов с учётом ключевых слов."""
        vectorstore = None
        print("Создаётся новый индекс.")

        total_texts = len(texts)
        for i in range(0, total_texts, batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]

            # Параллельное извлечение ключевых слов для батча
            keywords_list = self.batch_extract_keywords(batch_texts, kw_model, top_n=5, max_workers=6)

            # Создание текстов с учётом ключевых слов
            enriched_texts = [
                f"Название: {metadata['title']}. "
                f"Автор: {metadata['author']}. "
                f"Жанр: {metadata['genres']}. "
                f"Аннотация: {text}. "
                f"Ключевые слова: {keywords}"
                for text, metadata, keywords in zip(batch_texts, batch_metadatas, keywords_list)
            ]

            print(f"\nВекторизация батча {i // batch_size + 1}/{(total_texts + batch_size - 1) // batch_size}...")

            if vectorstore is None:
                vectorstore = FAISS.from_texts(enriched_texts, self.embedding_model, metadatas=batch_metadatas)
            else:
                new_vectorstore = FAISS.from_texts(enriched_texts, self.embedding_model, metadatas=batch_metadatas)
                vectorstore.merge_from(new_vectorstore)

            # Сохранение промежуточного результата
            vectorstore.save_local(index_save_path)
            print(f"Промежуточный индекс сохранён в '{index_save_path}'.")

        print("\nИндекс успешно создан и сохранён.")
        return vectorstore

    # Функция для загрузки аннотаций и метаданных
    def load_annotations_and_metadata(self, directory):
        """Загрузка аннотаций и метаданных из файлов с прогрессом выполнения и оценкой времени."""
        texts = []
        metadatas = []
        
        txt_files = [f for f in os.listdir(directory) if f.endswith(".txt")]

        start_time = time.time()
        
        for filename in tqdm(txt_files, desc="Загрузка аннотаций и метаданных", unit="файл"):
            book_id = filename.replace(".txt", "")
            annotation_path = os.path.join(directory, f"{book_id}.txt")
            metadata_path = os.path.join(directory, f"{book_id}.json")
            
            if os.path.exists(annotation_path) and os.path.exists(metadata_path):
                with open(annotation_path, "r", encoding="utf-8") as f:
                    annotation = f.read()
                    texts.append(annotation)
                
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    metadatas.append({
                        "title": metadata.get("Title", "Неизвестное название"),
                        "author": f"{metadata.get('Last Name', '')} {metadata.get('First Name', '')} {metadata.get('Middle Name', '')}".strip(),
                        "year": metadata.get("Year", "Неизвестный год"),
                        "language": metadata.get("Language", "Неизвестный язык"),
                        "genres": ', '.join(metadata.get("Genres", ["Жанр не указан"])),
                        "id": book_id
                    })
        
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"\nЗагрузка завершена. Всего загружено {len(texts)} аннотаций.")
        print(f"Затраченное время: {elapsed_time:.2f} секунд.")
        
        return texts, metadatas

    def detect_genre_from_query(self, query):
        """Попытка определить жанр из текста запроса."""

        genre_keywords = {
            "детектив": [
                "детектив", "расследование", "преступление", "сыщик", "убийство", 
                "криминал", "следствие", "интрига", "улики", "алиби", "тайна", 
                "прокурор", "суд", "адвокат", "подозреваемый", "шпион"
            ],
            "фантастика": [
                "фантастика", "научная фантастика", "футуризм", "космос", "роботы", 
                "инопланетяне", "киберпанк", "технологии", 
                "антиутопия", "утопия", "временные петли", "телекинез", 
                "апокалипсис", "постапокалипсис", "параллельные миры", "космические битвы"
            ],
            "фэнтези": [
                "фэнтези", "магия", "эльфы", "гномы", "волшебство", "чародеи", 
                "заклинания", "драконы", "мечи", "средневековье", "королевства", 
                "пророчества", "тролли", "вампиры", "оборотни", "темные силы", 
                "артефакты", "магические существа", "эпические сражения"
            ],
            "роман": [
                "роман", "любовь", "отношения", "драма", "лавстори", "страсть", 
                "влюбленность", "развод", "предательство", "семья", "любовный треугольник", 
                "свадьба", "первая любовь", "разлука", "эмоции", "чувства", 
                "поиск счастья"
            ],
            "приключения": [
                "приключения", "путешествие", "исследование", "авантюра", 
                "экспедиция", "выживание", "неизведанные земли", "пираты", 
                "сокровища", "опасности"
            ],
            "история": [
                "история", "исторический", "прошлое", "война", "революция", 
                "средневековье", "древний мир", "империя", "монархи", "парады", 
                "события", "культура", "исторические личности", "колониализм", 
                "регенты", "вековые традиции", "битвы"
            ],
            "ужасы": [
                "ужасы", "хоррор", "страх", "кошмары", "демоны", "проклятие", 
                "призраки", "паранормальные явления", "темнота", "дом с привидениями", 
                "монстры", "зомби", "кровь"
            ],
            "биография": [
                "биография", "жизнь", "история личности", "воспоминания", 
                "автобиография", "мемуары", "личностный рост", "творчество", 
                "карьера", "успех"
            ],
            "научпоп": [
                "научпоп", "наука", "технологии", "физика", "астрономия", 
                "биология", "психология", "медицина", "эволюция", "космос", 
                "новые открытия", "популярная наука", "эксперименты", "исследования", 
                "техника", "гипотезы", "экология"
            ],
            "философия": [
                "философия", "размышления", "мудрость", "экзистенциальность", 
                "жизнь", "смысл жизни", "логика", "религия", 
                "метафизика", "истина", "вопросы", "самопознание", 
                "духовность", "идеологии", "традиции", "этика"
            ]
        }

        query_lower = query.lower()
        for genre, keywords in genre_keywords.items():
            if any(re.search(rf"\b{kw}\b", query_lower) for kw in keywords):
                return genre.capitalize()
        return None

    def generate_keywords_with_llm(self, query):
        """Генерирует ключевые слова для запроса с помощью LLM."""
        prompt = (
            f"На основе следующего запроса пользователя:\n'{query}'\n"
            f"Сгенерируй 5-7 ключевых слов или фраз, которые лучше всего описывают запрос. "
            f"Ключевые слова должны быть конкретными и релевантными."
        )
        try:
            response = self.llm(prompt)
            keywords = response.strip().split(", ")
            return keywords
        except Exception as e:
            print(f"Ошибка при генерации ключевых слов: {e}")
            return []

    def filter_keywords(self, keywords):
        """Фильтрует ключевые слова, удаляя лишние или бессмысленные фразы."""
        filtered = [kw for kw in keywords if len(kw) > 3 and "..." not in kw]
        return filtered

    def enrich_query_with_llm(query, genre=None):
        """Обогащает запрос с помощью LLM и добавления контекста."""
        context = "Ты — ассистент по подбору книг. На основе запроса пользователя предложи уточненный и расширенный вариант запроса для поиска подходящих книг."
        if genre:
            context += f" Убедись, что запрос связан с жанром '{genre}'."
        
        prompt = f"{context}\n\nЗапрос пользователя: {query}\n\nУточненный запрос:"
        
        try:
            enriched_query = llm(prompt)
            return enriched_query.strip()
        except Exception as e:
            print(f"Ошибка при обогащении запроса: {e}")
            return query


    def rerank_documents(self, query, results, embedding_model, top_k=3):
        """Реранкинг документов на основе косинусного сходства."""
        # Векторизация запроса
        query_vector = np.array(embedding_model.embed_query(query))

        # Вычисление схожести для каждого документа
        scored_results = []
        for doc in results:
            doc_vector = np.array(embedding_model.embed_query(doc.page_content))
            similarity = np.dot(doc_vector, query_vector) / (np.linalg.norm(doc_vector) * np.linalg.norm(query_vector))
            scored_results.append((doc, similarity))

        # Сортировка по убыванию оценки схожести
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Ограничение результатов до top_k
        return scored_results[:top_k]

    # Функция для поиска книг
    def recommend_book(self, query, max_results=3, relevance_threshold=0.6):
        # Генерация ключевых слов из запроса
        #enriched_query = enrich_query_with_keywords(query)
        print("Исходный запрос: ", query)

        # Определение жанра из запроса
        genre_hint = self.detect_genre_from_query(query)
        if genre_hint:
            print(f"Определённый жанр: {genre_hint}")
        else:
            print("Определённый жанр: Не определён")
        
        #enriched_query = enrich_query_with_llm(query, genre_hint)
        #print(f"Обогащённый запрос: {enriched_query}")
        #keywords = generate_keywords_with_llm(user_query)

        keywords = self.generate_keywords_with_llm(query)
        enriched_query = f"\n{query}. Ключевые слова:\n {', '.join(keywords)}"

        print(f"Обогащённый запрос: {enriched_query}")

        # Использование ретривера для первичного поиска
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 100})
        results = retriever.get_relevant_documents(enriched_query)

        if not results:
            return "Извините, не удалось найти подходящих книг по вашему запросу в базе данных."

        # Жёсткая фильтрация по жанру, если жанр определён
        if genre_hint:
            genre_filtered_results = [doc for doc in results if genre_hint.lower() in doc.metadata['genres'].lower()]
            if genre_filtered_results:
                results = genre_filtered_results

        # Реранкинг результатов
        reranked_results = self.rerank_documents(enriched_query, results, self.embedding_model, top_k=max_results * 2)

        # Фильтрация результатов по порогу релевантности
        scored_results = [(doc, score) for doc, score in reranked_results if score >= relevance_threshold]

        if not scored_results:
            return "Извините, не удалось найти релевантных книг по вашему запросу."

        # Ограничиваем количество результатов до max_results
        top_results = scored_results[:max_results]

        # Формируем информацию о найденных книгах
        formatted_results = []
        for doc, score in top_results:
            book_info = {
                "id": doc.metadata['id'],
                "title": doc.metadata['title'],
                "author": doc.metadata['author'] or "Неизвестный автор",
                "year": doc.metadata['year'] or "Неизвестный год",
                "genres": doc.metadata['genres'] or "Жанр не указан",
                "annotation": doc.page_content,
                "link": f"https://flibusta.site/b/{doc.metadata['id']}",
                "score": f"{score:.2f}"
            }
            formatted_results.append(book_info)

        return formatted_results