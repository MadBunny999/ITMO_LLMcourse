# Интеграция с LLM

В данном проекте в качестве LLM-модели использовалась модель **GigaChat-Pro**

### Получение ключевых слов
> Для выделения ключевых слов используется **KeyBERT** на основе модели `paraphrase-multilingual-MiniLM-L12-v2`

### Обогащения пользовательских запросов
> LLM-модель используется для обогащения пользовательских запросов ключевыми словами

### Уточнение контекста запросов
> LLM-модель уточняет контекст запросов
