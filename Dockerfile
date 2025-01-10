FROM python:3.11.9

# Установить зависимости
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Скопировать проект
COPY . .


# Установить шрифты с поддержкой эмодзи
RUN apt-get update && apt-get install -y \
    fonts-noto-color-emoji \
    && apt-get clean

    
# Установить порт и команду запуска
EXPOSE 8080
CMD ["streamlit", "run", "app.py","--server.port", "8080", "host", "0.0.0.0"]
