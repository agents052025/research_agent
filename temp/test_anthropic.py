"""
Тестовий скрипт для перевірки API Anthropic
"""

import os
import requests
import json
from dotenv import load_dotenv

# Завантажуємо змінні середовища з .env файлу
load_dotenv()

# Отримуємо API ключ Anthropic з середовища
api_key = os.environ.get("ANTHROPIC_API_KEY")

# Перевіряємо чи ключ доступний
if not api_key:
    print("API ключ Anthropic не знайдено в змінних середовища!")
    exit(1)

print(f"Використовуємо ключ API (перші та останні 4 символи): {api_key[:4]}...{api_key[-4:]}")

# Налаштування запиту
headers = {
    "x-api-key": api_key,
    "anthropic-version": "2023-06-01",
    "content-type": "application/json"
}

# Дані запиту
data = {
    "model": "claude-3-haiku-20240307",
    "max_tokens": 100,
    "messages": [
        {"role": "user", "content": "Привіт, як справи?"}
    ]
}

# Виконуємо запит
print("Надсилаємо запит до Anthropic API...")
response = requests.post(
    "https://api.anthropic.com/v1/messages",
    headers=headers,
    json=data
)

# Виводимо результат
print(f"Статус відповіді: {response.status_code}")
print(f"Заголовки відповіді: {response.headers}")

if response.status_code == 200:
    result = response.json()
    print("\nУспішна відповідь:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
else:
    print("\nПомилка відповіді:")
    print(response.text)
