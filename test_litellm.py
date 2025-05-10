"""
Test script for LiteLLM with Anthropic
"""

import os
import json
from dotenv import load_dotenv
import litellm

# Завантажуємо змінні середовища з .env файлу
load_dotenv()

# Отримуємо API ключ Anthropic з середовища
api_key = os.environ.get("ANTHROPIC_API_KEY")

# Перевіряємо чи ключ доступний
if not api_key:
    print("API ключ Anthropic не знайдено в змінних середовища!")
    exit(1)

print(f"Використовуємо ключ API (перші та останні 4 символи): {api_key[:4]}...{api_key[-4:]}")

try:
    # Налаштування litellm з детальним логуванням
    litellm.set_verbose = True
    
    # Специфічний формат моделі для Anthropic через litellm
    model = "anthropic/claude-3-haiku-20240307"
    
    print(f"Надсилаємо запит до {model} через LiteLLM...")
    
    # Виконуємо запит
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": "Привіт, як справи?"}],
        api_key=api_key,
        max_tokens=100
    )
    
    # Виводимо результат
    print("\nУспішна відповідь:")
    print(json.dumps(response, indent=2, ensure_ascii=False, default=str))
    
except Exception as e:
    print(f"\nПомилка: {e}")
