import os
from litellm import completion
import sys

# Отримати API ключ з аргументів командного рядка або .env файлу
if len(sys.argv) > 1:
    api_key = sys.argv[1]
    print(f"Використовуємо ключ API з аргументів командного рядка")
else:
    api_key = os.environ.get("OPENAI_API_KEY")
    print(f"Використовуємо ключ API з змінної середовища OPENAI_API_KEY")

# Перевіряємо формат ключа
if api_key.startswith("sk-proj-"):
    print("УВАГА: Ви використовуєте проектний ключ OpenAI (sk-proj-...).")
    print("Цей скрипт призначений для тестування стандартних ключів OpenAI (sk-...).")
    print("Продовжуємо, але можуть виникнути помилки аутентифікації.")
elif api_key.startswith("sk-"):
    print("Виявлено стандартний ключ OpenAI (sk-...).")
else:
    print(f"Невідомий формат ключа API, починається з: {api_key[:5]}...")

# Показуємо перші та останні 4 символи ключа для діагностики
key_preview = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "Ключ занадто короткий"
print(f"Ключ API (перші та останні 4 символи): {key_preview}")

try:
    print("Надсилаємо запит до OpenAI API через LiteLLM...")
    response = completion(
        model="openai/gpt-4",
        messages=[{"role": "user", "content": "Привіт! Розкажи, як у тебе справи, коротко."}]
    )
    
    print("\nУспішна відповідь:")
    print(response['choices'][0]['message']['content'])
    
except Exception as e:
    print(f"\nВиникла помилка: {str(e)}")
    
print("\nТестування завершено.")
