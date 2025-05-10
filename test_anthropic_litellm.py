import os
from litellm import completion
import sys

# Отримати API ключ з аргументів командного рядка або .env файлу
if len(sys.argv) > 1:
    api_key = sys.argv[1]
    print(f"Використовуємо ключ API з аргументів командного рядка")
else:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    print(f"Використовуємо ключ API з змінної середовища ANTHROPIC_API_KEY")

# Показуємо перші та останні 4 символи ключа для діагностики
key_preview = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "Ключ занадто короткий"
print(f"Ключ Anthropic API (перші та останні 4 символи): {key_preview}")

try:
    print("Надсилаємо запит до Anthropic API через LiteLLM...")
    response = completion(
        model="anthropic/claude-3-haiku-20240307",
        messages=[{"role": "user", "content": "Привіт! Розкажи, як у тебе справи, коротко."}],
        api_key=api_key
    )
    
    print("\nУспішна відповідь:")
    print(response['choices'][0]['message']['content'])
    
except Exception as e:
    print(f"\nВиникла помилка: {str(e)}")
    
print("\nТестування завершено.")
