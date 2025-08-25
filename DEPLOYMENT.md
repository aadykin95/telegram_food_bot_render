# 🚀 Развёртывание бота с ChatGPT

## Быстрый старт

### 1. Получите API ключи

**OpenAI API:**
- Зарегистрируйтесь на [platform.openai.com](https://platform.openai.com/)
- Создайте API ключ в разделе API Keys
- Скопируйте ключ (начинается с `sk-`)

**Telegram Bot:**
- Напишите [@BotFather](https://t.me/BotFather) в Telegram
- Создайте нового бота командой `/newbot`
- Получите токен бота

### 2. Настройка Render

1. Зарегистрируйтесь на [render.com](https://render.com)
2. Создайте новый **Web Service**
3. Подключите ваш GitHub репозиторий
4. Настройте переменные окружения:

```bash
TOKEN=ваш_telegram_токен
OPENAI_API_KEY=ваш_openai_ключ
GCP_CREDENTIALS_JSON={"type": "service_account", ...}
SPREADSHEET_NAME=FoodLog
SHEET_NAME=log
```

### 3. Настройка Google Sheets

1. Создайте Google Sheets документ
2. Настройте сервисный аккаунт Google Cloud (только для Sheets, не для Vision)
3. Добавьте JSON credentials в переменную `GCP_CREDENTIALS_JSON`

### 4. Запуск

После настройки всех переменных:
1. Render автоматически установит зависимости из `requirements.txt`
2. Бот запустится и будет доступен в Telegram
3. Проверьте работу командой `/start`

## 🔧 Переменные окружения

| Переменная | Описание | Пример |
|------------|----------|---------|
| `TOKEN` | Telegram Bot Token | `1234567890:ABCdefGHIjklMNOpqrsTUVwxyz` |
| `OPENAI_API_KEY` | OpenAI API Key | `sk-1234567890abcdef...` |
| `GCP_CREDENTIALS_JSON` | Google Cloud credentials | `{"type": "service_account", ...}` |
| `SPREADSHEET_NAME` | Название Google Sheets | `FoodLog` |
| `SHEET_NAME` | Название листа | `log` |
| `PROXY_URL` | Прокси (опционально) | `http://proxy:8080` |

## 📊 Структура Google Sheets

Бот создаст таблицу со следующими колонками:
- Дата
- Время  
- User ID
- Username
- Продукт
- Вес (г)
- Калории
- Белки (г)
- Жиры (г)
- Углеводы (г)

## 🧪 Тестирование

После запуска протестируйте бота:

1. **Текстовый ввод:**
   ```
   яблоко 150г
   куриная грудка 200г
   салат цезарь
   ```

2. **Фото:**
   - Отправьте фото еды
   - Бот распознает продукты
   - Уточните количество

3. **Отчёты:**
   ```
   /report today
   /report week
   /report month
   ```

## 💰 Стоимость

- **OpenAI API**: ~$0.002 за 1K токенов (текст) + ~$0.01 за изображение
- **Render**: Бесплатный план (750 часов/месяц)
- **Google Cloud**: Бесплатный план (только для Sheets)

## 🐛 Устранение неполадок

### Бот не отвечает
- Проверьте переменные окружения
- Посмотрите логи в Render Dashboard
- Убедитесь, что API ключи корректны

### Ошибки ChatGPT
- Проверьте баланс OpenAI аккаунта
- Убедитесь, что API ключ активен
- Проверьте лимиты запросов

### Проблемы с Google Sheets
- Проверьте права доступа сервисного аккаунта
- Убедитесь, что таблица существует и доступна
