import logging
from datetime import datetime, timedelta
import base64

import gspread
import matplotlib
matplotlib.use("Agg")  # серверный backend
import matplotlib.pyplot as plt
import pandas as pd
import openai

from telegram import InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, CallbackQueryHandler, filters
from telegram.request import HTTPXRequest

# === SETTINGS (Render-ready) ===
import os, http.server, socketserver, threading

from dotenv import load_dotenv
load_dotenv()  # локально подтянет .env; на Render не мешает

# --- Читаем переменные окружения ---
TOKEN = os.environ["TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
SPREADSHEET_NAME = os.environ.get("SPREADSHEET_NAME", "FoodLog")
SHEET_NAME = os.environ.get("SHEET_NAME", "log")
PROXY_URL = os.environ.get("PROXY_URL", "")

# Настройка OpenAI
openai.api_key = OPENAI_API_KEY

# GCP credentials: кладём JSON целиком в переменную и сохраняем во временный файл
GCP_CREDENTIALS_JSON = os.environ["GCP_CREDENTIALS_JSON"]
GCP_CREDENTIALS_FILE = "/tmp/gcp_credentials.json"
if not os.path.exists(GCP_CREDENTIALS_FILE):
    with open(GCP_CREDENTIALS_FILE, "w") as f:
        f.write(GCP_CREDENTIALS_JSON)

# === Google Sheets ===
gc = gspread.service_account(filename=GCP_CREDENTIALS_FILE)
sh = gc.open(SPREADSHEET_NAME)
worksheet = sh.worksheet(SHEET_NAME)

# --- маленький HTTP-сервер для Render Web Service ---
def _start_keepalive_server():
    port = int(os.getenv("PORT", "8080"))  # Render всегда задаёт PORT
    class _Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write("✅ Bot is alive!".encode("utf-8"))
        def log_message(self, format, *args):
            return  # отключаем лишние логи

    def _serve():
        with socketserver.TCPServer(("", port), _Handler) as httpd:
            print(f"Keepalive server listening on port {port}")
            httpd.serve_forever()

    threading.Thread(target=_serve, daemon=True).start()

# === Логирование ===
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.ERROR)
logger = logging.getLogger(__name__)

# === Состояние подтверждений ===
PENDING_CONFIRMATIONS = {}



# === ChatGPT API ===
def get_food_info(query):
    """
    Получает информацию о продукте через ChatGPT API
    """
    prompt = f"""
    Проанализируй следующий продукт питания и верни точную информацию о его пищевой ценности.
    
    Продукт: {query}
    
    Верни ответ в строго определённом JSON формате:
    {{
        "name": "название продукта",
        "grams": число_граммов,
        "calories": число_калорий,
        "protein": число_граммов_белков,
        "fat": число_граммов_жиров,
        "carbs": число_граммов_углеводов
    }}
    
    Важные правила:
    1. Если в запросе указано количество (например "150 г банана"), используй это количество
    2. Если количество не указано, используй стандартную порцию (обычно 100 г)
    3. Все числовые значения должны быть float
    4. Название продукта должно быть на русском языке
    5. Верни ТОЛЬКО JSON, без дополнительного текста
    """
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Ты эксперт по питанию и пищевой ценности продуктов. Твоя задача - точно определить калории, белки, жиры, углеводы и вес продуктов."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=200
        )
        
        content = response.choices[0].message.content.strip()
        
        # Извлекаем JSON из ответа
        import json
        try:
            # Пытаемся найти JSON в ответе
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = content[start_idx:end_idx]
                data = json.loads(json_str)
                
                return {
                    "name": data.get("name", ""),
                    "grams": float(data.get("grams", 0)),
                    "calories": float(data.get("calories", 0)),
                    "protein": float(data.get("protein", 0)),
                    "fat": float(data.get("fat", 0)),
                    "carbs": float(data.get("carbs", 0))
                }
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON: {e}, ответ: {content}")
            return None
            
    except Exception as e:
        logger.error(f"ChatGPT запрос упал: {e}")
        return None
    
    return None

def safe_float(value):
    try:
        return float(str(value).replace(",", "."))
    except (ValueError, TypeError):
        return 0.0

# === Лог в Google Sheets ===
def log_to_sheets(user_id, username, dish, grams="", calories="", protein="", fat="", carbs=""):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    worksheet.append_row([
        date_str, time_str, user_id, username, dish,
        grams, calories, protein, fat, carbs
    ])

# === ChatGPT — распознать еду на фото ===
def detect_food_in_photo(image_bytes, max_items=6):
    """
    Распознаёт продукты питания на фото используя ChatGPT Vision
    """
    # image_bytes должен быть bytes, не bytearray
    if isinstance(image_bytes, bytearray):
        image_bytes = bytes(image_bytes)

    # Кодируем изображение в base64
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    
    try:
        prompt = """
        Проанализируй это изображение и определи, какие продукты питания на нём изображены, а также их примерное количество или вес.
        
        Верни ответ в строго определённом JSON формате:
        {
            "food_items": [
                {"name": "название продукта", "amount": "примерное количество или вес"},
                {"name": "название продукта", "amount": "примерное количество или вес"}
            ]
        }
        
        Правила:
        1. Верни только съедобные продукты питания
        2. Используй русские названия продуктов
        3. Максимум 6 продуктов
        4. Если на фото нет еды, верни пустой массив
        5. Игнорируй посуду, мебель, одежду и другие непищевые предметы
        6. Для количества используй: "1 шт", "2 шт", "150 г", "200 мл", "1 стакан", "1 тарелка" и т.д.
        7. Если количество определить сложно, используй "1 порция"
        8. Верни ТОЛЬКО JSON, без дополнительного текста
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300,
            temperature=0.1
        )
        
        content = response.choices[0].message.content.strip()
        
        # Парсим JSON ответ
        import json
        try:
            # Пытаемся найти JSON в ответе
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = content[start_idx:end_idx]
                data = json.loads(json_str)
                food_items = data.get("food_items", [])
                
                # Формируем список продуктов с количеством
                formatted_items = []
                seen_names = set()
                
                for item in food_items:
                    if isinstance(item, dict):
                        name = item.get("name", "").strip().lower()
                        amount = item.get("amount", "1 порция").strip()
                    else:
                        # Fallback для старого формата
                        name = str(item).strip().lower()
                        amount = "1 порция"
                    
                    if name and name not in seen_names:
                        seen_names.add(name)
                        formatted_items.append(f"{name} {amount}")
                
                return formatted_items[:max_items]
                
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка парсинга JSON от ChatGPT Vision: {e}, ответ: {content}")
            return []
            
    except Exception as e:
        logger.error(f"ChatGPT Vision запрос упал: {e}")
        return []
    
    return []

# === ОТЧЁТЫ ===
async def handle_report(update, context):
    if len(context.args) == 0:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="❗ Используй: /report today | week | month"
        )
        return

    user_id = str(update.effective_user.id)
    period = context.args[0].lower()
    today = datetime.now().date()

    if period == "today":
        period_start = today
    elif period == "week":
        period_start = today - timedelta(days=today.weekday())
    elif period == "month":
        period_start = today.replace(day=1)
    else:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="❗ Неизвестный период. Доступно: today | week | month"
        )
        return

    rows = worksheet.get_all_values()[1:]
    records = []
    for row in rows:
        try:
            row_user_id = row[2].strip()
            if row_user_id != user_id:
                continue
            date_str = row[0].strip()
            grams = row[5].strip()
            cal = row[6].strip(); prot = row[7].strip(); fat = row[8].strip(); carb = row[9].strip()
            if not cal:
                continue
            try:
                date_obj = datetime.strptime(date_str, "%Y-%m-%d").date()
            except ValueError:
                try:
                    date_obj = datetime.strptime(date_str, "%d.%m.%Y").date()
                except ValueError:
                    continue
            records.append({
                "date": date_obj,
                "grams": safe_float(grams),
                "cal": safe_float(cal),
                "prot": safe_float(prot),
                "fat": safe_float(fat),
                "carb": safe_float(carb),
            })
        except Exception:
            continue

    if not records:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="📭 У тебя нет данных за этот период."
        )
        return

    df_all = pd.DataFrame(records)
    df_all["date"] = pd.to_datetime(df_all["date"]).dt.date

    df_sum = df_all[(df_all["date"] >= period_start) & (df_all["date"] <= today)]
    if df_sum.empty:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="📭 У тебя нет данных за выбранный период."
        )
        return

    # Данные для графика
    if period == "today":
        chart_start = today - timedelta(days=29)
        df_chart = df_all[(df_all["date"] >= chart_start) & (df_all["date"] <= today)]
        g = pd.DataFrame(df_chart).assign(date=pd.to_datetime(df_chart["date"])).groupby("date").sum(numeric_only=True).reset_index()
        rng = pd.date_range(start=chart_start, end=today, freq="D")
        full_df = pd.DataFrame({"date": rng})
        grouped = full_df.merge(g, on="date", how="left").fillna(0)
        grouped["label"] = grouped["date"].dt.strftime("%d.%m.%y")

    elif period == "week":
        this_monday = today - timedelta(days=today.weekday())
        chart_start = this_monday - timedelta(weeks=11)
        df_chart = df_all[(df_all["date"] >= chart_start) & (df_all["date"] <= today)]
        df_tmp = pd.DataFrame(df_chart).assign(date=pd.to_datetime(df_chart["date"]))
        iso = df_tmp["date"].dt.isocalendar()
        df_tmp["year"] = iso.year.astype(int)
        df_tmp["week"] = iso.week.astype(int)
        g = df_tmp.groupby(["year", "week"]).sum(numeric_only=True).reset_index()
        rng = pd.date_range(start=chart_start, end=this_monday, freq="W-MON")
        iso_rng = rng.isocalendar()
        full_df = pd.DataFrame({
            "year": iso_rng.year.astype(int),
            "week": iso_rng.week.astype(int),
        }).drop_duplicates()
        grouped = full_df.merge(g, on=["year", "week"], how="left").fillna(0)
        grouped["label"] = grouped["week"].astype(int).astype(str)

    else:  # month
        first_day_cur = today.replace(day=1)
        months_rng = pd.date_range(end=first_day_cur, periods=12, freq="MS")
        chart_start = months_rng.min().date()
        df_chart = df_all[(df_all["date"] >= chart_start) & (df_all["date"] <= today)]
        df_tmp = pd.DataFrame(df_chart).assign(date=pd.to_datetime(df_chart["date"]))
        df_tmp["year"] = df_tmp["date"].dt.year
        df_tmp["month"] = df_tmp["date"].dt.month
        g = df_tmp.groupby(["year", "month"]).sum(numeric_only=True).reset_index()
        full_df = pd.DataFrame({
            "year": months_rng.year,
            "month": months_rng.month,
        })
        grouped = full_df.merge(g, on=["year", "month"], how="left").fillna(0)
        grouped["label"] = grouped.apply(lambda r: f"{int(r['month']):02d}.{int(r['year'])%100:02d}", axis=1)

    # График
    plt.figure(figsize=(9, 5))
    plt.plot(grouped["label"], grouped["grams"], marker="o", linewidth=2, label="Вес ⚖️")
    plt.plot(grouped["label"], grouped["cal"], marker="o", linewidth=2, label="Калории 🔥")
    plt.plot(grouped["label"], grouped["prot"], marker="o", linewidth=2, label="Белки 💪")
    plt.plot(grouped["label"], grouped["fat"], marker="o", linewidth=2, label="Жиры 🥑")
    plt.plot(grouped["label"], grouped["carb"], marker="o", linewidth=2, label="Углеводы 🍞")
    plt.xlabel("Период", fontsize=12)
    plt.ylabel("Количество", fontsize=12)
    plt.title(f"Отчёт за {period}", fontsize=14)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    chart_path = "report_chart.png"
    plt.savefig(chart_path)
    plt.close()

    # Итоги
    total_grams = df_sum["grams"].sum()
    total_cal = df_sum["cal"].sum()
    total_prot = df_sum["prot"].sum()
    total_fat = df_sum["fat"].sum()
    total_carb = df_sum["carb"].sum()

    text_report = (
        f"📊 Отчёт за {period}:\n"
        f"⚖️ Вес: {total_grams:.0f} г\n"
        f"🔥 Калории: {total_cal:.1f}\n"
        f"💪 Белки: {total_prot:.1f} г\n"
        f"🥑 Жиры: {total_fat:.1f} г\n"
        f"🍞 Углеводы: {total_carb:.1f} г"
    )

    await context.bot.send_message(chat_id=update.effective_chat.id, text=text_report)
    await context.bot.send_photo(chat_id=update.effective_chat.id, photo=open(chart_path, "rb"))

# === Обработчики ===
async def handle_text(update, context):
    user_id = update.message.from_user.id
    username = update.message.from_user.username or str(user_id)
    text = update.message.text or ""

    # Ожидание подтверждения по фото
    if user_id in PENDING_CONFIRMATIONS:
        pending_data = PENDING_CONFIRMATIONS.pop(user_id)  # Очищаем состояние
        
        # Если пользователь просто подтвердил (написал "да", "да", "ок" и т.д.)
        if text.lower().strip() in ['да', 'да', 'ок', 'ok', 'yes', 'верно', 'правильно']:
            # Используем уже распознанные продукты
            detected_items = pending_data.get("detected", [])
            if detected_items:
                # Объединяем все продукты в один запрос
                combined_text = ", ".join(detected_items)
                food_info = get_food_info(combined_text)
                
                if food_info:
                    log_to_sheets(
                        user_id, username, combined_text,
                        food_info["grams"], food_info["calories"], food_info["protein"], food_info["fat"], food_info["carbs"]
                    )
                    await update.message.reply_text(
                        f"🍽 {food_info['name'].title()}\n"
                        f"⚖️ {food_info['grams']:.0f}г\n"
                        f"🔥 {food_info['calories']:.0f}ккал\n"
                        f"💪 Б{food_info['protein']:.1f}г\n"
                        f"🥑 Ж{food_info['fat']:.1f}г\n"
                        f"🍞 У{food_info['carbs']:.1f}г\n"
                        f"✅ Записано в журнал!"
                    )
                else:
                    log_to_sheets(user_id, username, combined_text)
                    await update.message.reply_text("✅ Записано в журнал! (калории не найдены)")
            else:
                await update.message.reply_text("❌ Не удалось обработать фото. Попробуйте написать продукты вручную.")
        else:
            # Пользователь написал конкретные продукты - обрабатываем как обычно
            food_info = get_food_info(text)
            
            if food_info:
                log_to_sheets(
                    user_id, username, text,
                    food_info["grams"], food_info["calories"], food_info["protein"], food_info["fat"], food_info["carbs"]
                )
                await update.message.reply_text(
                    f"🍽 {food_info['name'].title()}\n"
                    f"⚖️ {food_info['grams']:.0f}г\n"
                    f"🔥 {food_info['calories']:.0f}ккал\n"
                    f"💪 Б{food_info['protein']:.1f}г\n"
                    f"🥑 Ж{food_info['fat']:.1f}г\n"
                    f"🍞 У{food_info['carbs']:.1f}г\n"
                    f"✅ Записано в журнал!"
                )
            else:
                log_to_sheets(user_id, username, text)
                await update.message.reply_text("✅ Записано в журнал! (калории не найдены)")
        return

    # Обычная текстовая запись
    food_info = get_food_info(text)

    if food_info:
        log_to_sheets(
            user_id, username, text,
            food_info["grams"], food_info["calories"], food_info["protein"], food_info["fat"], food_info["carbs"]
        )
        await update.message.reply_text(
            f"🍽 {food_info['name'].title()}\n"
            f"⚖️ {food_info['grams']:.0f}г\n"
            f"🔥 {food_info['calories']:.0f}ккал\n"
            f"💪 Б{food_info['protein']:.1f}г\n"
            f"🥑 Ж{food_info['fat']:.1f}г\n"
            f"🍞 У{food_info['carbs']:.1f}г\n"
            f"✅ Записано в журнал!"
        )
    else:
        log_to_sheets(user_id, username, text)
        await update.message.reply_text("✅ Записано в журнал! (калории не найдены)")

async def handle_photo(update, context):
    user_id = update.message.from_user.id
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)

    # скачиваем как bytes
    try:
        image_bytes = await file.download_as_bytearray()
        image_bytes = bytes(image_bytes)
    except Exception as e:
        logger.error(f"Не удалось скачать фото: {e}")
        await update.message.reply_text("Не получилось скачать фото. Попробуй ещё раз.")
        return

    # распознаём продукты
    try:
        detected = detect_food_in_photo(image_bytes)
    except Exception as e:
        logger.error(f"Ошибка распознавания фото: {e}")
        await update.message.reply_text("Не получилось распознать еду на фото. Напиши вручную, например: «банан 1шт, яблоко 150 г».")
        return

    if not detected:
        await update.message.reply_text(
            "На фото не распознал еду. Напиши, что на фото и сколько.\n\n"
            "Например: «овсянка 200г, кофе 250мл»")
        PENDING_CONFIRMATIONS[user_id] = {"detected": []}
        return

    # Просим уточнить количество/вес с кнопками
    PENDING_CONFIRMATIONS[user_id] = {"detected": detected}
    guess_list = ", ".join(detected)
    prompt = (
        f"На фото вижу: {guess_list}.\n\n"
        "Выберите действие или напишите корректировки:"
    )
    
    keyboard = [
        [
            InlineKeyboardButton("✅ Принять как есть", callback_data="accept_photo"),
            InlineKeyboardButton("✏️ Написать вручную", callback_data="manual_input")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(prompt, reply_markup=reply_markup)

# === Приветствие ===
async def start(update, context):
    user_first = update.effective_user.first_name
    welcome_text = (
        f"👋 Привет, {user_first}!\n\n"
        "Я бот для подсчёта калорий. Пиши продукты или отправляй фото еды.\n"
        "📊 Отчёты: /report today|week|month"
    )
    await context.bot.send_message(chat_id=update.effective_chat.id, text=welcome_text)
    await menu(update, context)

# === Меню (Inline кнопки) ===
async def menu(update, context):
    menu_text = (
        "📌 Меню:\n\n"
        "🍏 Пиши продукты или отправляй фото\n"
        "📊 Выбери период для отчёта:"
    )

    keyboard = [
        [
            InlineKeyboardButton("📊 Сегодня", callback_data="report_today"),
            InlineKeyboardButton("📊 Неделя", callback_data="report_week"),
            InlineKeyboardButton("📊 Месяц", callback_data="report_month"),
        ],
        [InlineKeyboardButton("ℹ️ Помощь", callback_data="help")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await context.bot.send_message(chat_id=update.effective_chat.id, text=menu_text, reply_markup=reply_markup)

# === Help ===
async def help_cmd(update, context):
    help_text = (
        "ℹ️ Команды:\n"
        "• /start — приветствие\n"
        "• /menu — меню\n"
        "• /report today|week|month — отчёты\n\n"
        "🍏 Пиши продукты или отправляй фото еды"
    )
    await context.bot.send_message(chat_id=update.effective_chat.id, text=help_text)

# === Inline кнопки ===
async def button_handler(update, context):
    query = update.callback_query
    await query.answer()

    if query.data == "report_today":
        context.args = ["today"]
        await handle_report(update, context)
    elif query.data == "report_week":
        context.args = ["week"]
        await handle_report(update, context)
    elif query.data == "report_month":
        context.args = ["month"]
        await handle_report(update, context)
    elif query.data == "help":
          await help_cmd(update, context)
    elif query.data == "accept_photo":
        # Обрабатываем принятие фото как есть
        user_id = query.from_user.id
        username = query.from_user.username or str(user_id)
        
        if user_id in PENDING_CONFIRMATIONS:
            pending_data = PENDING_CONFIRMATIONS.pop(user_id)
            detected_items = pending_data.get("detected", [])
            
            if detected_items:
                # Объединяем все продукты в один запрос
                combined_text = ", ".join(detected_items)
                food_info = get_food_info(combined_text)
                
                if food_info:
                    log_to_sheets(
                        user_id, username, combined_text,
                        food_info["grams"], food_info["calories"], food_info["protein"], food_info["fat"], food_info["carbs"]
                    )
                    await query.edit_message_text(
                        f"🍽 {food_info['name'].title()}\n"
                        f"⚖️ {food_info['grams']:.0f}г\n"
                        f"🔥 {food_info['calories']:.0f}ккал\n"
                        f"💪 Б{food_info['protein']:.1f}г\n"
                        f"🥑 Ж{food_info['fat']:.1f}г\n"
                        f"🍞 У{food_info['carbs']:.1f}г\n"
                        f"✅ Записано в журнал!"
                    )
                else:
                    log_to_sheets(user_id, username, combined_text)
                    await query.edit_message_text("✅ Записано в журнал! (калории не найдены)")
            else:
                await query.edit_message_text("❌ Не удалось обработать фото. Попробуйте написать продукты вручную.")
        else:
            await query.edit_message_text("❌ Данные о фото не найдены. Попробуйте отправить фото снова.")
            
    elif query.data == "manual_input":
        # Просим пользователя написать продукты вручную
        user_id = query.from_user.id
        PENDING_CONFIRMATIONS[user_id] = {"detected": []}
        await query.edit_message_text("✏️ Напишите продукты и количество вручную:\n\nНапример: «банан 150г, яблоко 200г»")

# === Запуск ===
if __name__ == "__main__":
    # 1. Запускаем keepalive сервер для Render
    _start_keepalive_server()

    # 2. Запускаем Telegram-бота
    builder = ApplicationBuilder().token(TOKEN)
    if PROXY_URL:
        builder = builder.request(HTTPXRequest(proxy_url=PROXY_URL))
    app = builder.build()

    # Команды
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("menu", menu))
    app.add_handler(CommandHandler("report", handle_report))

    # inline-кнопки
    app.add_handler(CallbackQueryHandler(button_handler))

    # Сообщения
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    app.run_polling(allowed_updates=["message", "callback_query"])