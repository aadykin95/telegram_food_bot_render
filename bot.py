import logging
import re
from datetime import datetime, timedelta

import gspread
import requests
import matplotlib
matplotlib.use("Agg")  # серверный backend
import matplotlib.pyplot as plt
import pandas as pd

from deep_translator import GoogleTranslator
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, filters
from telegram.request import HTTPXRequest

# === Google Vision ===
from google.cloud import vision
from google.oauth2 import service_account

# === SETTINGS (Render-ready) ===
import os, http.server, socketserver, threading

from dotenv import load_dotenv
load_dotenv()  # локально подтянет .env; на Render не мешает

# --- Читаем переменные окружения ---
TOKEN = os.environ["TOKEN"]
CALORIE_NINJAS_API_KEY = os.environ["CALORIE_NINJAS_API_KEY"]
SPREADSHEET_NAME = os.environ.get("SPREADSHEET_NAME", "FoodLog")
SHEET_NAME = os.environ.get("SHEET_NAME", "log")
PROXY_URL = os.environ.get("PROXY_URL", "")

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

# === Google Vision ===
creds = service_account.Credentials.from_service_account_file(GCP_CREDENTIALS_FILE)
vision_client = vision.ImageAnnotatorClient(credentials=creds)

# --- маленький HTTP-сервер для Render Web Service ---
def _start_keepalive_server():
    port = int(os.getenv("PORT", "8080"))  # Render всегда задаёт PORT
    class _Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"✅ Bot is alive!")
        def log_message(self, format, *args):
            return  # отключаем лишние логи

    def _serve():
        with socketserver.TCPServer(("", port), _Handler) as httpd:
            print(f"Keepalive server listening on port {port}")
            httpd.serve_forever()

    threading.Thread(target=_serve, daemon=True).start()

# === Логирование ===
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram.vendor.ptb_urllib3").setLevel(logging.WARNING)

# === Состояние подтверждений ===
PENDING_CONFIRMATIONS = {}

# === Справочники ===
UNIT_MAP_RU_TO_EN = {
    "шт": "piece", "штука": "piece", "штук": "pieces",
    "г": "g", "гр": "g", "gram": "g", "грамм": "g", "граммов": "g",
    "кг": "kg", "килограмм": "kg",
    "мл": "ml", "л": "l",
    "ложка": "tbsp", "ст.л": "tbsp", "столовая ложка": "tbsp",
    "ч.л": "tsp", "чайная ложка": "tsp",
    "ломтик": "slice", "кусок": "piece", "батон": "loaf",
    "бутерброд": "sandwich",
    "яйцо": "egg", "яйца": "eggs",
}
FOOD_HINTS = {
    "банан","яблоко","груша","апельсин","мандарины","апельсины","огурец","помидор","томат","картофель","лук","чеснок",
    "хлеб","батон","булка","булочка","сыр","яйцо","яйца","курица","филе","индейка","говядина","свинина","рыба","лосось",
    "тунец","рис","гречка","макароны","паста","овсянка","йогурт","молоко","кефир","творог","масло","орехи","миндаль",
    "фундук","арахис","печенье","шоколад","торт","пицца","бургер","суп","салат","брокколи","цветная капуста","авокадо",
    "виноград","персик","слива","черника","клубника","малина","арбуз","дыня","ковбаса","колбаса","сосиски"
}

# === Утилиты текста ===
def clean_food_text(text):
    text = text.strip().lower()
    text = re.sub(r"[!?,;:]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text

def translate_if_needed(text):
    if not text:
        return text
    text = clean_food_text(text)
    try:
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        return clean_food_text(translated)
    except Exception as e:
        logger.error(f"Ошибка перевода: {e}")
        return text

# === CalorieNinjas API ===
def get_food_info(query):
    url = f"https://api.calorieninjas.com/v1/nutrition?query={query}"
    headers = {"X-Api-Key": CALORIE_NINJAS_API_KEY}
    try:
        response = requests.get(url, headers=headers, timeout=20)
    except Exception as e:
        logger.error(f"CalorieNinjas запрос упал: {e}")
        return None
    if response.status_code == 200:
        data = response.json()
        if data.get("items"):
            item = data["items"][0]
            return {
                "name": item.get("name",""),
                "calories": float(item.get("calories",0)),
                "protein": float(item.get("protein_g",0)),
                "fat": float(item.get("fat_total_g",0)),
                "carbs": float(item.get("carbohydrates_total_g",0))
            }
    logger.warning(f"CalorieNinjas response {response.status_code}: {response.text[:200] if 'response' in locals() else 'no response'}")
    return None

def safe_float(value):
    try:
        return float(str(value).replace(",", "."))
    except:
        return 0.0

# === Лог в Google Sheets ===
def log_to_sheets(user_id, username, dish, translated_dish="", photo_url="", calories="", protein="", fat="", carbs=""):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    worksheet.append_row([
        date_str, time_str, user_id, username, dish, translated_dish,
        calories, protein, fat, carbs, photo_url
    ])

# === Vision — распознать еду на фото ===
def detect_food_in_photo(image_bytes, max_items=6):
    # image_bytes должен быть bytes, не bytearray
    if isinstance(image_bytes, bytearray):
        image_bytes = bytes(image_bytes)

    image = vision.Image(content=image_bytes)

    # Лейблы
    labels_response = vision_client.label_detection(image=image)
    labels = labels_response.label_annotations or []
    logger.info("Vision labels (top 10): " + ", ".join(f"{l.description}:{l.score:.2f}" for l in labels[:10]))

    # Объекты (может быть отключено в проекте — тогда просто пропустим)
    try:
        objects_response = vision_client.object_localization(image=image)
        objects = objects_response.localized_object_annotations or []
        logger.info("Vision objects (top 10): " + ", ".join(f"{o.name}:{o.score:.2f}" for o in objects[:10]))
    except Exception as e:
        logger.warning(f"Object localization недоступно: {e}")
        objects = []

    candidates = []
    for lb in labels[:25]:
        candidates.append(lb.description.lower())
    for obj in objects[:25]:
        candidates.append(obj.name.lower())

    items, seen = [], set()
    for name in candidates:
        name = clean_food_text(name)
        if name in FOOD_HINTS or any(k in name for k in [
            "bread","banana","apple","tomato","cucumber","salmon","fish","meat",
            "cheese","egg","rice","pasta","yogurt","milk","oat","beef","pork","chicken","sausage","ham","bacon","noodle","potato"
        ]):
            if name not in seen:
                seen.add(name)
                items.append(name)

    if not items:
        # если ничего «едового» — возьмём 1–3 верхних лейбла как догадку
        items = [clean_food_text(lb.description) for lb in labels[:3]]

    return items[:max_items]

# === Парсинг подтверждения пользователя ===
def parse_user_confirmation(text, fallback_items):
    """
    Формат: 'банан 1шт, яблоко 150 г, хлеб 1 ломтик'
    Если пусто — 1 шт для каждого распознанного.
    """
    text = (text or "").strip()
    if not text:
        return [{"name_ru": it, "amount": 1.0, "unit_ru": "шт"} for it in fallback_items]

    parts = [p.strip() for p in text.split(",") if p.strip()]
    items = []
    for p in parts:
        m = re.match(r"([^\d]+?)\s*([\d.,]+)?\s*([^\d,]+)?$", p, flags=re.UNICODE)
        if m:
            name_ru = clean_food_text(m.group(1))
            amount = safe_float(m.group(2)) if m.group(2) else 1.0
            unit_ru = clean_food_text(m.group(3)) if m.group(3) else "шт"
            unit_ru = (unit_ru
                       .replace("грамм", "г").replace("гр", "г")
                       .replace("килограмм", "кг").replace("килог", "кг")
                       .replace("милилитр","мл").replace("миллилитр","мл")
                       .replace("штук","шт").replace("штуки","шт")
                       .replace("slice","ломтик"))
            items.append({"name_ru": name_ru, "amount": amount, "unit_ru": unit_ru})
    if not items:
        items = [{"name_ru": it, "amount": 1.0, "unit_ru": "шт"} for it in fallback_items]
    return items

# === Построить запрос к CalorieNinjas ===
def to_cninjas_query(name_ru, amount, unit_ru):
    name_en = translate_if_needed(name_ru)
    unit_key = (unit_ru or "").strip().lower()
    unit_en = UNIT_MAP_RU_TO_EN.get(unit_key, unit_key or "piece")
    if unit_en in ("piece", "pieces", "egg", "eggs", "loaf", "slice", "sandwich", "tbsp", "tsp"):
        qty_val = int(amount) if float(amount).is_integer() else amount
        qty_str = f"{qty_val} {unit_en}"
    else:
        qty_str = f"{amount}{unit_en if unit_en in ('g','kg','ml','l') else ' ' + unit_en}"
    return f"{qty_str} {name_en}".strip()

# === Подсчёт нутриентов ===
def compute_totals_from_items(items):
    totals = {"cal": 0.0, "prot": 0.0, "fat": 0.0, "carb": 0.0}
    per_item = []
    for it in items:
        name_ru = it["name_ru"]
        amount = it.get("amount", 1.0)
        unit_ru = it.get("unit_ru", "шт")
        query = to_cninjas_query(name_ru, amount, unit_ru)
        info = get_food_info(query)
        if not info:
            fallback_query = translate_if_needed(name_ru)
            info = get_food_info(fallback_query)
            if not info:
                per_item.append({"name_ru": name_ru, "query": query, "info": None})
                continue
        totals["cal"] += info["calories"]
        totals["prot"] += info["protein"]
        totals["fat"] += info["fat"]
        totals["carb"] += info["carbs"]
        per_item.append({"name_ru": name_ru, "query": query, "info": info})
    return totals, per_item

def format_items_ru(items):
    chunks = []
    for it in items:
        amount = it.get("amount", 1.0)
        amount_str = str(int(amount)) if float(amount).is_integer() else str(amount)
        unit_ru = it.get("unit_ru", "шт")
        chunks.append(f"{it['name_ru']} {amount_str} {unit_ru}")
    return "; ".join(chunks)

def format_per_item_breakdown(per_item):
    lines = []
    for p in per_item:
        if p["info"]:
            info = p["info"]
            lines.append(f"• {p['name_ru']} — {info['calories']:.0f} ккал, Б {info['protein']:.1f} г, Ж {info['fat']:.1f} г, У {info['carbs']:.1f} г")
        else:
            lines.append(f"• {p['name_ru']} — не удалось найти в базе, пропущено")
    return "\n".join(lines)

# === ОТЧЁТЫ ===
async def handle_report(update, context):
    if len(context.args) == 0:
        await update.message.reply_text("❗ Используй: /report today | week | month")
        return

    period = context.args[0].lower()
    today = datetime.now().date()

    if period == "today":
        start_date = today - timedelta(days=29)   # 30 дней
    elif period == "week":
        start_date = today - timedelta(weeks=11)  # 12 недель
    elif period == "month":
        start_date = today.replace(day=1) - timedelta(days=365)  # 12 мес
    else:
        await update.message.reply_text("❗ Неизвестный период. Доступно: today | week | month")
        return

    rows = worksheet.get_all_values()[1:]  # без заголовка
    records = []
    for row in rows:
        try:
            date_str = row[0].strip()
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

            if start_date <= date_obj <= today:
                records.append({
                    "date": date_obj,
                    "cal": safe_float(cal),
                    "prot": safe_float(prot),
                    "fat": safe_float(fat),
                    "carb": safe_float(carb),
                })
        except Exception:
            continue

    if not records:
        await update.message.reply_text("📭 Данных за этот период нет.")
        return

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])

    if period == "today":
        # по дням
        grouped = df.groupby("date").sum(numeric_only=True).reset_index()
        grouped["label"] = grouped["date"].dt.strftime("%d.%m")

        # заполним пропуски днями
        rng = pd.date_range(start=start_date, end=today, freq="D")
        full_df = pd.DataFrame({"date": rng, "label": rng.strftime("%d.%m")})
        grouped = pd.merge(full_df, grouped, on=["date", "label"], how="left").fillna(0)

    elif period == "week":
        # по ISO-неделям
        iso = df["date"].dt.isocalendar()
        df["week"] = iso.week
        df["year"] = iso.year
        grouped = df.groupby(["year", "week"]).sum(numeric_only=True).reset_index()
        # Подпись — номер недели (01..53)
        grouped["label"] = grouped["week"].apply(lambda w: f"{int(w):02d}")

        # заполним пропуски ИСО-неделями в диапазоне
        rng = pd.date_range(start=start_date, end=today, freq="W-MON")
        iso_rng = rng.isocalendar()
        full_df = pd.DataFrame({
            "year": iso_rng.year.astype(int),
            "week": iso_rng.week.astype(int),
            "label": [f"{int(w):02d}" for w in iso_rng.week],
        }).drop_duplicates()
        grouped = pd.merge(full_df, grouped, on=["year", "week", "label"], how="left").fillna(0)

    else:  # month
        df["month"] = df["date"].dt.month
        df["year"] = df["date"].dt.year
        grouped = df.groupby(["year", "month"]).sum(numeric_only=True).reset_index()
        # Подпись — MM.YY (например 01.25)
        grouped["label"] = grouped.apply(lambda r: f"{int(r['month']):02d}.{int(r['year'])%100:02d}", axis=1)

        # полный ряд месяцев
        rng = pd.date_range(start=start_date, end=today, freq="MS")
        full_df = pd.DataFrame({
            "year": rng.year,
            "month": rng.month,
            "label": [f"{int(m):02d}.{int(y)%100:02d}" for m, y in zip(rng.month, rng.year)],
        })
        grouped = pd.merge(full_df, grouped, on=["year", "month", "label"], how="left").fillna(0)

    # Построение графика (X — готовые подписи)
    plt.figure(figsize=(9, 5))
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
    total_cal = grouped["cal"].sum()
    total_prot = grouped["prot"].sum()
    total_fat = grouped["fat"].sum()
    total_carb = grouped["carb"].sum()

    text_report = (
        f"📊 Отчёт за {period}:\n"
        f"🔥 Калории: {total_cal:.1f}\n"
        f"💪 Белки: {total_prot:.1f} г\n"
        f"🥑 Жиры: {total_fat:.1f} г\n"
        f"🍞 Углеводы: {total_carb:.1f} г"
    )

    await update.message.reply_text(text_report)
    await update.message.reply_photo(photo=open(chart_path, "rb"))

# === Обработчики ===
async def handle_text(update, context):
    user_id = update.message.from_user.id
    username = update.message.from_user.username or str(user_id)
    text = update.message.text or ""

    # Ожидание подтверждения по фото
    if user_id in PENDING_CONFIRMATIONS:
        fallback_items = PENDING_CONFIRMATIONS.pop(user_id).get("detected", [])
        items = parse_user_confirmation(text, fallback_items)
        if not items:
            await update.message.reply_text("Не понял формат. Пример: «банан 1шт, яблоко 150 г, хлеб 1 ломтик». Попробуй ещё раз.")
            PENDING_CONFIRMATIONS[user_id] = {"detected": fallback_items}
            return

        totals, per_item = compute_totals_from_items(items)
        dish_ru = format_items_ru(items)
        translated_dish = translate_if_needed(dish_ru)

        log_to_sheets(
            user_id, username, dish_ru, translated_dish, "",
            f"{totals['cal']:.1f}", f"{totals['prot']:.1f}", f"{totals['fat']:.1f}", f"{totals['carb']:.1f}"
        )

        breakdown = format_per_item_breakdown(per_item)
        msg = (
            "✅ Записано в журнал!\n\n"
            f"{breakdown}\n\n"
            f"Итого: 🔥 {totals['cal']:.0f} ккал, "
            f"Б {totals['prot']:.1f} г, Ж {totals['fat']:.1f} г, У {totals['carb']:.1f} г"
        )
        await update.message.reply_text(msg)
        return

    # Обычная текстовая запись
    cleaned_text = clean_food_text(text)
    translated_text = translate_if_needed(cleaned_text)
    logger.warning(f"💬 Сообщение от {username}: {cleaned_text} → {translated_text}")

    food_info = get_food_info(translated_text)
    if food_info:
        log_to_sheets(
            user_id, username, cleaned_text, translated_text, "",
            food_info["calories"], food_info["protein"], food_info["fat"], food_info["carbs"]
        )
        await update.message.reply_text(
            f"🍽 {food_info['name'].title()}\n"
            f"🔥 Калории: {food_info['calories']:.0f}\n"
            f"💪 Белки: {food_info['protein']:.1f} г\n"
            f"🥑 Жиры: {food_info['fat']:.1f} г\n"
            f"🍞 Углеводы: {food_info['carbs']:.1f} г\n✅ Записано в журнал!"
        )
    else:
        log_to_sheets(user_id, username, cleaned_text, translated_text)
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

    # распознаём Vision
    try:
        detected = detect_food_in_photo(image_bytes)
        logger.info(f"Vision API нашёл: {detected}")
    except Exception as e:
        logger.exception("Vision API ошибка")
        await update.message.reply_text("Не получилось распознать еду на фото. Напиши вручную, например: «банан 1шт, яблоко 150 г».")
        return

    if not detected:
        await update.message.reply_text(
            "На фото не распознал еду. Напиши, что на фото и сколько:\n"
            "например: «банан 1шт, яблоко 150 г».")
        PENDING_CONFIRMATIONS[user_id] = {"detected": []}
        return

    # Просим уточнить количество/вес
    PENDING_CONFIRMATIONS[user_id] = {"detected": detected}
    guess_list = ", ".join(detected)
    prompt = (
        f"На фото вижу: {guess_list}.\n\n"
        "Уточни количество/вес в формате:\n"
        "банан 1шт, яблоко 150 г, хлеб 1 ломтик\n\n"
        "Можно исправлять список (добавлять/удалять), я всё просуммирую."
    )
    await update.message.reply_text(prompt)

async def handle_command(update, context):
    user_id = update.message.from_user.id
    username = update.message.from_user.username or str(user_id)
    command = update.message.text
    log_to_sheets(user_id, username, command)
    await update.message.reply_text(f"📌 Команда '{command}' записана в журнал.")

# === Запуск ===
if __name__ == "__main__":
    from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
    from telegram.request import HTTPXRequest

    # 1. Запускаем keepalive сервер для Render
    _start_keepalive_server()

    # 2. Запускаем Telegram-бота
    builder = ApplicationBuilder().token(TOKEN)
    if PROXY_URL:
        builder = builder.request(HTTPXRequest(proxy_url=PROXY_URL))
    app = builder.build()

    # Команды
    app.add_handler(CommandHandler(["start", "help"], handle_command))
    app.add_handler(CommandHandler("report", handle_report))
    # Сообщения
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    logger.warning("🚀 Бот запущен...")
    app.run_polling(allowed_updates=["message"])