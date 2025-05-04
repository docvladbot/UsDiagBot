
import os
import logging
from aiogram import Bot, Dispatcher, Router, types
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Message
from PIL import Image
from aiohttp import web
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application

# Настройки
BOT_TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_PATH = "/webhook"
WEBHOOK_URL = os.getenv("RENDER_EXTERNAL_URL", "") + WEBHOOK_PATH

# Инициализация
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())
router = Router()
dp.include_router(router)

@router.message(lambda m: m.text in ["/start", "/help"])
async def start_handler(message: Message):
    await message.answer("Привет! Отправь мне УЗИ изображение, и я его проанализирую.")

@router.message(lambda m: m.photo)
async def handle_photo(message: Message):
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    file_path = file.file_path
    file_bytes = await bot.download_file(file_path)

    filename = f"temp_{message.from_user.id}.jpg"
    with open(filename, "wb") as f:
        f.write(file_bytes.read())

    img = Image.open(filename)
    width, height = img.size
    os.remove(filename)

    await message.answer(f"Изображение получено. Размер: {width}x{height} пикселей.")

@router.message()
async def fallback(message: Message):
    await message.answer("Пожалуйста, отправьте изображение УЗИ.")

# Webhook-приложение
async def on_startup(app: web.Application):
    await bot.set_webhook(WEBHOOK_URL)

async def on_shutdown(app: web.Application):
    await bot.delete_webhook()

def create_app():
    app = web.Application()
    SimpleRequestHandler(dispatcher=dp, bot=bot).register(app, path=WEBHOOK_PATH)
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    return app

if __name__ == "__main__":
    web.run_app(create_app(), host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
