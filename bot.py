import os
import logging
from aiogram import Bot, Dispatcher, Router, types
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Message
from PIL import Image
from aiohttp import web
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
import torch
import torchvision.transforms as transforms
from torchvision.io import read_video
import tempfile
import shutil

BOT_TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_PATH = "/webhook"
WEBHOOK_URL = os.getenv("RENDER_EXTERNAL_URL", "") + WEBHOOK_PATH

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())
router = Router()
dp.include_router(router)

@router.message(lambda m: m.text in ["/start", "/help"])
async def start_handler(message: Message):
    await message.answer("Привет! Отправь мне УЗИ изображение или видео, и я его обработаю.")

@router.message(lambda m: m.photo)
async def handle_photo(message: Message):
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    file_path = file.file_path
    file_bytes = await bot.download_file(file_path)

    temp_dir = tempfile.mkdtemp()
    filename = os.path.join(temp_dir, "image.jpg")
    with open(filename, "wb") as f:
        f.write(file_bytes.read())

    img = Image.open(filename)
    width, height = img.size
    await message.answer(f"Изображение получено. Размер: {width}x{height} пикселей.")
    await message.answer("Анализ изображения выполняется... (модель пока не подключена)")

    shutil.rmtree(temp_dir)

@router.message(lambda m: m.video)
async def handle_video(message: Message):
    video = message.video
    file = await bot.get_file(video.file_id)
    file_path = file.file_path
    file_bytes = await bot.download_file(file_path)

    temp_dir = tempfile.mkdtemp()
    filename = os.path.join(temp_dir, "video.mp4")
    with open(filename, "wb") as f:
        f.write(file_bytes.read())

    try:
        video_tensor, _, _ = read_video(filename, pts_unit='sec')
        frame_count = video_tensor.shape[0]
        await message.answer(f"Видео получено. Количество кадров: {frame_count}")
        await message.answer("Анализ видео выполняется... (модель пока не подключена)")
    except Exception as e:
        await message.answer(f"Ошибка при обработке видео: {str(e)}")

    shutil.rmtree(temp_dir)

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
    web.run_app(create_app(), host="0.0.0.0", port=int(os.getenv("PORT", 3000)))