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

from models.fetalnet import FetalNet  # Локально
from models.fetalclip import FetalCLIP  # Локально

BOT_TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_PATH = "/webhook"
WEBHOOK_URL = os.getenv("RENDER_EXTERNAL_URL", "") + WEBHOOK_PATH

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())
router = Router()
dp.include_router(router)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_net = FetalNet().to(device)
model_clip = FetalCLIP().to(device)

@router.message(lambda m: m.text in ["/start", "/help"])
async def start_handler(message: Message):
    await message.answer("Привет! Отправь мне УЗИ изображение или видео, и я его обработаю.")

@router.message(lambda m: m.photo)
async def handle_photo(message: Message):
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    file_path = file.file_path
    file_bytes = await bot.download_file(file_path)

    filename = f"temp_{message.from_user.id}.jpg"
    with open(filename, "wb") as f:
        f.write(file_bytes.read())

    try:
        img = Image.open(filename).convert("RGB")
        width, height = img.size
        await message.answer(f"Изображение получено. Размер: {width}x{height} пикселей.")
        await message.answer("Анализ изображения выполняется...")

        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        input_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            result_net = model_net(input_tensor)
            result_clip = model_clip(input_tensor)

        await message.answer(f"FetalNet вывод: {result_net.argmax().item()}")
        await message.answer(f"FetalCLIP вывод: {result_clip.argmax().item()}")

    except Exception as e:
        await message.answer(f"Ошибка при анализе: {str(e)}")
    finally:
        os.remove(filename)

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
    web.run_app(create_app(), host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
