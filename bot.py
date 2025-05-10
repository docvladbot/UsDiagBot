import os
import logging
from aiogram import Bot, Dispatcher, Router, types
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Message
from PIL import Image
from aiohttp import web
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
import torchvision.transforms as transforms
import torch

from fetalnet import FetalNet
from fetalclip import FetalCLIP
import torchvision.io as io

BOT_TOKEN = os.getenv("BOT_TOKEN")
WEBHOOK_PATH = "/webhook"
WEBHOOK_URL = os.getenv("RENDER_EXTERNAL_URL", "") + WEBHOOK_PATH

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())
router = Router()
dp.include_router(router)

model_fetalnet = FetalNet()
model_fetalnet.eval()

model_fetalclip = FetalCLIP()
model_fetalclip.eval()

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

    img = Image.open(filename).convert("RGB")
    width, height = img.size
    await message.answer(f"Изображение получено. Размер: {width}x{height} пикселей.")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output_net = model_fetalnet(input_tensor)
        output_clip = model_fetalclip(input_tensor)

    os.remove(filename)

    await message.answer("Обработка завершена (результаты заглушка).")

@router.message(lambda m: m.video)
async def handle_video(message: Message):
    video = message.video
    file = await bot.get_file(video.file_id)
    file_path = file.file_path
    file_bytes = await bot.download_file(file_path)

    filename = f"temp_{message.from_user.id}.mp4"
    with open(filename, "wb") as f:
        f.write(file_bytes.read())

    try:
        vr = io.read_video(filename, pts_unit='sec')[0]
        frames = vr[::30]  # 1 кадр в секунду при 30 FPS
        results = []
        for frame in frames:
            image = transforms.ToPILImage()(frame)
            image = image.convert("RGB")
            input_tensor = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])(image).unsqueeze(0)
            with torch.no_grad():
                _ = model_fetalnet(input_tensor)
                _ = model_fetalclip(input_tensor)
        await message.answer("Видео успешно обработано (результаты заглушка).")
    except Exception as e:
        await message.answer(f"Ошибка обработки видео: {e}")

    os.remove(filename)

@router.message()
async def fallback(message: Message):
    await message.answer("Пожалуйста, отправьте изображение или видео УЗИ.")

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