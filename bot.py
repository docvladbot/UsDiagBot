
import os
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.types import Message
from aiogram.utils import executor
from PIL import Image
import torch
import torchvision.transforms as transforms
from fetalnet.models import FetalNet
from fetalclip.model import FetalCLIP
from aiohttp import web

BOT_TOKEN = os.getenv("BOT_TOKEN")
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)
model_fetalnet = FetalNet()
model_fetalclip = FetalCLIP()

model_fetalnet.eval()
model_fetalclip.eval()

@dp.message_handler(commands=["start", "help"])
async def start(message: Message):
    await message.answer("Привет! Отправь мне УЗИ изображение, и я его проанализирую.")

@dp.message_handler(content_types=["photo"])
async def handle_photo(message: Message):
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    file_path = file.file_path
    file_bytes = await bot.download_file(file_path)

    image_path = f"temp_{message.from_user.id}.jpg"
    with open(image_path, "wb") as f:
        f.write(file_bytes.read())

    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        fetalnet_output = model_fetalnet(tensor)
        fetalclip_output = model_fetalclip(tensor)

    prediction = fetalnet_output.argmax().item()
    description = f"FetalNet предсказал класс: {prediction}"

    os.remove(image_path)
    await message.answer(description)

if __name__ == "__main__":
    executor.start_polling(dp)
