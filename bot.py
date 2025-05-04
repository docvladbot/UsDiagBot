import os
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
from PIL import Image

BOT_TOKEN = os.getenv("BOT_TOKEN")

logging.basicConfig(level=logging.INFO)
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher(bot)

@dp.message_handler(commands=["start", "help"])
async def start(message: types.Message):
    await message.reply("Привет! Отправь мне УЗИ изображение, и я его проанализирую.")

@dp.message_handler(content_types=["photo"])
async def handle_photo(message: types.Message):
    try:
        photo = message.photo[-1]
        file = await bot.get_file(photo.file_id)
        file_path = file.file_path
        downloaded_file = await bot.download_file(file_path)

        image_name = f"temp_{message.from_user.id}.jpg"
        with open(image_name, "wb") as new_file:
            new_file.write(downloaded_file.read())

        img = Image.open(image_name)
        width, height = img.size
        await message.reply(f"Изображение получено. Размер: {width}x{height} пикселей.")
        os.remove(image_name)
    except Exception as e:
        await message.reply("Произошла ошибка при обработке изображения.")

@dp.message_handler()
async def echo(message: types.Message):
    await message.reply("Пожалуйста, отправьте изображение.")

if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True)
