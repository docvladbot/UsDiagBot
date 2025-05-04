

import os
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Message
from aiogram import Router
from aiogram import executor
from PIL import Image

BOT_TOKEN = os.getenv("BOT_TOKEN")
bot = Bot(token=BOT_TOKEN, parse_mode=ParseMode.HTML)
dp = Dispatcher(storage=MemoryStorage())
router = Router()

@router.message(commands={"start", "help"})
async def start_handler(message: Message):
    await message.answer("Привет! Отправь мне УЗИ изображение, и я его проанализирую.")

@router.message(content_types=["photo"])
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
async def default_handler(message: Message):
    await message.answer("Пожалуйста, отправьте изображение УЗИ.")

dp.include_router(router)

if __name__ == "__main__":
    import asyncio
    async def run():
        await dp.start_polling(bot)

    asyncio.run(run())


---

Теперь:

1. Перейди на GitHub → файл bot.py


2. Нажми иконку карандаша (редактировать)


3. Удали старый текст и вставь этот новый


4. Нажми “Commit changes”



Готов помочь, если понадобится повторно отправить. Напиши, когда вставишь — и мы запустим деплой!

