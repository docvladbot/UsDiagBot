from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telegram import InputFile
import logging
import os
import torch

# Инициализация моделей (заглушки)
from fetalnet import FetalNet
from fetalclip import FetalCLIP

fetalnet_model = FetalNet()
fetalclip_model = FetalCLIP()

# Включаем логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Обработка команды /start
def start(update, context):
    update.message.reply_text("Привет! Я бот для анализа УЗИ. Пришлите мне фото или видео.")

# Обработка изображений
def handle_photo(update, context):
    file = update.message.photo[-1].get_file()
    file_path = file.download()
    update.message.reply_text("Фото получено. Анализирую...")

    # Здесь должен быть вызов модели
    result = "FetalNet: диагноз УЗИ..."  # пример
    update.message.reply_text(result)

    os.remove(file_path)

# Обработка видео
def handle_video(update, context):
    file = update.message.video.get_file()
    file_path = file.download()
    update.message.reply_text("Видео получено. Извлекаю кадры и анализирую...")

    # Здесь должен быть вызов модели
    result = "FetalCLIP: анализ видео..."  # пример
    update.message.reply_text(result)

    os.remove(file_path)

# Основная функция
def main():
    TOKEN = "7555855256:AAHfTwjfkEaPz3Z89RBD41Q5Y-i51EGyHms"
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.photo, handle_photo))
    dp.add_handler(MessageHandler(Filters.video, handle_video))

    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
