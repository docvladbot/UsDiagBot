import logging
from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from fetalnet import FetalNet
from fetalclip import FetalCLIP
import torch
import cv2
import tempfile
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOKEN = "YOUR_BOT_TOKEN"

fetalnet_model = FetalNet()
fetalclip_model = FetalCLIP()

def start(update: Update, context: CallbackContext):
    update.message.reply_text("Бот запущен и готов к работе! Пришлите изображение или видео УЗИ.")

def handle_photo(update: Update, context: CallbackContext):
    photo_file = update.message.photo[-1].get_file()
    photo_path = tempfile.mktemp(suffix=".jpg")
    photo_file.download(photo_path)
    result = fetalnet_model.predict(photo_path)
    update.message.reply_text(f"FetalNet результат: {result}")
    os.remove(photo_path)

def handle_video(update: Update, context: CallbackContext):
    video_file = update.message.video.get_file()
    video_path = tempfile.mktemp(suffix=".mp4")
    video_file.download(video_path)
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    success, frame_id = True, 0
    results = []

    while success:
        success, frame = cap.read()
        if not success:
            break
        if int(frame_id % round(frame_rate)) == 0:
            temp_image_path = tempfile.mktemp(suffix=".jpg")
            cv2.imwrite(temp_image_path, frame)
            result = fetalnet_model.predict(temp_image_path)
            results.append(result)
            os.remove(temp_image_path)
        frame_id += 1

    cap.release()
    os.remove(video_path)
    update.message.reply_text("Результаты анализа по кадрам:
" + "\n".join(results))

def main():
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(MessageHandler(Filters.photo, handle_photo))
    dp.add_handler(MessageHandler(Filters.video, handle_video))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
