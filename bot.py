import torch
from fetalnet import FetalNet
from fetalclip import FetalCLIP

# Проверка загрузки моделей
fetalnet_model = FetalNet()
fetalclip_model= FetalClip()
    
from telegram.ext import Updater, CommandHandler

def start(update, context):
    update.message.reply_text("Привет! Я бот и я работаю!")

def main():
    # Токен Telegram-бота
    updater = Updater("7555855256:AAHfTwjfkEaPz3Z89RBD41Q5Y-i51EGyHms", use_context=True)

    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))

    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    main()
