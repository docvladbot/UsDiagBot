import torch
from fetalnet import FetalNet
from fetalclip import FetalCLIP

# Проверка загрузки моделей
fetalnet_model = FetalNet()
fetalclip_model = FetalCLIP()

print("Модели успешно загружены.")
Отлично, давай сделаем так, чтобы он точно начал отвечать в Telegram. Сейчас мы заменим bot.py на минимальный рабочий вариант с polling и ответом на /start.


---
from telegram.ext import Updater, CommandHandler
def start(update, context):
    update.message.reply_text("Привет! Я работаю.")

def main():
    # Замени 'YOUR_TELEGRAM_TOKEN' на свой реальный токен Telegram-бота
    updater = Updater("YOUR_TELEGRAM_TOKEN", use_context=True)

    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))

    updater.start_polling()
    updater.idle()

if __name__ == "__main__":
    

