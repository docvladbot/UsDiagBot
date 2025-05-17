from telegram.ext import Updater, CommandHandler

# Вставляем твой токен
TOKEN = "7555855256:AAHfTwjfkEaPz3Z89RBD41Q5Y-i5lEGyHms"

# Обработчик команды /start
def start(update, context):
    update.message.reply_text("Бот запущен и работает!")

# Основной запуск
def main():
    updater = Updater(token=TOKEN, use_context=True)
    dp = updater.dispatcher

    dp.add_handler(CommandHandler("start", start))

    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()
