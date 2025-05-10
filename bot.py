import torch
from fetalnet import FetalNet
from fetalclip import FetalCLIP

# Проверка загрузки моделей
fetalnet_model = FetalNet()
fetalclip_model = FetalCLIP()

print("Модели успешно загружены.")
