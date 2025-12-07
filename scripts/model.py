import torch

import json

import torch.nn as nn

from torchvision import models

from transformers import BertModel

import torchvision.models as models

from torchvision.models import resnet50

# ================================

# Модель: загрузка image encoder + BERT + head

# ================================



print("=== Создание модели ===")

print("Загружаем конфиг...")



with open("config.json", "r", encoding="utf-8") as f:

    config = json.load(f)

cfg_model = config["model"]



# ------------------------------------------

# Image Encoder (ResNet50)

# ------------------------------------------



print("\n[1] Загружаем ResNet50...")

resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

resnet.fc = nn.Identity()  # убираем классификатор → получаем фичи 2048

print("✔ ResNet50 загружен. Выход размерности: 2048")



# ------------------------------------------

# Text Encoder (BERT)

# ------------------------------------------

print("\n[2] Загружаем BERT (bert-base-uncased)...")

bert = BertModel.from_pretrained("bert-base-uncased")

print("✔ BERT загружен. Выход размерности: 768")



# ------------------------------------------

# Финальная модель

# ------------------------------------------

class CalorieModel(nn.Module):

    def __init__(self, resnet, bert, image_dim, text_dim, hidden_dim, dropout=0.2):

        super().__init__()

        self.image_encoder = resnet

        self.text_encoder = bert



        self.fc = nn.Sequential(

            nn.Linear(image_dim + text_dim, hidden_dim),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 1)

        )



    def forward(self, images, input_ids, attention_mask):

        # IMAGE

        img_feat = self.image_encoder(images)  # (B, 2048)



        # TEXT

        text_outputs = self.text_encoder(

            input_ids=input_ids,

            attention_mask=attention_mask

        )

        text_feat = text_outputs.last_hidden_state[:, 0, :]  # CLS токен (B, 768)



        # CONCAT

        fused = torch.cat([img_feat, text_feat], dim=1)



        # HEAD

        out = self.fc(fused)

        return out.squeeze(1)  # (B,)



print("\n[3] Собираем общую модель...")

model = CalorieModel(

    resnet=resnet,

    bert=bert,

    image_dim=cfg_model["image_embed_dim"],

    text_dim=cfg_model["text_embed_dim"],

    hidden_dim=cfg_model["hidden_dim"],

    dropout=cfg_model["dropout"]

)



print("✔ Модель собрана.")


