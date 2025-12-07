import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import BertTokenizerFast


class CalorieDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str,
        dish_csv: str = "dish.csv",
        ingr_csv: str = "ingredients.csv",
        transforms=None,
        text_model_name="bert-base-uncased",
        max_len: int = 64
    ):
        self.data_dir = Path(data_dir)
        self.img_dir = self.data_dir / "images"
        self.transforms = transforms
        self.max_len = max_len

        # ---------- CSV загрузка ----------
        self.dish_df = pd.read_csv(self.data_dir / dish_csv)
        self.ingr_df = pd.read_csv(self.data_dir / ingr_csv)

        # приведение id ингредиентов к строковому виду
        self.ingr_df["id"] = self.ingr_df["id"].astype(str)
        self.ingr_dict = dict(zip(self.ingr_df["id"], self.ingr_df["ingr"]))

        # фильтрация
        self.dish_df = self.dish_df[self.dish_df["split"] == split].reset_index(drop=True)

        # токенизатор
        self.tokenizer = BertTokenizerFast.from_pretrained(text_model_name)

        # ---------- обработка всех блюд ----------
        self.samples = []
        for _, row in self.dish_df.iterrows():
            dish_id = row["dish_id"]
            raw_ingrs = str(row["ingredients"])

            # Парсим IDs
            if raw_ingrs.strip():
                raw_list = [
                    x.strip() for x in raw_ingrs.replace(",", ";").split(";") if x.strip()
                ]
            else:
                raw_list = []

            # нормализуем "ingr_0000000528" → "528"
            norm_ids = [self.normalize_ingr_id(x) for x in raw_list]

            # собираем названия ингредиентов
            names = []
            for nid in norm_ids:
                if nid in self.ingr_dict:
                    names.append(self.ingr_dict[nid])

            # финальный текст для подачи в BERT
            final_text = "; ".join(names)

            calories = float(row["total_calories"]) if not pd.isna(row["total_calories"]) else 0.0

            self.samples.append({
                "dish_id": dish_id,
                "text": final_text,
                "target": calories
            })

    # ---- преобразователь ingr_0000000528 → 528 ----
    def normalize_ingr_id(self, raw_id: str) -> str:
        if not isinstance(raw_id, str):
            return None
        if raw_id.startswith("ingr_"):
            num = raw_id.replace("ingr_", "").lstrip("0")
            return num if num != "" else "0"
        return raw_id

    # ---------- загрузка изображения ----------
    def _load_image(self, dish_id):
        dish_folder = self.img_dir / dish_id
        rgb_path = dish_folder / "rgb.png"
        if not rgb_path.exists():
            raise FileNotFoundError(f"Image not found: expected {rgb_path}")
        return Image.open(rgb_path).convert("RGB")

    # ---------- выборка ----------
    def __getitem__(self, idx):
        s = self.samples[idx]

        # изображение
        img = self._load_image(s["dish_id"])
        if self.transforms:
            img = self.transforms(img)
        else:
            img = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.

        # текст → токены BERT
        encoded = self.tokenizer(
            s["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        target = torch.tensor([s["target"]], dtype=torch.float32)

        return {
            "image": img,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target": target
        }

    def __len__(self):
        return len(self.samples)
 