import argparse

import json

from utils import train_model



def load_config(path: str):

    """Загрузка JSON-конфига."""

    with open(path, "r", encoding="utf-8") as f:

        return json.load(f)





if __name__ == "__main__":

    # -----------------------------

    # Аргументы запуска

    # -----------------------------

    parser = argparse.ArgumentParser(description="Train calorie prediction model")

    parser.add_argument("--config", type=str, required=True, help="Path to config.json")



    args = parser.parse_args()



    # -----------------------------

    # Загрузка конфига

    # -----------------------------

    config = load_config(args.config)



    print("\n======================================")

    print("ЗАПУСК ОБУЧЕНИЯ МОДЕЛИ")

    print("Используем конфиг:", args.config)

    print("======================================\n")



    # -----------------------------

    # Старт тренировки

    # -----------------------------

    train_model(config)