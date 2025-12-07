import os

import torch

import numpy as np

from typing import Dict

from tqdm import tqdm

from torch.utils.data import DataLoader

from transformers import BertTokenizerFast, BertModel

from torchvision.models import resnet50, ResNet50_Weights

import torchvision.models as models


from dataset import CalorieDataset

from transforms import get_train_transforms, get_val_transforms, get_test_transforms

from model import CalorieModel, resnet



def set_seed(seed: int = 42):

    import random

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False




def collate_fn(batch, tokenizer):

    images = []

    texts = []

    targets = []



    for img, ingr_idxs, target in batch:

        images.append(img)

        texts.append(" ".join([str(x.item()) for x in ingr_idxs]))

        targets.append(target)



    images = torch.stack(images)

    targets = torch.stack(targets).view(-1)



    encoded = tokenizer(

        texts,

        padding=True,

        truncation=True,

        max_length=64,

        return_tensors="pt"

    )



    return {

        "images": images,

        "input_ids": encoded["input_ids"],

        "attention_mask": encoded["attention_mask"],

        "targets": targets

    }





def build_dataloaders(config):

    data_cfg = config["data"]

    tf_cfg = config["transforms"]



    tokenizer = BertTokenizerFast.from_pretrained(config["model"]["text_model"])



    train_ds = CalorieDataset(

        data_dir=data_cfg["data_dir"],

        split=data_cfg["train_split"],

        dish_csv=data_cfg["dish_csv"],

        ingr_csv=data_cfg["ingredients_csv"],

        transforms=get_train_transforms(tf_cfg["image_size"])

        if tf_cfg["use_train_augs"]

        else get_val_transforms(tf_cfg["image_size"])

    )



    val_ds = CalorieDataset(

        data_dir=data_cfg["data_dir"],

        split=data_cfg["val_split"],

        dish_csv=data_cfg["dish_csv"],

        ingr_csv=data_cfg["ingredients_csv"],

        transforms=get_val_transforms(tf_cfg["image_size"])

    )



    train_loader = DataLoader(

        train_ds,

        batch_size=data_cfg["batch_size"],

        shuffle=True,

        num_workers=data_cfg["num_workers"],

        pin_memory=True,

        collate_fn=lambda x: collate_fn(x, tokenizer)

    )



    val_loader = DataLoader(

        val_ds,

        batch_size=data_cfg["batch_size"],

        shuffle=False,

        num_workers=data_cfg["num_workers"],

        pin_memory=True,

        collate_fn=lambda x: collate_fn(x, tokenizer)

    )



    return train_loader, val_loader





class AverageMeter:

    def __init__(self):

        self.reset()



    def reset(self):

        self.val = 0

        self.sum = 0

        self.count = 0



    def update(self, value, n=1):

        self.val = value

        self.sum += value * n

        self.count += n



    @property

    def avg(self):

        return self.sum / max(1, self.count)





def mae(preds, targets):

    return torch.mean(torch.abs(preds - targets)).item()





def train_one_epoch(model, loader, optimizer, device):

    model.train()

    loss_meter = AverageMeter()



    for batch in tqdm(loader, desc="Train"):

        images = batch["images"].to(device)

        ids = batch["input_ids"].to(device)

        mask = batch["attention_mask"].to(device)

        targets = batch["targets"].float().to(device)



        preds = model(images, ids, mask)

        loss = torch.nn.functional.l1_loss(preds, targets)



        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



        loss_meter.update(loss.item(), images.size(0))



    return loss_meter.avg







@torch.no_grad()

def validate(model, loader, device):

    model.eval()

    loss_meter = AverageMeter()



    for batch in tqdm(loader, desc="Val"):

        images = batch["images"].to(device)

        ids = batch["input_ids"].to(device)

        mask = batch["attention_mask"].to(device)

        targets = batch["targets"].float().to(device)



        preds = model(images, ids, mask)

        loss = torch.nn.functional.l1_loss(preds, targets)



        loss_meter.update(loss.item(), images.size(0))



    return loss_meter.avg







def train_model(config: Dict):



    set_seed(42)



    device = torch.device(config["training"]["device"])



    lr = config["training"]["lr"]

    epochs = config["training"]["epochs"]

    wd = config["training"]["weight_decay"]

    save_dir = config["logging"]["save_dir"]

    os.makedirs(save_dir, exist_ok=True)



  

    resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

    resnet.fc = torch.nn.Identity()

    image_dim = 2048



   

    bert = BertModel.from_pretrained(config["model"]["text_model"])

    text_dim = 768



    model = CalorieModel(

        resnet=resnet,

        bert=bert,

        image_dim=image_dim,

        text_dim=text_dim,

        hidden_dim=config["model"]["hidden_dim"],

        dropout=config["model"]["dropout"]

    ).to(device)



  

    train_loader, val_loader = build_dataloaders(config)



    

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=lr,

        weight_decay=wd

    )



    

    best_val = float("inf")

    best_path = os.path.join(save_dir, "best_model.pth")



    for epoch in range(1, epochs + 1):



        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        val_loss = validate(model, val_loader, device)



        print(f"Epoch {epoch}/{epochs} | Train MAE: {train_loss:.4f} | Val MAE: {val_loss:.4f}")



        if val_loss < best_val:

            best_val = val_loss

            torch.save(model.state_dict(), best_path)



    print(f"Training completed. Best MAE = {best_val:.4f}")