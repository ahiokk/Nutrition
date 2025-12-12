import os
import random
from typing import Dict, Any

import numpy as np
import yaml
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from .dataset import DishDataset, build_transforms
from .models import CalorieRegressor


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_path(rel_path: str) -> str:
    if os.path.isabs(rel_path):
        return rel_path

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(project_root, rel_path)


def get_dataloaders(cfg: Dict[str, Any]):
    """Создаём train и test DataLoader-ы по конфигу."""
    data_cfg = cfg["data"]
    loader_cfg = cfg["loader"]
    augment_cfg = cfg.get("augment", {})

    dish_csv_path = resolve_path(data_cfg["dish_csv"])
    ingredients_csv_path = resolve_path(data_cfg["ingredients_csv"])
    images_dir_path = resolve_path(data_cfg["images_dir"])

    train_transforms = build_transforms(
        resize_height=augment_cfg.get("resize_height", 224),
        resize_width=augment_cfg.get("resize_width", 224),
        is_train=True,
    )
    test_transforms = build_transforms(
        resize_height=augment_cfg.get("resize_height", 224),
        resize_width=augment_cfg.get("resize_width", 224),
        is_train=False,
    )

    train_dataset = DishDataset(
        dish_csv=dish_csv_path,
        ingredients_csv=ingredients_csv_path,
        images_dir=images_dir_path,
        split_name=data_cfg.get("train_split_name", "train"),
        transforms=train_transforms,
    )

    test_dataset = DishDataset(
        dish_csv=dish_csv_path,
        ingredients_csv=ingredients_csv_path,
        images_dir=images_dir_path,
        split_name=data_cfg.get("test_split_name", "test"),
        transforms=test_transforms,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=loader_cfg["batch_size"],
        shuffle=loader_cfg.get("shuffle_train", True),
        num_workers=loader_cfg.get("num_workers", 4),
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=loader_cfg["batch_size"],
        shuffle=False,
        num_workers=loader_cfg.get("num_workers", 4),
        pin_memory=True,
    )

    return train_loader, test_loader


def mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target))


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    log_interval: int = 50,
):
    model.train()
    running_loss = 0.0

    for batch_idx, batch in enumerate(tqdm(loader, desc="Train", leave=False)):
        images = batch["image"].to(device, non_blocking=True)
        calories = batch["calories"].to(device, non_blocking=True)
        ingredients_vec = batch["ingredients_vec"].to(device, non_blocking=True)
        mass = batch["mass"].to(device, non_blocking=True)

        optimizer.zero_grad()

        preds = model(images, ingredients_vec, mass)
        loss = mae_loss(preds, calories)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = running_loss / (batch_idx + 1)
            print(f"Batch [{batch_idx+1}/{len(loader)}] MAE loss: {avg_loss:.4f}")

    epoch_loss = running_loss / len(loader)
    return epoch_loss


@torch.no_grad()
def validate(
    model,
    loader: DataLoader,
    device: torch.device,
):
    model.eval()
    all_preds = []
    all_targets = []

    for batch in tqdm(loader, desc="Validate", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        calories = batch["calories"].to(device, non_blocking=True)
        ingredients_vec = batch["ingredients_vec"].to(device, non_blocking=True)
        mass = batch["mass"].to(device, non_blocking=True)

        preds = model(images, ingredients_vec, mass)

        all_preds.append(preds.detach().cpu())
        all_targets.append(calories.detach().cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    mae = mae_loss(all_preds, all_targets).item()
    return mae


def build_model(cfg: Dict[str, Any], device: torch.device, num_ingredients: int):
    """Создаём модель с учётом конфига и числа ингредиентов."""
    model_cfg = cfg.get("model", {})

    backbone_name = model_cfg.get("backbone_name", "resnet18")
    pretrained = model_cfg.get("pretrained", True)
    dropout = float(model_cfg.get("dropout", 0.3))

    print(
        f"Создаём модель: backbone={backbone_name}, "
        f"pretrained={pretrained}, dropout={dropout}, "
        f"num_ingredients={num_ingredients}"
    )

    model = CalorieRegressor(
        backbone_name=backbone_name,
        pretrained=pretrained,
        dropout=dropout,
        num_ingredients=num_ingredients,
    )
    model.to(device)
    return model


def train(config_path: str):
    cfg = load_config(config_path)
    train_cfg = cfg["train"]
    logging_cfg = cfg.get("logging", {})

    set_seed(train_cfg.get("seed", 42))

    device_str = train_cfg.get("device", "cuda")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print(f"Используем устройство: {device}")

    train_loader, test_loader = get_dataloaders(cfg)

    # берём число ингредиентов из train-датасета
    num_ingredients = train_loader.dataset.num_ingredients

    model = build_model(cfg, device, num_ingredients)

    lr = float(train_cfg["learning_rate"])
    weight_decay = float(train_cfg.get("weight_decay", 0.0))

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_mae = float("inf")
    save_dir = train_cfg.get("save_dir", "models")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, train_cfg.get("save_name", "best_model.pth"))

    num_epochs = train_cfg["num_epochs"]
    log_interval = logging_cfg.get("log_interval", 50)

    for epoch in range(1, num_epochs + 1):
        print(f"\n=== Эпоха {epoch}/{num_epochs} ===")
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            log_interval=log_interval,
        )
        val_mae = validate(model, test_loader, device)

        print(f"Эпоха {epoch}: train MAE={train_loss:.4f}, val MAE={val_mae:.4f}")

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), save_path)
            print(f"Найдена лучшая модель, MAE={best_mae:.4f}. Сохраняем в {save_path}")

    print(f"\nОбучение завершено. Лучшая MAE на валидации: {best_mae:.4f}")
