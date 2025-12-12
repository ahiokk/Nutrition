import os
import torch
import pandas as pd
from torch.utils.data import DataLoader

from .dataset import DishDataset, build_transforms
from .models import CalorieRegressor


def load_model(
    model_path: str,
    device: torch.device,
    backbone_name: str,
    dropout: float,
    num_ingredients: int,
):
    model = CalorieRegressor(
        backbone_name=backbone_name,
        pretrained=False,    # при инференсе предобученные веса не нужны
        dropout=dropout,
        num_ingredients=num_ingredients,
    )
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def run_inference(
    dish_csv: str,
    ingredients_csv: str,
    images_dir: str,
    model_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    top_k: int = 5,
    backbone_name: str = "resnet18",
    dropout: float = 0.3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Инференс на устройстве:", device)

    test_dataset = DishDataset(
        dish_csv=dish_csv,
        ingredients_csv=ingredients_csv,
        images_dir=images_dir,
        split_name="test",
        transforms=build_transforms(is_train=False),
    )

    num_ingredients = test_dataset.num_ingredients

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = load_model(
        model_path=model_path,
        device=device,
        backbone_name=backbone_name,
        dropout=dropout,
        num_ingredients=num_ingredients,
    )

    all_preds = []
    all_targets = []
    all_dish_ids = []

    for batch in test_loader:
        images = batch["image"].to(device, non_blocking=True)
        calories = batch["calories"].to(device, non_blocking=True)
        ingredients_vec = batch["ingredients_vec"].to(device, non_blocking=True)
        mass = batch["mass"].to(device, non_blocking=True)
        dish_ids = batch["dish_id"]

        preds = model(images, ingredients_vec, mass)

        all_preds.append(preds.detach().cpu())
        all_targets.append(calories.detach().cpu())
        all_dish_ids.extend(dish_ids)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    mae = torch.mean(torch.abs(all_preds - all_targets)).item()
    print(f"MAE на test-сплите: {mae:.4f}")

    abs_errors = torch.abs(all_preds - all_targets).numpy()

    result_df = pd.DataFrame({
        "dish_id": all_dish_ids,
        "true_calories": all_targets.numpy(),
        "pred_calories": all_preds.numpy(),
        "abs_error": abs_errors,
    })

    worst_df = result_df.sort_values("abs_error", ascending=False).head(top_k)
    print(f"\nТоп-{top_k} блюд с наибольшей ошибкой:")
    print(worst_df)

    return mae, worst_df
