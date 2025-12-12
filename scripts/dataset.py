import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class DishDataset(Dataset):
    def __init__(
            self,
            dish_csv,
            ingredients_csv,
            images_dir,
            split_name,
            transforms=None,
    ):
        super().__init__()

        self.dish_csv = dish_csv
        self.ingredients_csv = ingredients_csv
        self.images_dir = images_dir
        self.split_name = split_name
        self.transforms = transforms

        # таблица блюд
        self.dish_df = pd.read_csv(self.dish_csv)
        self.dish_df = self.dish_df[self.dish_df["split"] == self.split_name].reset_index(drop=True)

        # таблица ингредиентов
        self.ingredients_df = pd.read_csv(self.ingredients_csv)

        # словарь id_ингредиента -> индекс (0..N-1)
        # в ingredients.csv колонка "id" — число
        # в dish.csv значения вида "ingr_0000000123"
        self.ingr2idx = {}
        for i, row in self.ingredients_df.iterrows():
            ingr_id = int(row["id"])
            self.ingr2idx[ingr_id] = i

        # ВАЖНО: теперь у датасета есть num_ingredients
        self.num_ingredients = len(self.ingr2idx)

    def __len__(self):
        return len(self.dish_df)

    def _get_image_path(self, dish_id):
        return os.path.join(self.images_dir, str(dish_id), "rgb.png")

    def _build_ingredients_vector(self, ingredients_raw: str) -> torch.Tensor:
        """
        Строим multi-hot вектор по ингредиентам:
        длина = число ингредиентов, 1 там, где ингредиент есть в блюде.
        """
        vec = torch.zeros(self.num_ingredients, dtype=torch.float32)

        if not isinstance(ingredients_raw, str):
            return vec

        parts = ingredients_raw.split(";")
        for p in parts:
            p = p.strip()
            if not p:
                continue
            # формат "ingr_0000000123"
            try:
                num = int(p.replace("ingr_", ""))
            except ValueError:
                continue

            idx = self.ingr2idx.get(num)
            if idx is not None:
                vec[idx] = 1.0

        return vec

    def __getitem__(self, idx):
        row = self.dish_df.iloc[idx]

        dish_id = row["dish_id"]
        calories = float(row["total_calories"])
        total_mass = float(row["total_mass"])
        ingredients_raw = row["ingredients"]

        img_path = self._get_image_path(dish_id)
        image = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            image = self.transforms(image)

        ingredients_vec = self._build_ingredients_vector(ingredients_raw)

        sample = {
            "image": image,                              # тензор [C, H, W]
            "ingredients": str(ingredients_raw),        # сырая строка
            "ingredients_vec": ingredients_vec,         # тензор [num_ingredients]
            "mass": torch.tensor(total_mass, dtype=torch.float32),
            "calories": torch.tensor(calories, dtype=torch.float32),
            "dish_id": dish_id,
        }
        return sample


def build_transforms(resize_height=224, resize_width=224, is_train=True):
    """
    Преобразования для картинок:
    - изменение размера
    - (для train) аугментации
    - перевод в тензор
    - нормализация под ResNet (ImageNet mean/std)
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if is_train:
        transforms = T.Compose([
            T.Resize((resize_height, resize_width)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
    else:
        transforms = T.Compose([
            T.Resize((resize_height, resize_width)),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])

    return transforms
