import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights


class CalorieRegressor(nn.Module):
    """
    Модель для предсказания калорий:
    - фичи с картинки (ResNet)
    - multi-hot вектор ингредиентов
    - масса блюда (total_mass)
    """

    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.3,
        num_ingredients: int | None = None,
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.num_ingredients = num_ingredients

        # backbone по картинке
        if backbone_name == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = resnet18(weights=weights)
            in_features_img = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif backbone_name == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = resnet50(weights=weights)
            in_features_img = backbone.fc.in_features
            backbone.fc = nn.Identity()
        else:
            raise ValueError(f"Неизвестный backbone: {backbone_name}")

        self.backbone = backbone

        # небольшая "голова" для ингредиентов
        if num_ingredients is not None and num_ingredients > 0:
            self.ingr_head = nn.Sequential(
                nn.Linear(num_ingredients, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            ingr_dim = 256
        else:
            self.ingr_head = None
            ingr_dim = 0

        # +1 признак под массу
        fused_dim = in_features_img + ingr_dim + 1

        self.head = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1),
        )

    def forward(
        self,
        images: torch.Tensor,
        ingredients_vec: torch.Tensor | None = None,
        mass: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        images: [B, 3, H, W]
        ingredients_vec: [B, num_ingredients] (multi-hot) или None
        mass: [B] или [B, 1]
        """
        # признаки с картинки
        img_feats = self.backbone(images)  # [B, F]

        feats_list = [img_feats]

        # признаки по ингредиентам
        if self.ingr_head is not None and ingredients_vec is not None:
            ingr_feats = self.ingr_head(ingredients_vec)  # [B, 256]
            feats_list.append(ingr_feats)

        # признак массы
        if mass is not None:
            if mass.dim() == 1:
                mass = mass.view(-1, 1)
            feats_list.append(mass)

        x = torch.cat(feats_list, dim=1)
        out = self.head(x)          # [B, 1]
        return out.squeeze(-1)      # [B]
