# Оценка калорийности блюд по фото

Учебный проект по построению модели, которая по фотографии блюда и списку ингредиентов оценивает его калорийность.  
Данные: изображения тарелок, состав блюда, масса порции и целевая переменная total_calories.

Цель проекта: построить DL-модель, которая на тестовой выборке даёт MAE меньше 50.
---

## Данные

Структура папки `data`:

- `data/dish.csv`  
  - `dish_id`  
  - `total_calories`  
  - `total_mass`  
  - `ingredients` (список id ингредиентов)  
  - `split` (train или test)

- `data/ingredients.csv`  
  - `id`  
  - `ingr` (название ингредиента)

- `data/images/`  
  - каталоги по `dish_id`, внутри каждого лежит `rgb.png`  
  - пример пути: `data/images/0000000123/rgb.png`

Файлы с данными в репозиторий не выкладываются, сохраняется только структура и пути.

---

## Структура проекта

```text
Nutrition/
├── configs/
│   └── base_config.yaml
├── data/
│   ├── dish.csv
│   ├── ingredients.csv
│   └── images/
├── logs/
├── models/
├── notebooks/
│   └── notebook.ipynb
├── scripts/
│   ├── __init__.py
│   ├── dataset.py
│   ├── inference.py
│   ├── models.py
│   └── utils.py
└── README.md
