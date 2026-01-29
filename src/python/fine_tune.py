import os
import numpy as np
import torch
from PIL import Image, ImageDraw
from datasets import Dataset
from transformers import (
    SegformerImageProcessor,
    SegformerForSemanticSegmentation,
    TrainingArguments,
    Trainer,
)

# 1. Configuration - CPU recommandé pour la stabilité du Fine-tuning sur Mac
device = "cpu"
model_id = "nvidia/mit-b0"


# 2. GÉNÉRATEUR DE DONNÉES SATELLITES SIMULÉES (3 classes)
def create_satellite_sample(size=512):
    # Fond (Nature)
    img = Image.new("RGB", (size, size), (34, 139, 34))
    mask = Image.new("L", (size, size), 0)  # 0 = Nature
    draw_img = ImageDraw.Draw(img)
    draw_mask = ImageDraw.Draw(mask)

    # Simuler une Route (Classe 1)
    x1, y1 = np.random.randint(0, size), 0
    x2, y2 = np.random.randint(0, size), size
    draw_img.line((x1, y1, x2, y2), fill=(100, 100, 100), width=25)
    draw_mask.line((x1, y1, x2, y2), fill=1, width=25)

    # Simuler un Plan d'eau (Classe 2)
    rx, ry = np.random.randint(50, 400), np.random.randint(50, 400)
    draw_img.ellipse((rx, ry, rx + 120, ry + 90), fill=(0, 0, 255))
    draw_mask.ellipse((rx, ry, rx + 120, ry + 90), fill=2)

    return img, mask


print("Génération du dataset synthétique (40 images)...")
imgs, msks = zip(*[create_satellite_sample() for _ in range(40)])
dataset = Dataset.from_dict(
    {"image": list(imgs), "label": list(msks)}
).train_test_split(test_size=0.2)

# 3. Préparation SegFormer
processor = SegformerImageProcessor.from_pretrained(model_id)


def transforms(example_batch):
    # On convertit tout en RGB/L et on laisse le processor gérer les tensors
    images = [x.convert("RGB") for x in example_batch["image"]]
    labels = [x.convert("L") for x in example_batch["label"]]
    inputs = processor(images, labels, return_tensors="pt")
    return inputs


dataset["train"].set_transform(transforms)
dataset["test"].set_transform(transforms)

# 4. Modèle (0: Nature, 1: Route, 2: Eau)
model = SegformerForSemanticSegmentation.from_pretrained(
    model_id, num_labels=3, ignore_mismatched_sizes=True
)

# 5. Entraînement - Paramètres de stabilité maximale
training_args = TrainingArguments(
    output_dir="segformer-finetuned-local",
    num_train_epochs=5,
    learning_rate=1e-4,
    per_device_train_batch_size=2,
    eval_strategy="epoch",
    logging_steps=5,
    remove_unused_columns=False,
    push_to_hub=False,
    # Paramètres de compatibilité Mac
    dataloader_pin_memory=False,
    use_cpu=True,  # Force le CPU pour éviter le RuntimeError de view/stride
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

# 6. Exécution
print(f"Lancement du fine-tuning sur {device} (Mode stabilité)...")
trainer.train()

# 7. Sauvegarde
save_path = "./models/segformer_finetuned"
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
processor.save_pretrained(save_path)

print(f"\nSuccès ! Modèle sauvegardé dans : {save_path}")
