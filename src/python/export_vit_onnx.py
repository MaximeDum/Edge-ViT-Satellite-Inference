import torch
import os
from transformers import SegformerForSemanticSegmentation

# 1. Utiliser le chemin absolu pour éviter la confusion avec le Hub
# os.path.abspath convertit "./models/..." en "/Users/maximedumont/Projects/..."
model_path = "segformer-finetuned-local/checkpoint-80/"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Le dossier n'existe pas : {model_path}")

print(f"Chargement du modèle local depuis : {model_path}")

# Chargement du modèle avec local_files_only pour forcer Python à ne pas aller sur Internet
model = SegformerForSemanticSegmentation.from_pretrained(
    model_path, local_files_only=True
)
model.eval()

# 2. Dummy input (format standard SegFormer B0)
dummy_input = torch.randn(1, 3, 512, 512)

# 3. Export ONNX
output_onnx = "/models/segformer_satellite_v1.onnx"
torch.onnx.export(
    model,
    dummy_input,
    output_onnx,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)

print(f"Succès ! Ton fichier unique est prêt : {output_onnx}")
