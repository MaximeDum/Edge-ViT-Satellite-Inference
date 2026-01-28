import torch
import os
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


def export_model():
    # 1. Model selection (SegFormer is a ViT optimized for segmentation)
    model_name = "nvidia/segformer-b0-finetuned-ade-512-512"

    print(f"Loading model: {model_name}...")
    processor = SegformerImageProcessor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name)
    model.eval()

    # 2. Prepare dummy input (1 sample, 3 channels, 512x512 pixels)
    dummy_input = torch.randn(1, 3, 512, 512)

    # 3. Create output directory if it doesn't exist
    output_dir = "./models"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, "segformer_vit.onnx")

    # 4. Export to ONNX format
    print("Exporting to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=18,  # Use 18 to avoid conversion errors
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print(f"Success! Model saved at: {output_path}")


if __name__ == "__main__":
    export_model()
