import os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize_model():
    # Paths to your models
    input_model_path = "models/segformer_vit.onnx"
    output_model_path = "models/segformer_vit_int8.onnx"

    # 1. Safety check: does the source model exist?
    if not os.path.exists(input_model_path):
        print(
            f"Error: {input_model_path} not found. Please run the export script first."
        )
        return

    print(f"Starting quantization for: {input_model_path}...")

    # 2. Apply Dynamic Quantization
    # This converts weights from Float32 to Int8, reducing size and speeding up CPU inference
    quantize_dynamic(
        model_input=input_model_path,
        model_output=output_model_path,
        weight_type=QuantType.QUInt8,  # Standard for CPU optimization
    )

    # 3. Compare and display results
    size_original = os.path.getsize(input_model_path) / (1024 * 1024)
    size_quantized = os.path.getsize(output_model_path) / (1024 * 1024)
    reduction = (1 - size_quantized / size_original) * 100

    print("-" * 30)
    print(f"Quantization Successful!")
    print(f"Original Model:  {size_original:.2f} MB")
    print(f"Quantized Model: {size_quantized:.2f} MB")
    print(f"Size Reduction:  {reduction:.1f}%")
    print("-" * 30)
    print(f"Quantized model saved at: {output_model_path}")


if __name__ == "__main__":
    # Ensure we are running from the project root
    # (The script expects a 'models/' folder in the current directory)
    quantize_model()
