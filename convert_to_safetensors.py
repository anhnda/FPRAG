"""
Helper script to convert models to safetensors format for AWQ quantization.
Some models (like MiniCPM-2B-sft-bf16) are stored in PyTorch bin format,
but AutoAWQ requires safetensors format.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import os


def convert_model_to_safetensors(model_name, output_dir):
    """
    Convert a model from PyTorch format to safetensors format.

    Args:
        model_name: HuggingFace model name or path
        output_dir: Directory to save converted model
    """
    print("=" * 80)
    print("Converting Model to Safetensors Format")
    print("=" * 80)
    print(f"Source model: {model_name}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load model
    print("Loading model (this may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="auto"
    )

    # Save in safetensors format
    print(f"\nSaving model to {output_dir} in safetensors format...")
    os.makedirs(output_dir, exist_ok=True)

    model.save_pretrained(
        output_dir,
        safe_serialization=True,  # This forces safetensors format
        max_shard_size="5GB"
    )

    tokenizer.save_pretrained(output_dir)

    print("\n" + "=" * 80)
    print("✅ Conversion complete!")
    print("=" * 80)
    print(f"Model saved to: {output_dir}")
    print(f"\nYou can now use this path in quantize_minicpm_awq.py:")
    print(f'  model_name = "{output_dir}"')
    print("=" * 80)


if __name__ == "__main__":
    # Configuration
    source_model = "openbmb/MiniCPM-2B-sft-bf16"
    output_directory = "./models/MiniCPM-2B-sft-bf16-safetensors"

    convert_model_to_safetensors(source_model, output_directory)
