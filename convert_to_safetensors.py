"""
Convert PyTorch Model to Safetensors Format

AutoAWQ requires models in safetensors format. This script converts
MiniCPM-2B from PyTorch bin format to safetensors format.

Usage:
    python convert_to_safetensors.py
    python convert_to_safetensors.py --input-model openbmb/MiniCPM-2B-sft-bf16 --output-dir ./models/MiniCPM-2B-safetensors
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
from safetensors.torch import save_file


def convert_to_safetensors(model_path, output_dir):
    """
    Convert a PyTorch model to safetensors format.

    Args:
        model_path: HuggingFace model ID or local path
        output_dir: Directory to save converted model
    """
    print("=" * 80)
    print("Converting PyTorch Model to Safetensors Format")
    print("=" * 80)
    print(f"Input model: {model_path}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print("\nLoading model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise

    # Load tokenizer
    print("\nLoading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        print("✅ Tokenizer loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        raise

    # Get model state dict
    print("\nExtracting model state dict...")
    state_dict = model.state_dict()

    # Get model size
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    size_mb = param_size / 1024**2
    print(f"Model size: {size_mb:.2f} MB")
    print(f"Number of tensors: {len(state_dict)}")

    # Handle shared tensors (e.g., lm_head.weight and model.embed_tokens.weight)
    print("\nChecking for shared tensors...")
    shared_tensors = {}
    tensor_ptrs = {}

    for key, tensor in state_dict.items():
        ptr = tensor.data_ptr()
        if ptr in tensor_ptrs:
            shared_tensors[key] = tensor_ptrs[ptr]
            print(f"  Found shared tensor: {key} shares memory with {tensor_ptrs[ptr]}")
        else:
            tensor_ptrs[ptr] = key

    # Clone shared tensors to make them independent
    if shared_tensors:
        print(f"\nCloning {len(shared_tensors)} shared tensor(s) to make them independent...")
        for key in shared_tensors:
            state_dict[key] = state_dict[key].clone()
            print(f"  ✅ Cloned {key}")
    else:
        print("  No shared tensors found")

    # Save as safetensors
    print(f"\nSaving model to safetensors format...")
    safetensors_path = os.path.join(output_dir, "model.safetensors")

    try:
        save_file(state_dict, safetensors_path)
        print(f"✅ Saved model weights to {safetensors_path}")
    except Exception as e:
        print(f"❌ Failed to save safetensors: {e}")
        raise

    # Save config
    print("\nSaving model config...")
    model.config.save_pretrained(output_dir)
    print(f"✅ Saved config to {output_dir}")

    # Save tokenizer
    print("\nSaving tokenizer...")
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Saved tokenizer to {output_dir}")

    # Copy custom modeling files if they exist
    print("\nCopying custom modeling files...")
    try:
        import glob
        import shutil
        from pathlib import Path

        # Find the cached model directory
        # HuggingFace caches models in ~/.cache/huggingface/hub/models--{org}--{model}/snapshots/{hash}/
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"

        # Search for the model cache directory
        model_cache_pattern = model_path.replace("/", "--")
        model_dirs = list(cache_dir.glob(f"models--{model_cache_pattern}"))

        if model_dirs:
            # Get the latest snapshot
            snapshot_dirs = list(model_dirs[0].glob("snapshots/*"))
            if snapshot_dirs:
                source_dir = snapshot_dirs[0]  # Use first (usually only one)

                # Copy all Python files (custom modeling code)
                python_files = list(source_dir.glob("*.py"))
                copied_count = 0
                for py_file in python_files:
                    dest_file = os.path.join(output_dir, py_file.name)
                    shutil.copy(py_file, dest_file)
                    print(f"  ✅ Copied {py_file.name}")
                    copied_count += 1

                if copied_count == 0:
                    print("  ℹ️  No custom Python files found (model may not need them)")
            else:
                print("  ⚠️  No snapshot found in cache")
        else:
            print(f"  ℹ️  Model cache not found, checking if model uses custom code...")
            # If model doesn't use custom code, this is fine
            if hasattr(model.config, 'auto_map'):
                print("  ⚠️  Model requires custom code but cache not accessible")
            else:
                print("  ✅ Model doesn't require custom code")

    except Exception as e:
        print(f"  ⚠️  Could not copy custom files: {e}")
        # Not critical, continue anyway

    # Verify the saved files
    print("\n" + "=" * 80)
    print("Verifying saved files...")
    print("=" * 80)

    required_files = ["model.safetensors", "config.json", "tokenizer_config.json"]
    all_present = True

    for filename in required_files:
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / 1024**2
            print(f"✅ {filename} ({size_mb:.2f} MB)")
        else:
            print(f"❌ {filename} (missing)")
            all_present = False

    if all_present:
        print("\n" + "=" * 80)
        print("CONVERSION SUCCESSFUL!")
        print("=" * 80)
        print(f"Model saved to: {output_dir}")
        print("\nYou can now use this model with AutoAWQ:")
        print(f"  python quantize_autoawq_library.py --model-path {output_dir}")
        print("=" * 80)
    else:
        print("\n⚠️  Some files are missing. Please check for errors above.")

    return all_present


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch model to safetensors format for AutoAWQ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input-model", type=str, default="openbmb/MiniCPM-2B-sft-bf16",
                       help="Input model path or HuggingFace ID")
    parser.add_argument("--output-dir", type=str, default="./models/MiniCPM-2B-safetensors",
                       help="Output directory for converted model")
    args = parser.parse_args()

    try:
        success = convert_to_safetensors(args.input_model, args.output_dir)
        if not success:
            exit(1)
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
