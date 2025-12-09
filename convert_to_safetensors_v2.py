"""
Convert PyTorch Model to Safetensors Format (v2 - Simplified)

This version uses snapshot_download to ensure all files including custom code are downloaded.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
from safetensors.torch import save_file
from huggingface_hub import snapshot_download
import shutil


def convert_to_safetensors(model_path, output_dir):
    """
    Convert a PyTorch model to safetensors format.
    Uses snapshot_download to ensure all files are present.
    """
    print("=" * 80)
    print("Converting PyTorch Model to Safetensors Format (v2)")
    print("=" * 80)
    print(f"Input model: {model_path}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Download all model files including custom code
    print("\nStep 1: Downloading all model files (including custom code)...")
    try:
        local_model_path = snapshot_download(
            repo_id=model_path,
            ignore_patterns=["*.safetensors"],  # Don't download existing safetensors
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        print(f"✅ Downloaded model files to {output_dir}")
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        raise

    # Step 2: Load model
    print("\nStep 2: Loading model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            output_dir,  # Load from the directory we just downloaded to
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise

    # Step 3: Extract state dict
    print("\nStep 3: Extracting model state dict...")
    state_dict = model.state_dict()

    # Get model size
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    size_mb = param_size / 1024**2
    print(f"Model size: {size_mb:.2f} MB")
    print(f"Number of tensors: {len(state_dict)}")

    # Step 4: Handle shared tensors
    print("\nStep 4: Checking for shared tensors...")
    shared_tensors = {}
    tensor_ptrs = {}

    for key, tensor in state_dict.items():
        ptr = tensor.data_ptr()
        if ptr in tensor_ptrs:
            shared_tensors[key] = tensor_ptrs[ptr]
            print(f"  Found shared tensor: {key} shares memory with {tensor_ptrs[ptr]}")
        else:
            tensor_ptrs[ptr] = key

    # Clone shared tensors
    if shared_tensors:
        print(f"\nCloning {len(shared_tensors)} shared tensor(s)...")
        for key in shared_tensors:
            state_dict[key] = state_dict[key].clone()
            print(f"  ✅ Cloned {key}")
    else:
        print("  No shared tensors found")

    # Step 5: Save as safetensors
    print(f"\nStep 5: Saving model to safetensors format...")
    safetensors_path = os.path.join(output_dir, "model.safetensors")

    try:
        save_file(state_dict, safetensors_path)
        print(f"✅ Saved model weights to {safetensors_path}")
    except Exception as e:
        print(f"❌ Failed to save safetensors: {e}")
        raise

    # Clean up PyTorch bin files (optional)
    print("\nStep 6: Cleaning up PyTorch bin files...")
    bin_files = [f for f in os.listdir(output_dir) if f.endswith('.bin')]
    for bin_file in bin_files:
        bin_path = os.path.join(output_dir, bin_file)
        try:
            os.remove(bin_path)
            print(f"  🗑️  Removed {bin_file}")
        except Exception as e:
            print(f"  ⚠️  Could not remove {bin_file}: {e}")

    # Verify files
    print("\n" + "=" * 80)
    print("Verifying saved files...")
    print("=" * 80)

    required_files = ["model.safetensors", "config.json"]
    all_files = os.listdir(output_dir)

    print("\nRequired files:")
    for filename in required_files:
        if filename in all_files:
            filepath = os.path.join(output_dir, filename)
            size_mb = os.path.getsize(filepath) / 1024**2
            print(f"  ✅ {filename} ({size_mb:.2f} MB)")
        else:
            print(f"  ❌ {filename} (missing)")
            return False

    print("\nCustom Python files:")
    py_files = [f for f in all_files if f.endswith('.py')]
    if py_files:
        for py_file in py_files:
            print(f"  ✅ {py_file}")
    else:
        print("  ℹ️  No custom Python files (model may not need them)")

    print("\nTokenizer files:")
    tokenizer_files = [f for f in all_files if 'tokenizer' in f.lower()]
    for tok_file in tokenizer_files:
        print(f"  ✅ {tok_file}")

    print("\n" + "=" * 80)
    print("CONVERSION SUCCESSFUL!")
    print("=" * 80)
    print(f"Model saved to: {output_dir}")
    print("\nAll files present:")
    for f in sorted(all_files)[:20]:  # Show first 20 files
        print(f"  - {f}")
    if len(all_files) > 20:
        print(f"  ... and {len(all_files) - 20} more files")
    print("\nYou can now use this model with AutoAWQ:")
    print(f"  python quantize_autoawq_library.py --model-path {output_dir}")
    print("=" * 80)

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert PyTorch model to safetensors format (v2)",
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
            print("\n⚠️  Conversion completed with warnings")
            exit(1)
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
