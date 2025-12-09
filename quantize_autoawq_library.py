"""
AutoAWQ Library-Based Quantization

This script uses the official AutoAWQ library to quantize MiniCPM-2B to 4-bit precision.
It serves as a baseline for comparing against custom implementations like gw_awq_asym_l2.py.

AutoAWQ Features:
- Official AWQ implementation from MIT Han Lab
- Activation-aware weight quantization with per-channel scales
- Optimized CUDA kernels for INT4 inference
- Standard group-wise quantization (group_size=128)

Comparison Points:
- Library AWQ: Uses official algorithm with optimized kernels
- Custom gw_awq_asym_l2.py: Uses asymmetric quantization + L2 salience
"""

import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import argparse
import os


def main():
    parser = argparse.ArgumentParser(
        description="AutoAWQ Library-Based Quantization for MiniCPM-2B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model-path", type=str, default="openbmb/MiniCPM-2B-sft-bf16",
                       help="Path or HuggingFace model ID")
    parser.add_argument("--output-dir", type=str, default="./quantized_models/minicpm_autoawq",
                       help="Output directory for quantized model")
    parser.add_argument("--w-bit", type=int, default=4, help="Weight quantization bits")
    parser.add_argument("--q-group-size", type=int, default=128, help="Group size for quantization")
    parser.add_argument("--zero-point", action="store_true", help="Use zero point (asymmetric quantization)")
    parser.add_argument("--calib-samples", type=int, default=128, help="Number of calibration samples")
    args = parser.parse_args()

    print("=" * 80)
    print("AutoAWQ Library-Based Quantization")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Weight bits: {args.w_bit}")
    print(f"Group size: {args.q_group_size}")
    print(f"Zero point: {args.zero_point}")
    print(f"Calibration samples: {args.calib_samples}")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    try:
        model = AutoAWQForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            safetensors=True,
            device_map="cuda"
        )
    except OSError as e:
        if "model.safetensors" in str(e):
            print("\n" + "=" * 80)
            print("ERROR: Model requires safetensors format")
            print("=" * 80)
            print("AutoAWQ requires models in safetensors format, but this model")
            print("is stored in PyTorch bin format.")
            print("\nSOLUTION:")
            print("1. Convert the model to safetensors format:")
            print(f"   python convert_to_safetensors.py --input-model {args.model_path} --output-dir ./models/MiniCPM-2B-safetensors")
            print("\n2. Then run quantization with the converted model:")
            print("   python quantize_autoawq_library.py --model-path ./models/MiniCPM-2B-safetensors")
            print("\nAlternatively, use a pre-converted safetensors model from HuggingFace.")
            print("=" * 80)
            exit(1)
        else:
            raise

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )

    # Get model size before quantization
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb_before = (param_size + buffer_size) / 1024**2
    print(f"Model size before quantization: {size_mb_before:.2f} MB")

    # Quantization config
    quant_config = {
        "zero_point": args.zero_point,
        "q_group_size": args.q_group_size,
        "w_bit": args.w_bit,
        "version": "GEMM"  # Use GEMM for better compatibility
    }

    print("\n" + "=" * 80)
    print("Quantization Configuration")
    print("=" * 80)
    for key, value in quant_config.items():
        print(f"  {key}: {value}")
    print("=" * 80)

    # Quantize model
    print("\n" + "=" * 80)
    print("Starting Quantization...")
    print("=" * 80)
    print("This will:")
    print("  1. Load calibration data from WikiText-2")
    print("  2. Collect activation statistics")
    print("  3. Compute per-channel importance scores")
    print("  4. Apply group-wise INT4 quantization")
    print("=" * 80)

    try:
        model.quantize(
            tokenizer,
            quant_config=quant_config,
            calib_data="wikitext",  # Use WikiText-2 for calibration
            n_samples=args.calib_samples,
            split="train"
        )

        print("\n✅ Quantization successful!")

    except Exception as e:
        print(f"\n❌ Quantization failed: {e}")
        print("\nTroubleshooting:")
        print("  - Ensure you have enough GPU memory (at least 16GB)")
        print("  - Try reducing --calib-samples")
        print("  - Check that the model supports AutoAWQ")
        raise

    # Get model size after quantization
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    size_mb_after = (param_size + buffer_size) / 1024**2

    print("\n" + "=" * 80)
    print("Quantization Statistics")
    print("=" * 80)
    print(f"Model size before: {size_mb_before:.2f} MB")
    print(f"Model size after: {size_mb_after:.2f} MB")
    print(f"Compression ratio: {size_mb_before / size_mb_after:.2f}x")
    print("=" * 80)

    # Save quantized model
    print(f"\nSaving quantized model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)

    model.save_quantized(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print("\n" + "=" * 80)
    print("QUANTIZATION COMPLETE!")
    print("=" * 80)
    print(f"Quantized model saved to: {args.output_dir}")
    print("\nNext steps:")
    print("  1. Run comparison script to evaluate against custom implementations")
    print("  2. Test inference speed with quantized model")
    print("  3. Evaluate perplexity on WikiText-2 validation set")
    print("=" * 80)


if __name__ == "__main__":
    main()
