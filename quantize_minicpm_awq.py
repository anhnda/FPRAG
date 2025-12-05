from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM
from datasets import load_dataset
import os


def load_wikitext2(split="train", n_samples=None):
    """Load WikiText-2 dataset."""
    print(f"Loading WikiText-2 {split} dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)

    # Filter out empty texts and prepare for AWQ format
    texts = [item['text'] for item in dataset if len(item['text'].strip()) > 0]

    if n_samples:
        texts = texts[:n_samples]

    return texts


def prepare_calibration_data(texts, tokenizer, n_samples=500, max_length=512):
    """
    Prepare calibration data in the format expected by AutoAWQ.

    Args:
        texts: List of text samples
        tokenizer: Tokenizer
        n_samples: Number of samples to use
        max_length: Maximum sequence length

    Returns:
        List of tokenized samples
    """
    print(f"Preparing {min(n_samples, len(texts))} calibration samples...")

    calib_data = []
    for text in texts[:n_samples]:
        try:
            tokens = tokenizer(
                text,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=False
            )
            calib_data.append(tokens)
        except Exception as e:
            print(f"Error tokenizing sample: {e}")
            continue

    return calib_data


def main():
    # Configuration
    # NOTE: If you get "model.safetensors not found" error, run convert_to_safetensors.py first
    # to convert the model, then use the local path here
    model_name = "openbmb/MiniCPM-2B-sft-bf16"  # MiniCPM-2.4B
    # model_name = "./models/MiniCPM-2B-sft-bf16-safetensors"  # Use this if converted locally
    n_calib_samples = 500
    output_dir = "./quantized_models/minicpm_awq"

    # AWQ Configuration
    quant_config = {
        "zero_point": True,
        "q_group_size": 128,  # Quantization group size
        "w_bit": 4,  # 4-bit quantization
        "version": "GEMM"  # Use GEMM kernel
    }

    print("=" * 60)
    print("AWQ Quantization for MiniCPM-2.4")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Calibration samples: {n_calib_samples}")
    print(f"Quantization config: {quant_config}")
    print("=" * 60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load model for AWQ
    print("\nLoading model for AWQ quantization...")
    try:
        model = AutoAWQForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_cache=False
        )
    except OSError as e:
        if "safetensors" in str(e):
            print("\n" + "=" * 80)
            print("ERROR: Model requires safetensors format")
            print("=" * 80)
            print("AutoAWQ requires models in safetensors format.")
            print("\nSolution:")
            print("1. Run the conversion script first:")
            print("   python convert_to_safetensors.py")
            print("\n2. Then update model_name in this script to:")
            print("   model_name = './models/MiniCPM-2B-sft-bf16-safetensors'")
            print("\n3. Run this script again")
            print("=" * 80)
            raise
        raise

    # Load calibration data
    calib_texts = load_wikitext2(split="train", n_samples=n_calib_samples)

    # AWQ expects just raw text data
    print(f"\nCalibration dataset size: {len(calib_texts)}")

    # Quantize model
    print("\nStarting AWQ quantization...")
    print("This may take a while...")

    model.quantize(
        tokenizer,
        quant_config=quant_config,
        calib_data=calib_texts
    )

    print("\nQuantization complete!")

    # Save quantized model
    print(f"\nSaving quantized model to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)

    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\nDone!")
    print(f"Quantized model saved to: {output_dir}")
    print("\nModel info:")
    print(f"  - Quantization: 4-bit AWQ")
    print(f"  - Group size: {quant_config['q_group_size']}")
    print(f"  - Zero point: {quant_config['zero_point']}")


if __name__ == "__main__":
    main()
