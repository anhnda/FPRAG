"""Quick test to verify hooks work on MiniCPM model."""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    'openbmb/MiniCPM-2B-sft-bf16',
    torch_dtype=torch.float16,
    device_map='cuda',
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-2B-sft-bf16', trust_remote_code=True)

call_count = [0]

def hook_fn(m, i, o):
    call_count[0] += 1
    print(f"  Hook called! Output shape: {o[0].shape if isinstance(o, tuple) else o.shape}")

# Register hook on layer 0
print("\nRegistering hook on model.layers.0...")
for name, module in model.named_modules():
    if name == 'model.layers.0':
        hook = module.register_forward_hook(hook_fn)
        print(f"✓ Hook registered on {name}")
        break

# Run inference
print("\nRunning inference...")
model.eval()
inputs = tokenizer('Hello world', return_tensors='pt')
inputs = {k: v.to('cuda') for k, v in inputs.items()}

with torch.no_grad():
    output = model(**inputs, use_cache=False)

print(f"\n✓ Inference complete")
print(f"✓ Hook was called {call_count[0]} times")

if call_count[0] > 0:
    print("\n SUCCESS: Hooks are working!")
else:
    print("\n⚠ PROBLEM: Hook was never called despite successful inference")

hook.remove()
