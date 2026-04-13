  python adaround_xl.py \
    --model-path /home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920 \
    --output-dir ./quantized_models/Llama-3-8B_ar \
    --n-calib 128 \
    --adaround-iters 10000 \
    --adaround-lr 1e-3 \
    --layer-batch-size 16 --skip-lmhead


  python adaround_flip_xl.py \
    --model-path /home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920 \
    --output-dir ./quantized_models/Llama-3-8B_arf \
    --n-calib 128 \
    --adaround-iters 10000 \
    --adaround-lr 1e-3 \
    --layer-batch-size 16 --skip-lmhead


              python compare_slicing.py \
    --heuristic-path ./quantized_models/Llama-3-8B_ar \
    --standard-path ./quantized_models/Llama-3-8B_arf 