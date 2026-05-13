                                                                                
Llama-3-8B
Dataset         Standard AdaRound  AdaRound SFA    Delta        Winner             
--------------------------------------------------------------------------------
WikiText-2      5.9802          5.9384               +0.704%  Tie               
C4              9.6267          9.5433               +0.875%  Standard          

Mistral-7B-v0.3
--------------------------------------------------------------------------------
WikiText-2      5.0012          4.9935               +0.153%  Tie       
C4              7.8180          7.8069               +0.142%  Tie       



  python adaround_xl.py \
    --model-path /home/DATA/prometheus/anh/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796 \
    --output-dir ./quantized_models/Qwen2.5-7B_ar \
    --n-calib 128 \
    --adaround-iters 10000 \
    --adaround-lr 1e-3 \
    --layer-batch-size 16 --skip-lmhead


  python adaround_flip_xl.py \
    --model-path /home/DATA/prometheus/anh/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B/snapshots/d149729398750b98c0af14eb82c78cfe92750796 \
    --output-dir ./quantized_models/Qwen2.5-7B_arf \
    --n-calib 128 \
    --adaround-iters 10000 \
    --adaround-lr 1e-3 \
    --layer-batch-size 16 --skip-lmhead



              python compare_slicing.py \
    --heuristic-path ./quantized_models/Qwen2.5-7B_ar \
    --standard-path ./quantized_models/Qwen2.5-7B_arf 


  python adaround_xl.py \
    --model-path /home/DATA/prometheus/anh/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3/snapshots/caa1feb0e54d415e2df31207e5f4e273e33509b1 \
    --output-dir ./quantized_models/Mistral-7B_ar \
    --n-calib 128 \
    --adaround-iters 10000 \
    --adaround-lr 1e-3 \
    --layer-batch-size 16 --skip-lmhead


  python adaround_flip_xl.py \
    --model-path /home/DATA/prometheus/anh/.cache/huggingface/hub/models--mistralai--Mistral-7B-v0.3/snapshots/caa1feb0e54d415e2df31207e5f4e273e33509b1 \
    --output-dir ./quantized_models/Mistral-7B_arf \
    --n-calib 128 \/home/DATA/prometheus/anh/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920 \
 
    --adaround-iters 10000 \
    --adaround-lr 1e-3 \
    --layer-batch-size 16 --skip-lmhead
        models--mistralai--Mistral-7B-v0.3/snapshots/caa1feb0e54d415e2df31207e5f4e273e33509b1


              python compare_slicing.py \
    --heuristic-path ./quantized_models/Mistral-7B_ar \
    --standard-path ./quantized_models/Mistral-7B_arf 

  python adaround_xl.py \
    --model-path    --output-dir ./quantized_models/Llama-3-8B_ar \
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