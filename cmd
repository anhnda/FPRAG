
  python gptq_stand_xl.py \
    --model-path /models/Llama-3-8B --output-dir ./quantized_models/Llama-3-8B_gptq

python compare_slicing.py \
    --heuristic-path ./quantized_models/Llama-3-8B_gptq \
    --standard-path ./quantized_models/Llama-3-8B_awq 


python gptq_stand_xl.py \
    --model-path /models/Llama-3-8B \
    --output-dir ./quantized_models/Llama-3-8B_gptq \
    --wbits 4 \
    --groupsize -1 \
    --nsamples 128 \
    --seqlen 2048 \
    --percdamp 0.01 \
    --asym \
    --act-order \
    --true-sequential 

python compare_slicing.py \
    --heuristic-path ./quantized_models/Llama-3-8B_gptq \
    --standard-path ./quantized_models/Llama-3-8B_awq 





python gptq_stand_xl.py \
    --model-path /models/Llama-3-8B \
    --output-dir ./quantized_models/Llama-3-8B_gptq \
    --percdamp 0.01 \
    --asym \
    --act-order \
    --true-sequential 

python gptq_js.py \
    --model-path /models/Llama-3-8B \
    --output-dir ./quantized_models/Llama-3-8B_gptq_js \
    --asym --act-order \
    --smart-flip \
    --knee-tolerance 0.0 \
    --max-flip-percent 0.01 --true-sequential 

python compare_slicing.py \
    --heuristic-path ./quantized_models/Llama-3-8B_gptq_js \
    --standard-path ./quantized_models/Llama-3-8B_gptq 



Dataset         Heuristic GPTQ   Standard GPTQ    Delta        Winner                                                         
--------------------------------------------------------------------------------                                            
WikiText-2      5.7831          5.7822               +0.016%  Tie                                                           
C4              9.2843          9.3216               -0.401%  Tie          

python awq_tfim_correction.py --model-path /models/Llama-3-8B --output-dir ./quantized_models/Llama-3-8B_q

python awq_stand_xl.py --model-path /models/Llama-3-8B --output-dir ./quantized_models/Llama-3-8B_awq

python compare_slicing.py \
    --heuristic-path ./quantized_models/Llama-3-8B_gptq \
    --standard-path ./quantized_models/Llama-3-8B_awq 

python awq_quantum_correction.py --model-path /models/Llama-3-8B --output-dir ./quantized_models/Llama-3-8B_qt

  python adaround_xl.py \
    --model-path /models/Llama-3-8B \
    --output-dir ./quantized_models/Llama-3-8B_ar \
    --n-calib 128 \
    --adaround-iters 10000 \
    --adaround-lr 1e-3 \
    --layer-batch-size 16 --lmhead-chunks 16 


  python adaround_flip_xl.py \
    --model-path /models/Llama-3-8B \
    --output-dir ./quantized_models/Llama-3-8B_arf \
    --n-calib 128 \
    --adaround-iters 10000 \
    --adaround-lr 1e-3 \
    --layer-batch-size 16 --lmhead-chunks 16


              python compare_awq_slicing.py \
    --heuristic-path ./quantized_models/Llama-3-8B_ar \
    --standard-path ./quantized_models/Llama-3-8B_arf 

              python compare_awq_slicing.py \
    --heuristic-path ./quantized_models/Llama-3-8B_js \
    --standard-path ./quantized_models/Llama-3-8B_js 


python awq_stand_xl.py --model-path /models/Qwen2.5-7B --output-dir ./quantized_models/Qwen2.5-7B_awq_standard

python awq_js_xl.py --model-path ./models/Llama-3-8B \
--output-dir ./quantized_models/Llama-3-8B_js  --max-flip-percent 0.05 --knee-tolerance 0.00 --lmhead-chunks 8

              python compare_awq_slicing.py \
    --heuristic-path ./quantized_models/Llama-3-8B_js
    --standard-path ./modelss/Mistral-7B-v0.3 

          python compare_awq_heuristic.py \
--heuristic-path ./quantized_models/Llama-3-8B_js
--standard-path ./models/Llama-3-8B


              python compare_awq_heuristic.py \
    --heuristic-path ./quantized_models/mistral7b_awq_sh \
    --standard-path ./models/Mistral-7B-v0.3 \
    --n-samples 2000


              python compare_awq_heuristic.py \
    --heuristic-path ./quantized_models/mistral7b_awq_sh \
    --standard-path ./quantized_models/mistral7b_gw_awq_asym_l2 \
    --n-samples 2000


                  python compare_awq_heuristic.py \
    --heuristic-path ./quantized_models/mistral7b_awq_sh \
    --standard-path ./quantized_models/mistral7b_gw_awq_asym_l2

  python adaround_xl.py \
    --model-path ./models/Mistral-7B-v0.3 \
    --n-calib 128 \
    --adaround-iters 10000 \
    --adaround-lr 1e-3 \
    --layer-batch-size 16

Qwen2.5-7B

python awq_stand_xl.py --model-path ./models/Qwen2.5-7B --output-dir ./quantized_models/Qwen2.5-7B_awq_standard


python compare_awq_slicing.py \
    --heuristic-path ./quantized_models/Qwen2.5-7B_awq_standard

python compare_awq_slicing.py \
    --heuristic-path ./models/Qwen2.5-7B

python awq_sh_xl.py --model-path ./models/Qwen2.5-7B --output-dir ./quantized_models/Qwen2.5-7B_awq_sh

python awq_dh_xl.py --model-path ./models/Qwen2.5-7B --output-dir ./quantized_models/Qwen2.5-7B_awq_dh --knee-tolerance 0 --max-flip-percent 0.01


python compare_awq_slicing.py \
    --heuristic-path ./quantized_models/Qwen2.5-7B_awq_dh 

python awq_js_xl.py --model-path ./models/Qwen2.5-7B --output-dir ./quantized_models/Qwen2.5-7B_awq_js  --max-flip-percent 0.05


python compare_awq_slicing.py \
    --heuristic-path ./quantized_models/Qwen2.5-7B_awq_js



python compare_awq_heuristic.py \
    --heuristic-path ./models/Qwen2.5-7B \
    --standard-path ./quantized_models/Qwen2.5-7B_awq_standard \
    --n-samples 2000


python compare_awq_heuristic.py \
    --heuristic-path ./quantized_models/Qwen2.5-7B_awq_sh \
    --standard-path ./quantized_models/Qwen2.5-7B_awq_standardx \
    --n-samples 2000

Llama 8B


python awq_dh_xl.py --model-path ./models/Llama-3-8B --output-dir ./quantized_models/Llama-3-8B_awq_dh --knee-tolerance 0.0 --max-flip-percent 0.01 --lmhead-chunks 6

python compare_awq_slicing.py \
    --heuristic-path ./quantized_models/Llama-3-8B_awq_dh  


python awq_dh_xl.py --model-path ./models/Llama-3-8B --output-dir ./quantized_models/Llama-3-8B_awq_dh --knee-tolerance 0.01  --max-flip-percent 0.01 --lmhead-chunks 6

python compare_awq_slicing.py \
    --heuristic-path ./quantized_models/Llama-3-8B_awq_dh 

                                                             


python awq_js_xl.py --model-path ./models/Llama-3-8B --output-dir ./quantized_models/Llama-3-8B_awq_js --max-flip-percent 0.01 --lmhead-chunks 6

python compare_awq_slicing.py \
    --heuristic-path ./quantized_models/Llama-3-8B_awq_js


python awq_js_xl.py --model-path ./models/Llama-3-8B --output-dir ./quantized_models/Llama-3-8B_awq_js --max-flip-percent 0.05

python compare_awq_slicing.py \
    --heuristic-path ./quantized_models/Llama-3-8B_awq_js



python awq_stand_xl.py --model-path ./models/Llama-3-8B --output-dir ./quantized_models/Llama-3-8B_awq_standard --lmhead-chunks 6

python compare_awq_slicing.py \
    --heuristic-path ./quantized_models/Llama-3-8B_awq_standard


WikiText-2      6.7406          298,902        
C4              10.4777         409,600        



    python awq_sh_xl.py --model-path ./models/Llama-3-8B --output-dir ./quantized_models/Llama-3-8B_awq_sh 

              python compare_awq_heuristic.py \
    --heuristic-path ./quantized_models/Qwen2.5-7B_awq_sh \
    --standard-path ./quantized_models/Qwen2.5-7B_awq_standard \
    --n-samples 2000


              python compare_awq_heuristic.py \
    --heuristic-path ./quantized_models/Llama-3-8B_awq_sh \
    --standard-path ./quantized_models/Llama-3-8B_awq_standard \
    --n-samples 2000


                  python compare_awq_slicing.py \
    --heuristic-path ./quantized_models/Llama-3-8B_awq_sh \
    --standard-path ./quantized_models/Llama-3-8B_awq_standard

python compare_awq_slicing.py \
    --heuristic-path ./models/Llama-3-8B



python compare_awq_heuristic.py \
    --heuristic-path ./models/Llama-3-8B \
    --standard-path ./quantized_models/Llama-3-8B_awq_standard \
    --n-samples 2000

python awq_sh_xl.py --model-path ./models/Llama-3-8B --output-dir ./quantized_models/Llama-3-8B_awq_sh --outlier-percent 0.05
    
python compare_awq_slicing.py \
    --heuristic-path ./quantized_models/Llama-3-8B_awq_sh \
    --standard-path ./quantized_models/Llama-3-8B_awq_standard \
    --n-samples 2000


python awq_dh_xl.py --model-path ./models/Llama-3-8B --output-dir ./quantized_models/Llama-3-8B_awq_dh --knee-tolerance 0.01 --max-flip-percent 0.05 --lmhead-chunks 6

python compare_awq_slicing.py \
    --heuristic-path ./quantized_models/Llama-3-8B_awq_dh 




Llama2-7B


python awq_stand_xl.py --bits 3 --model-path ./models/Llama-2-7b-hf --output-dir ./quantized_models/Llama-2-7b-hf_awq_standard

python awq_js_xl.py --model-path ./models/Llama-2-7b-hf --output-dir ./quantized_models/Llama-2-7b-hf_awq_js --max-flip-percent 0.05

python compare_awq_slicing.py \
    --heuristic-path ./quantized_models/Llama-2-7b-hf_awq_js  \
    --standard-path ./quantized_models/Llama-2-7b-hf_awq_standard

python awq_sh_xl.py --model-path ./models/Llama-2-7b-hf --output-dir ./quantized_models/Llama-2-7b-hf_awq_sh 

python compare_awq_slicing.py \
    --heuristic-path ./quantized_models/Llama-2-7b-hf_awq_sh  \
    --standard-path ./quantized_models/Llama-2-7b-hf_awq_standard \

python awq_js_xl.py --model-path ./models/Llama-2-7b-hf --output-dir ./quantized_models/Llama-2-7b-hf_awq_js --max-flip-percent 0.05

python compare_awq_slicing.py \
    --heuristic-path ./quantized_models/Llama-2-7b-hf_awq_js

python awq_dh_xl.py --model-path ./models/Llama-2-7b-hf --output-dir ./quantized_models/Llama-2-7b-hf_awq_dh 

python compare_awq_slicing.py \
    --heuristic-path ./quantized_models/Llama-2-7b-hf_awq_dh 


python awq_sh_xl.py --model-path ./models/Llama-2-7b-hf --output-dir ./quantized_models/Llama-2-7b-hf_awq_sh --outlier-percent 0.01

python compare_awq_slicing.py \
    --heuristic-path ./quantized_models/Llama-2-7b-hf_awq_sh
(0.0065 outlier-percent best)

python compare_awq_slicing.py \
    --heuristic-path ./quantized_models/Llama-2-7b-hf_awq_sh
python compare_awq_slicing.py \
    --heuristic-path ./models/Llama-2-7b-hf



    Mistra 7B:


python awq_stand_xl.py --model-path ./models/Mistral-7B-v0.3 --output-dir ./quantized_models/Mistral-7B-v0.3_awq_standard
    python awq_sh_xl.py --model-path ./models/Mistral-7B-v0.3 --output-dir ./quantized_models/Mistral-7B-v0.3_awq_sh 

    python compare_awq_heuristic.py \
    --heuristic-path ./quantized_models/Mistral-7B-v0.3_awq_sh \
    --standard-path ./quantized_models/Mistral-7B-v0.3_awq_standard \
    --n-samples 2000


  python compare_awq_slicing.py \
    --heuristic-path ./quantized_models/Mistral-7B-v0.3_awq_sh \
    --standard-path ./quantized_models/Mistral-7B-v0.3_awq_standard



    python compare_awq_slicing.py \
    --heuristic-path ./models/Mistral-7B-v0.3

    
    python compare_awq_heuristic.py \
    --heuristic-path ./models/Mistral-7B-v0.3 \
    --standard-path ./quantized_models/Mistral-7B-v0.3_awq_standardx \
    --n-samples 2000

python awq_stand_xl.py --model-path ./models/Mistral-7B-v0.3 --output-dir ./quantized_models/Mistral-7B-v0.3_awq_standard    

python compare_awq_slicing.py \
    --heuristic-path ./quantized_models/Mistral-7B-v0.3_awq_standard




python awq_dh_xl.py --model-path ./models/Mistral-7B-v0.3 --output-dir ./quantized_models/Mistral-7B-v0.3_awq_dh --knee-tolerance 0.0 --max-flip-percent 0.01

python compare_awq_slicing.py \
    --heuristic-path ./quantized_models/Mistral-7B-v0.3_awq_dh 

python awq_js_xl.py --model-path ./models/Mistral-7B-v0.3 --output-dir ./quantized_models/Mistral-7B-v0.3_awq_js --max-flip-percent 0.05

python compare_awq_slicing.py \
    --heuristic-path ./quantized_models/Mistral-7B-v0.3_awq_js 

python awq_dh_xl.py --model-path ./models/Mistral-7B-v0.3 --output-dir ./quantized_models/Mistral-7B-v0.3_awq_dh --knee-tolerance 0.00

python compare_awq_slicing.py \
    --heuristic-path ./quantized_models/Mistral-7B-v0.3_awq_dh 

