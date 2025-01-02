export HF_ALLOW_CODE_EVAL=1

CUDA_VISIBLE_DEVICES="0" python main.py --data humaneval-prompt --data_url data/codes --model /data2/zhouyz/hf/Qwen/Qwen2.5-Coder-1.5B-Instruct --model_args device_map=cuda,trust_remote_code=True,torch_dtype=auto,_attn_implementation=flash_attention_2 --infer_args do_sample=True,temperature=0.2,top_p=0.95,top_k=0,max_length=1024 --infer_type coder --tokenizer_args trust_remote_code=True,num_crops=4 --time_str 03_26_00_00_00 --filter_type direct
