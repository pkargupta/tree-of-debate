# export HF_TOKEN="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"
# export HF_HOME="/work/nvme/bcaq/shivama2/hf_cache_rlhf/"
# export HF_DATASETS_CACHE="/work/nvme/bcaq/shivama2/hf_cache_rlhf/"

# export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real #WORKS MUST DO! 

# CUDA_VISIBLE_DEVICES=0,1 python run_model.py --dataset_path baseline_data.tsv --base_llm meta-llama/Meta-Llama-3.1-8B-Instruct --baseline_type split
CUDA_VISIBLE_DEVICES=0,1 python run_model.py --dataset_path baseline_data.tsv --base_llm meta-llama/Meta-Llama-3.1-8B-Instruct --baseline_type prompt_intro

