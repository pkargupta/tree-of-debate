export HF_TOKEN="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"
export HF_HOME="/work/nvme/bcaq/shivama2/hf_cache_rlhf/"
export HF_DATASETS_CACHE="/work/nvme/bcaq/shivama2/hf_cache_rlhf/"

export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real #WORKS MUST DO! 

python run_model.py --dataset_path opp_pap_data.json --base_llm nvidia/Llama-3.1-Nemotron-70B-Instruct-HF --baseline_type split
python run_model.py --dataset_path opp_pap_data.json --base_llm nvidia/Llama-3.1-Nemotron-70B-Instruct-HF --baseline_type prompt_intro

