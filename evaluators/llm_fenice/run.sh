export HF_TOKEN="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"
export HF_HOME="/work/nvme/bcaq/shivama2/hf_cache_rlhf/"
export HF_DATASETS_CACHE="/work/nvme/bcaq/shivama2/hf_cache_rlhf/"

export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real #WORKS MUST DO! 

python run_llm.py --results_path /work/nvme/bcaq/shivama2/new_tod/tree-of-debate/baselines/opp_pap_data_split.json --model_type split_posthoc
python run_llm.py --results_path /work/nvme/bcaq/shivama2/new_tod/tree-of-debate/baselines/opp_pap_data_prompt_intro.json --model_type prompt_intro_abs


