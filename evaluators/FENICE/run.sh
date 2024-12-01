export HF_TOKEN="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"
export HF_HOME="/work/nvme/bcaq/shivama2/hf_cache_rlhf/"
export HF_DATASETS_CACHE="/work/nvme/bcaq/shivama2/hf_cache_rlhf/"

export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real #WORKS MUST DO! 

python run.py --results_path /work/nvme/bcaq/shivama2/new_tod/tree-of-debate/baselines/results_split.csv --author_name 'author 0'
python run.py --results_path /work/nvme/bcaq/shivama2/new_tod/tree-of-debate/baselines/results_split.csv --author_name 'author 1'
python run.py --results_path /work/nvme/bcaq/shivama2/new_tod/tree-of-debate/baselines/results_prompt_intro.csv --author_name  'author 0'
python run.py --results_path /work/nvme/bcaq/shivama2/new_tod/tree-of-debate/baselines/results_prompt_intro.csv --author_name  'author 1'

