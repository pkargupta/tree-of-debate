export HF_TOKEN="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"
export HF_HOME="/work/nvme/bcaq/shivama2/hf_cache_rlhf/"
export HF_DATASETS_CACHE="/work/nvme/bcaq/shivama2/hf_cache_rlhf/"

export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real #WORKS MUST DO! 

# vllm serve "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"   

# curl -X POST "http://localhost:8000/v1/chat/completions" \
# 	-H "Content-Type: application/json" \
# 	--data '{
# 		"model": "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
# 		"messages": [
# 			{"role": "user", "content": "Hello!"}
# 		]
# 	}' 

python tree_of_debate.py