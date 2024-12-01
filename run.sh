# export HF_TOKEN="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"
# export HF_HOME="/work/nvme/bcaq/shivama2/hf_cache_rlhf/"
# export HF_DATASETS_CACHE="/work/nvme/bcaq/shivama2/hf_cache_rlhf/"
export HF_TOKEN="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"
export HF_HOME="/work/nvme/bcaq/pkargupta/hf_cache_rlhf/"
export HF_DATASETS_CACHE="/work/nvme/bcaq/pkargupta/hf_cache_rlhf/"

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

# rm -rf logs/*
focus_paper="https://arxiv.org/pdf/2406.11709" # treeinstruct
cited_paper="https://arxiv.org/pdf/2310.10648" # bridge
topic="helping students fix their errors"
python tree_of_debate.py --focus_paper $focus_paper --cited_paper $cited_paper --topic "helping students fix their errors"
# notify "tod"