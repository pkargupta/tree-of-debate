export HF_TOKEN="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"
export HF_HOME="/work/nvme/bcaq/shivama2/hf_cache_rlhf/"
export HF_DATASETS_CACHE="/work/nvme/bcaq/shivama2/hf_cache_rlhf/"
# export HF_TOKEN="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"
# export HF_HOME="/work/nvme/bcaq/pkargupta/hf_cache_rlhf/"
# export HF_DATASETS_CACHE="/work/nvme/bcaq/pkargupta/hf_cache_rlhf/"

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
focus_paper="2406_11709" # treeinstruct
cited_paper="2310_10648" # bridge
topic="helping students fix their mistakes"
python tree_of_debate.py --focus_paper $focus_paper --cited_paper $cited_paper --topic "$topic" --no_tree_struct True