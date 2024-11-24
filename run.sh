export HF_TOKEN="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"
# export HF_HOME="/shared/data/shivama2/hf_cache/"
# export HF_DATASETS_CACHE="/shared/data/shivama2/hf_cache/"

# export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real #WORKS MUST DO! 

# vllm serve "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"   

# curl -X POST "http://localhost:8000/v1/chat/completions" \
# 	-H "Content-Type: application/json" \
# 	--data '{
# 		"model": "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
# 		"messages": [
# 			{"role": "user", "content": "Hello!"}
# 		]
# 	}' 

rm -rf logs/*
CUDA_VISIBLE_DEVICES=1,4 python tree_of_debate.py --focus_paper "https://arxiv.org/pdf/1706.03762" --cited_paper "https://arxiv.org/pdf/1810.04805" --topic "language model architectures"
notify "tod"