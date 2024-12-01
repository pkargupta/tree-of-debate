# export HF_TOKEN="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"
# export HF_HOME="/work/nvme/bcaq/shivama2/hf_cache_rlhf/"
# export HF_DATASETS_CACHE="/work/nvme/bcaq/shivama2/hf_cache_rlhf/"
# export HF_TOKEN="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"
# export HF_HOME="/work/nvme/bcaq/pkargupta/hf_cache_rlhf/"
# export HF_DATASETS_CACHE="/work/nvme/bcaq/pkargupta/hf_cache_rlhf/"

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

# rm -rf logs/*
# focus_paper="https://arxiv.org/pdf/2406.11709" # treeinstruct
# cited_paper="https://arxiv.org/pdf/2310.10648" # bridge
# CUDA_VISIBLE_DEVICES=2,3 python tree_of_debate.py --focus_paper $focus_paper --cited_paper $cited_paper --topic "helping students fix their errors"
# notify "tod"


focus_paper="https://arxiv.org/pdf/2406.11709"
cited_paper="https://arxiv.org/pdf/2310.10648"
log_dir="logs/Instruct_Bridging/"
CUDA_VISIBLE_DEVICES=2,3 python tree_of_debate.py --focus_paper $focus_paper --cited_paper $cited_paper --log_dir $log_dir --topic "helping students fix their errors"

focus_paper="https://arxiv.org/pdf/2305.10601"
cited_paper="https://arxiv.org/pdf/2201.11903"
log_dir="logs/Tree_ChainofThought/"
CUDA_VISIBLE_DEVICES=2,3 python tree_of_debate.py --focus_paper $focus_paper --cited_paper $cited_paper --log_dir $log_dir --topic "enabling large language model reasoning via prompting"

focus_paper="https://arxiv.org/pdf/2404.02078"
cited_paper="https://arxiv.org/pdf/2406.09136"
log_dir="logs/Advancing_Chain/"
CUDA_VISIBLE_DEVICES=2,3 python tree_of_debate.py --focus_paper $focus_paper --cited_paper $cited_paper --log_dir $log_dir --topic "using preferences to train language models for better reasoning"

focus_paper="https://arxiv.org/pdf/2411.04425"
cited_paper="https://arxiv.org/pdf/2402.04333v3"
log_dir="logs/DELIFT_LESS/"
CUDA_VISIBLE_DEVICES=2,3 python tree_of_debate.py --focus_paper $focus_paper --cited_paper $cited_paper --log_dir $log_dir --topic "selecting subsets of data to improve language model performance"

focus_paper="https://arxiv.org/pdf/2402.04333v3"
cited_paper="https://arxiv.org/pdf/1906.01827"
log_dir="logs/LESS_Coresets/"
CUDA_VISIBLE_DEVICES=2,3 python tree_of_debate.py --focus_paper $focus_paper --cited_paper $cited_paper --log_dir $log_dir --topic "using gradient-based information to select subsets of data for improving language model performance"

focus_paper="https://arxiv.org/pdf/2402.04333v3"
cited_paper="https://arxiv.org/pdf/2301.13287"
log_dir="logs/LESS_MILO/"
CUDA_VISIBLE_DEVICES=2,3 python tree_of_debate.py --focus_paper $focus_paper --cited_paper $cited_paper --log_dir $log_dir --topic "selecting subsets of data to improve language model performance"

focus_paper="https://arxiv.org/pdf/2404.12522"
cited_paper="https://arxiv.org/pdf/2210.00423"
log_dir="logs/Neural_Improved/"
CUDA_VISIBLE_DEVICES=2,3 python tree_of_debate.py --focus_paper $focus_paper --cited_paper $cited_paper --log_dir $log_dir --topic "using active learning to train neural network-based multi-armed bandits for k-class classification"

focus_paper="https://arxiv.org/pdf/2404.12522"
cited_paper="https://arxiv.org/pdf/2110.08611"
log_dir="logs/Neural_Deep/"
CUDA_VISIBLE_DEVICES=2,3 python tree_of_debate.py --focus_paper $focus_paper --cited_paper $cited_paper --log_dir $log_dir --topic "using active learning to train neural network-based multi-armed bandits for k-class classification"

notify "tod"
