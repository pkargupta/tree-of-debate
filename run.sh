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
# # notify "tod"

focus_paper="https://arxiv.org/pdf/2406.11709"
cited_paper="https://arxiv.org/pdf/2310.10648"
log_dir="logs/2406_11709-2310_10648/"
CUDA_VISIBLE_DEVICES=2,3 python tree_of_debate.py --focus_paper $focus_paper --cited_paper $cited_paper --log_dir $log_dir --topic "helping students fix their mistakes"

focus_paper="https://arxiv.org/pdf/2305.10601"
cited_paper="https://arxiv.org/pdf/2201.11903"
log_dir="logs/2305_10601-2201_11903/"
CUDA_VISIBLE_DEVICES=2,3 python tree_of_debate.py --focus_paper $focus_paper --cited_paper $cited_paper --log_dir $log_dir --topic "enabling large language model reasoning via prompting"

focus_paper="https://arxiv.org/pdf/2404.02078"
cited_paper="https://arxiv.org/pdf/2406.09136"
log_dir="logs/2404_02078-2406_09136/"
CUDA_VISIBLE_DEVICES=2,3 python tree_of_debate.py --focus_paper $focus_paper --cited_paper $cited_paper --log_dir $log_dir --topic "using preferences to train language models for better reasoning"

focus_paper="https://arxiv.org/pdf/2411.04425"
cited_paper="https://arxiv.org/pdf/2402.04333v3"
log_dir="logs/2411_04425-2402_04333v3/"
CUDA_VISIBLE_DEVICES=2,3 python tree_of_debate.py --focus_paper $focus_paper --cited_paper $cited_paper --log_dir $log_dir --topic "selecting subsets of data to improve language model performance"

focus_paper="https://arxiv.org/pdf/2402.04333v3"
cited_paper="https://arxiv.org/pdf/1906.01827"
log_dir="logs/2402_04333v3-1906_01827/"
CUDA_VISIBLE_DEVICES=2,3 python tree_of_debate.py --focus_paper $focus_paper --cited_paper $cited_paper --log_dir $log_dir --topic "using gradient-based information to select subsets of data for improving language model performance"

focus_paper="https://arxiv.org/pdf/2402.04333v3"
cited_paper="https://arxiv.org/pdf/2301.13287"
log_dir="logs/2402_04333v3-2301_13287/"
CUDA_VISIBLE_DEVICES=2,3 python tree_of_debate.py --focus_paper $focus_paper --cited_paper $cited_paper --log_dir $log_dir --topic "selecting subsets of data to improve language model performance"

focus_paper="https://arxiv.org/pdf/2404.12522"
cited_paper="https://arxiv.org/pdf/2210.00423"
log_dir="logs/2404_12522-2210_00423/"
CUDA_VISIBLE_DEVICES=2,3 python tree_of_debate.py --focus_paper $focus_paper --cited_paper $cited_paper --log_dir $log_dir --topic "using active learning to train neural network-based multi-armed bandits for k-class classification"

focus_paper="https://arxiv.org/pdf/2404.12522"
cited_paper="https://arxiv.org/pdf/2110.08611"
log_dir="logs/2404_12522-2110_08611/"
CUDA_VISIBLE_DEVICES=2,3 python tree_of_debate.py --focus_paper $focus_paper --cited_paper $cited_paper --log_dir $log_dir --topic "using active learning to train neural network-based multi-armed bandits for k-class classification"

focus_paper="https://arxiv.org/pdf/2408.04873"
cited_paper="https://arxiv.org/pdf/2104.05919"
log_dir="logs/2408_04873-2104_05919/"
CUDA_VISIBLE_DEVICES=2,3 python tree_of_debate.py --focus_paper $focus_paper --cited_paper $cited_paper --log_dir $log_dir --topic "event analysis"

focus_paper="https://arxiv.org/pdf/2408.04873"
cited_paper="https://arxiv.org/pdf/2206.04153"
log_dir="logs/2408_04873-2206_04153/"
CUDA_VISIBLE_DEVICES=2,3 python tree_of_debate.py --focus_paper $focus_paper --cited_paper $cited_paper --log_dir $log_dir --topic "event granularities"

focus_paper="https://arxiv.org/pdf/2201.06771"
cited_paper="https://arxiv.org/pdf/2001.09522"
log_dir="logs/2201_06771-2001_09522/"
CUDA_VISIBLE_DEVICES=2,3 python tree_of_debate.py --focus_paper $focus_paper --cited_paper $cited_paper --log_dir $log_dir --topic "Taxonomy completion versus taxonomy expansion in weakly supervised settings"

focus_paper="https://arxiv.org/pdf/2004.12832"
cited_paper="https://arxiv.org/pdf/1810.04805"
log_dir="logs/2004_12832-1810_04805/"
CUDA_VISIBLE_DEVICES=2,3 python tree_of_debate.py --focus_paper $focus_paper --cited_paper $cited_paper --log_dir $log_dir --topic "Continual pretraining of Bert for retrieval tasks"

focus_paper="https://arxiv.org/pdf/2103.00113"
cited_paper="https://arxiv.org/pdf/2310.14525"
log_dir="logs/2103_00113-2310_14525/"
CUDA_VISIBLE_DEVICES=2,3 python tree_of_debate.py --focus_paper $focus_paper --cited_paper $cited_paper --log_dir $log_dir --topic "contrastive learning on graphs"

focus_paper="https://arxiv.org/pdf/2010.03768"
cited_paper="https://arxiv.org/pdf/2010.03768"
log_dir="logs/2010_03768-2010_03768/"
CUDA_VISIBLE_DEVICES=2,3 python tree_of_debate.py --focus_paper $focus_paper --cited_paper $cited_paper --log_dir $log_dir --topic "multi-model embodied agents + environments"

cd baselines
source run.sh
notify "tod"
