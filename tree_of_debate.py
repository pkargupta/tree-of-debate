import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
from debate import DebateNode
from paper_details import Paper
from persona import PaperAuthor
from moderator import Moderator
import argparse
from typing import List
from vllm import LLM
import os
import json
from data_pairer import parse_papers
import numpy as np

def run_code(args, f_pap, c_pap):

    focus_paper = PaperAuthor(
    model = model_server,
    paper = Paper(f_pap),
    focus=True,
    id=0
    )

    cited_paper = PaperAuthor(
        model = model_server,
        paper = Paper(c_pap),
        focus=False,
        id=1
    )

    moderator = Moderator(model_server)

    paper_authors = [focus_paper, cited_paper]
    leaf_node_label = args.topic

    if args.log_dir != "":
        with open(os.path.join(args.log_dir, 'self_deliberation.txt'), 'w') as f:
            f.write(f'Topic: {args.topic}\n\n')

    # each node has a topic
    root_node = DebateNode(leaf_node_label)
    subtrees = root_node.conduct_self_deliberation(leaf_node_label, paper_authors, log=args.log_dir) # k new, finer topics to discuss

    """
    TI and knowledge tracing
    args: (1) tree and (2) sse

    root: educational convos
        root_args: tree + sse

    level 1: tree || sse (2 nodes)
        tree_args: tree is better than not
        sse_args: sse is good

    level 2: tree_is_good | tree_is_bad || sse_with_nl | sse_with_vector

    """

    conversation_history = []

    queue_of_rounds: List[DebateNode] = []
    queue_of_rounds.extend(subtrees)

    while len(queue_of_rounds) > 0:
        round = queue_of_rounds.pop(0)
        conversation, new_focus_arg, new_cited_arg = round.conduct_debate(focus_paper, cited_paper)
        conversation_history.extend(conversation)
        if moderator.is_expand(round.arguments, [new_focus_arg, new_cited_arg]):
            new_subtrees = round.conduct_self_deliberation(round.round_topic, paper_authors)
            queue_of_rounds.extend(new_subtrees)

    with open('conversation_history.txt', 'w+') as f:
        f.write('\n'.join(conversation_history))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--focus_paper", default="https://arxiv.org/pdf/1706.03762")
    parser.add_argument("--cited_paper", default="https://arxiv.org/pdf/1810.04805")
    parser.add_argument("--topic", default="language model architectures")
    parser.add_argument("--log_dir", default="logs")
    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # if not os.path.exists("data.json"):
    parse_papers(args.focus_paper, args.cited_paper)
    with open('data.json', 'r') as file:
        data = json.load(file)

    model_server = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct",tensor_parallel_size=2,gpu_memory_utilization=0.5,max_num_seqs=100) #,enable_prefix_caching=True)

    for item in data:
        run_code(args, item['focus'], item['cited'])

