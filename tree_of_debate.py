import pickle
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
from debate import DebateNode
from paper_details import Paper
from persona import PaperAuthor
from moderator import Moderator
import argparse
from typing import List
from vllm import LLM
import os
import json
from data_pairer import parse_papers, parse_papers_docling

def print_path(node: DebateNode, prefix=""):
    if len(node.children) == 0:
        return prefix + node.round_topic['argument_title']
    
    path = prefix + node.round_topic['argument_title'] + "\n"
    for child in node.children:
        path += print_path(child, prefix + "\t") + "\n"
    
    return path

def topic_dict_to_str(topic):
    return f"{topic['argument_title']}: {topic['description']}." # keeping this in case we want to represent topics with the title and description
    # return topic['argument_title']

def run_code(args, f_pap, c_pap):

    focus_paper = PaperAuthor(
        model = model_server,
        paper = Paper(f_pap),
        focus=True,
        id=0,
        log_dir=args.log_dir
    )

    cited_paper = PaperAuthor(
        model = model_server,
        paper = Paper(c_pap),
        focus=False,
        id=1,
        log_dir=args.log_dir
    )

    moderator = Moderator(model_server, args.log_dir)

    paper_authors = [focus_paper, cited_paper]
    leaf_node_label = {'argument_title': args.topic, 'description': args.topic}

    if args.log_dir != "":
        with open(os.path.join(args.log_dir, 'self_deliberation.txt'), 'w') as f:
            f.write(f'Topic: {args.topic}\n\n')

    # each node has a topic
    root_node = DebateNode(leaf_node_label)
    subtrees = root_node.conduct_self_deliberation(leaf_node_label, paper_authors, log=args.log_dir) # k new, finer topics to discuss

    conversation_history = []

    queue_of_rounds: List[DebateNode] = []
    queue_of_rounds.extend(subtrees)

    debated_rounds = [root_node]

    depth = 0
    max_depth = 2

    while len(queue_of_rounds) > 0:
        round = queue_of_rounds.pop(0)
        debated_rounds.append(round)
        conversation = round.conduct_debate([focus_paper, cited_paper])
        conversation_history.append(conversation)
        if moderator.is_expand(round, conversation) and depth < max_depth:
            new_subtrees = round.conduct_self_deliberation(round.round_topic, paper_authors)
            queue_of_rounds.extend(new_subtrees)
            depth += 1

    conversation_history = ''.join(conversation_history)
    with open(f'{args.log_dir}/conversation_history.txt', 'w+') as f:
        f.write(conversation_history)
            
    # with open('temp.pkl', 'rb') as f:
    #     queue_of_rounds, debated_rounds, conversation_history, root_node = pickle.load(f)
    
    similarities, differences = [], []
    debated_rounds.extend(queue_of_rounds)
    for round in debated_rounds:
        if len(round.children) > 0:
            similarities.append(topic_dict_to_str(round.round_topic))
        else:
            differences.append(topic_dict_to_str(round.round_topic))


    summary = moderator.summarize_debate(conversation_history, similarities, differences)
    with open(f'{args.log_dir}/summary.txt', 'w+') as f:
        f.write(summary)

    paths = print_path(root_node)
    with open(f'{args.log_dir}/summary.txt', 'a+') as f:
        f.write("\n\n\n\n\n")
        f.write("PATHS:\n")
        f.write(paths)

    with open('temp.pkl', 'wb+') as f:
        pickle.dump([queue_of_rounds, debated_rounds, conversation_history, root_node, similarities, differences], f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--focus_paper", default="https://arxiv.org/pdf/1706.03762")
    parser.add_argument("--cited_paper", default="https://arxiv.org/pdf/1810.04805")
    parser.add_argument("--topic", default="language model architectures")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--download_dir", default="/")
    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # # if not os.path.exists("data.json"):
    # parse_papers(args.focus_paper, args.cited_paper)
    # with open('data.json', 'r') as file:
    #     data = json.load(file)

    # model_server = LLM(model="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",tensor_parallel_size=4,max_num_seqs=100,enable_prefix_caching=True)
    # # model_server = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct",tensor_parallel_size=2,max_num_seqs=100,enable_prefix_caching=True)

    # for item in data:
    #     run_code(args, item['focus'], item['cited'])

