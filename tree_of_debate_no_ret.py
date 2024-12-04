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
from data_pairer import parse_papers
from create_runs import process
import pandas as pd

extra_log_folder = ""

def collect_all_evidence(node: DebateNode, author_id):
    def gather_all_evidence(node, author_id):
        if len(node.children) == 0:
            try:
                return node.evidence[author_id]
            except:
                return []
        evidence = node.evidence[author_id]
        if evidence is None: evidence = []
        for child in node.children:
            evidence.extend(gather_all_evidence(child, author_id))
    
    all_evidence = gather_all_evidence(node, author_id)
    return list(set(all_evidence))
    
def print_path(node: DebateNode, prefix=""):
    if len(node.children) == 0:
        return prefix + node.round_topic['topic_title']
    
    path = prefix + node.round_topic['topic_title'] + "\n"
    for child in node.children:
        path += print_path(child, prefix + "\t") + "\n"
    
    return path

def get_conversation_of_path(node: DebateNode):
    if node.parent is None:
        return f"Topic: {node.round_topic}. {node.conversation_history}"
    
    return get_conversation_of_path(node.parent) + f"\n\nChild topic: {node.round_topic}. {node.conversation_history}"

def topic_dict_to_str(topic):
    if topic['topic_title'] == topic['topic_description']:
        return topic['topic_title']
    else:
        return f"{topic['topic_title']}: {topic['topic_description']}." # keeping this in case we want to represent topics with the title and description
        # return topic['argument_title']

def run_code(args, topic, f_pap, c_pap):

    focus_paper = PaperAuthor(
        model = model_server,
        paper = Paper(f_pap),
        focus=True,
        id=0,
        log_file=f'{extra_log_folder}/llm_calls.txt',
        is_retrieval=False
    )

    cited_paper = PaperAuthor(
        model = model_server,
        paper = Paper(c_pap),
        focus=False,
        id=1,
        log_file=f'{extra_log_folder}/llm_calls.txt',
        is_retrieval=False
    )

    moderator = Moderator(model_server, args.log_dir)

    paper_authors = [focus_paper, cited_paper]
    leaf_node_label = {'topic_title': topic, 'topic_description': topic}

    if args.log_dir != "":
        with open(os.path.join(extra_log_folder, 'self_deliberation.txt'), 'w') as f:
            f.write(f'Topic: {topic}\n\n')

    # each node has a topic
    root_node = DebateNode(leaf_node_label)
    subtrees = root_node.conduct_self_deliberation(leaf_node_label, paper_authors, moderator, log=args.log_dir) # k new, finer topics to discuss

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
        is_expand = moderator.is_expand(round, conversation)
        if is_expand and depth < max_depth:
            new_subtrees = round.conduct_self_deliberation(round.round_topic, paper_authors, moderator)
            queue_of_rounds.extend(new_subtrees)
            depth += 1

    conversation_history = ''.join(conversation_history)
    with open(f'{extra_log_folder}/conversation_history.txt', 'w+') as f:
        f.write(conversation_history)

    similarities, differences, conversation_paths = [], [], []
    debated_rounds.extend(queue_of_rounds)
    counter = 0
    for round in debated_rounds:
        if len(round.children) > 0:
            similarities.append(topic_dict_to_str(round.round_topic))
        else:
            differences.append(topic_dict_to_str(round.round_topic))
            with open(f'{extra_log_folder}/conversation_path_{counter}.txt', 'w+') as f:
                temp_path = get_conversation_of_path(round) 
                f.write(temp_path)
                conversation_paths.append(temp_path)
            counter += 1

    summary = moderator.summarize_debate(conversation_history, similarities, differences)
    summary_all, similarities_all, differences_all = moderator.summarize_debate_all_paths(conversation_paths)
    summary_sub, similarities_sub, differences_sub = moderator.summarize_debate_sub_paths(conversation_paths)
    with open(f'{args.log_dir}/summary_tod_no_ret.txt', 'w+') as f:
        f.write(summary + "\n")
        f.write(str(similarities) + "\n")
        f.write(str(differences) + "\n")

    with open(f'{args.log_dir}/summary_tod_no_ret_all.txt', 'w+') as f:
        f.write(summary_all + "\n")
        f.write(str(similarities_all) + "\n")
        f.write(str(differences_all) + "\n")

    with open(f'{args.log_dir}/summary_tod_no_ret_sub.txt', 'w+') as f:
        f.write(summary_sub + "\n")
        f.write(str(similarities_sub) + "\n")
        f.write(str(differences_sub) + "\n")

    paths = print_path(root_node)
    with open(f'{args.log_dir}/summary_tod_no_ret.txt', 'a+') as f:
        f.write("\n\n\n")
        f.write("PATHS:\n")
        f.write(paths)
    
    with open(f'{args.log_dir}/evidence_tod_no_ret.txt', 'a+') as f:
        f.write('|'.join(collect_all_evidence(root_node, focus_paper.id)))
        f.write('\n')
        f.write('|'.join(collect_all_evidence(root_node, cited_paper.id)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", default="data.tsv")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--download_dir", default="/")
    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

    # model_server = LLM(model="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",tensor_parallel_size=4,max_num_seqs=100,enable_prefix_caching=True)
    model_server = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct",tensor_parallel_size=2,max_num_seqs=100,enable_prefix_caching=True)

    all_papers = pd.read_csv(args.data_file, sep='\t')

    for _, row in all_papers.iterrows():
        focus_paper = process(row['focus_paper'])
        cited_paper = process(row['opp_paper'])
        args.log_dir = f"{args.log_dir}/{focus_paper}-{cited_paper}"
        if os.path.exists(os.path.join(args.log_dir, "summary_tod_no_ret.txt")):
            exit()
        
        if not os.path.exists(args.log_dir):
            os.mkdir(args.log_dir)

        extra_log_folder = os.path.join(args.log_dir, 'tod_no_ret')
        if not os.path.exists(extra_log_folder):
            os.mkdir(extra_log_folder)
        
        
        parse_papers(focus_paper, cited_paper)

        with open('data.json', 'r') as file:
            data = json.load(file)

        for item in data:
            run_code(args, row['topic'], item['focus'], item['cited'])