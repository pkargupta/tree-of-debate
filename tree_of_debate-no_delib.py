import pickle
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
from no_delib.debate import DebateNode
from no_delib.paper_details import Paper
from no_delib.persona import PaperAuthor
from no_delib.moderator import Moderator
import argparse
from typing import List
from vllm import LLM
import os
import json
from no_delib.data_pairer import parse_papers

def print_path(node: DebateNode, prefix=""):
    if len(node.children) == 0:
        # return prefix + node.round_topic['topic_title']
        out = f"""{prefix}{node.round_topic['topic_title']}: {node.round_topic['topic_description']}
{prefix}Author 0's Argument: {node.final_arguments[0]['revised_argument_title']}. {node.final_arguments[0]['revised_argument_description']}
{prefix}Author 1's Argument: {node.final_arguments[1]['revised_argument_title']}. {node.final_arguments[1]['revised_argument_description']}

"""
        return out
    elif node.parent is None:
        path = prefix + node.round_topic['topic_title'] + "\n\n"
    else:
        path = f"""{prefix}{node.round_topic['topic_title']}: {node.round_topic['topic_description']}
{prefix}Author 0's Argument: {node.final_arguments[0]['revised_argument_title']}. {node.final_arguments[0]['revised_argument_description']}
{prefix}Author 1's Argument: {node.final_arguments[1]['revised_argument_title']}. {node.final_arguments[1]['revised_argument_description']}

"""
    for child in node.children:
        path += print_path(child, prefix + "\t") + "\n"
    
    return path

def topic_dict_to_str(topic):
    if topic['topic_title'] == topic['topic_description']:
        return topic['topic_title']
    else:
        return f"{topic['topic_title']}: {topic['topic_description']}." # keeping this in case we want to represent topics with the title and description
        # return topic['argument_title']

def collect_evidence(evidence, subtrees):
    for c in subtrees:
        for author_id, e in c.evidence.items():
            evidence[author_id].extend(e)
    return evidence
    

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
    leaf_node_label = {'topic_title': args.topic, 'topic_description': args.topic}

    if args.log_dir != "":
        with open(os.path.join(args.log_dir, 'self_deliberation.txt'), 'w') as f:
            f.write(f'Topic: {args.topic}\n\n')

    # each node has a topic
    root_node = DebateNode(leaf_node_label)
    all_evidence = {p.id:[] for p in paper_authors}
    
    subtrees = root_node.conduct_self_deliberation(leaf_node_label, paper_authors, moderator, log=args.log_dir) # k new, finer topics to discuss
    all_evidence = collect_evidence(all_evidence, subtrees)

    conversation_history = []

    queue_of_rounds: List[DebateNode] = []
    queue_of_rounds.extend(subtrees)

    debated_rounds = [root_node]

    depth = 0
    max_depth = 3

    while len(queue_of_rounds) > 0:
        round = queue_of_rounds.pop(0)
        debated_rounds.append(round)
        conversation = round.conduct_debate([focus_paper, cited_paper])
        conversation_history.append(conversation)
        if moderator.is_expand(round, conversation) and depth < max_depth:
            new_subtrees = round.conduct_self_deliberation(round.round_topic, paper_authors, moderator)
            all_evidence = collect_evidence(all_evidence, new_subtrees)
            queue_of_rounds.extend(new_subtrees)
            depth += 1

    conversation_history = ''.join(conversation_history)
    with open(f'{args.log_dir}/conversation_history.txt', 'w') as f:
        f.write(conversation_history)

    with open(f'{args.log_dir}/evidence.txt', 'w') as f:
        for author_id, e in all_evidence.items():
            unique_e = list(set(e))
            f.write(str(unique_e))
            f.write('\n')

    paths = print_path(root_node)
    with open(f'{args.log_dir}/path.txt', 'w') as f:
        f.write("\n\n\n\n\n")
        f.write("PATHS:\n")
        f.write(paths)

    path_summary = moderator.summarize_path_debate(paper_authors, leaf_node_label, paths)
    with open(f'{args.log_dir}/path_summary.txt', 'w') as f:
        f.write(path_summary)

    # with open('temp.pkl', 'wb+') as f:
    #     pickle.dump([queue_of_rounds, debated_rounds, conversation_history, root_node, similarities, differences], f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--focus_paper", default="2406_11709")
    parser.add_argument("--cited_paper", default="2310_10648")
    parser.add_argument("--topic", default="helping students fix their mistakes")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--download_dir", default="/")
    args = parser.parse_args()

    args.log_dir = f"{args.log_dir}/no_delib/{args.focus_paper.split('.json')[0]}-{args.cited_paper.split('.json')[0]}"
    
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    # if not os.path.exists("data.json"):
    
    parse_papers(args.focus_paper, args.cited_paper)
    
    with open('data.json', 'r') as file:
        data = json.load(file)

    model_server = LLM(model="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",tensor_parallel_size=4,max_num_seqs=256,enable_prefix_caching=True)
    # model_server = LLM(model="meta-llama/Meta-Llama-3.1-8B-Instruct",tensor_parallel_size=2,max_num_seqs=100,enable_prefix_caching=True)

    for item in data:
        run_code(args, item['focus'], item['cited'])

