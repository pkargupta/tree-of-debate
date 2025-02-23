import os
from unidecode import unidecode
from vllm import LLM
import argparse

from data_pairer import parse_papers_url
from tree_of_debate import run_code
from baselines.run_baseline import run_baseline_code
from tree_of_debate_no_delib import run_no_delib_code
from tree_of_debate_no_tree import run_no_tree_code

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", default="dataset/tree_of_debate_dataset.tsv")
    parser.add_argument("--log", default="dataset/")
    parser.add_argument("--experiment", default="tod") # options: tod, single, two, no-tree, no-retrieval
    

    args = parser.parse_args()
    
    model_server = LLM(model="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",tensor_parallel_size=4,max_num_seqs=256,enable_prefix_caching=True)

    with open(args.csv_file, 'r', encoding="utf-8") as f:
        rows = f.readlines()

    for row in rows:
        cols = row.split('\t')
        contributor, topic, paper_0_url, paper_0_title, paper_0_abstract, paper_0_intro, paper_1_url, paper_1_title, paper_1_abstract, paper_1_intro, method, cite = cols

        args.topic = topic
        print(f"######## NOW PROCESSING {unidecode(paper_0_title)} VS. {unidecode(paper_1_title)}")
        log_id = paper_0_title.lower().replace(' ', '_')[:15] + "_vs_" + paper_1_title.lower().replace(' ', '_')[:15]
        args.log_dir = f"{args.log}/{log_id}/{args.experiment}"
        
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
    
        paper_0_text, paper_1_text = parse_papers_url(paper_0_url, paper_1_url)
        paper_0 = {'title':unidecode(paper_0_title), 'abstract':unidecode(paper_0_abstract), 'introduction':unidecode(paper_0_intro), 'full_text':paper_0_text}
        paper_1 = {'title':unidecode(paper_1_title), 'abstract':unidecode(paper_1_abstract), 'introduction':unidecode(paper_1_intro), 'full_text':paper_1_text}

        with open(os.path.join(args.log_dir, f"{paper_0_title.lower().replace(' ', '_')[:15]}_text.txt"), "w", encoding='utf-8') as f:
            f.write(paper_0["full_text"])

        with open(os.path.join(args.log_dir, f"{paper_1_title.lower().replace(' ', '_')[:15]}_text.txt"), "w", encoding='utf-8') as f:
            f.write(paper_1["full_text"])

        if args.experiment == 'tod':
            run_code(args, paper_0, paper_1, model_server)
        elif (args.experiment == 'single') or (args.experiment == 'two'):
            run_baseline_code(args, paper_0, paper_1, model_server)
        elif args.experiment == 'no-retrieval':
            run_no_delib_code(args, paper_0, paper_1, model_server)
        elif args.experiment == 'no-tree':
            run_no_tree_code(args, paper_0, paper_1, model_server)
        else:
            print("experiment is not supported!")