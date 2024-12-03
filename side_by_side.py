import numpy as np
import argparse
import glob
import os

def evaluate_one(topic, summary, similarties, differences):
    print(' '.join(summary))

    metric_logs = []

    metrics = {
        "fidelity": ["Does the sentence maintain fidelity to the papers wrt the facts? i.e., are the facts NOT mixed up between papers? 1 for YES, 0 for NO", 0],
        # "factual": ["Are the facts present in the sentence accurate? (Enter for no fact) 2 for COMPLETELY ACCURATE, 1, 0 for COMPLETELY INACCURATE", 0],
        # "importance": ["Are the facts present in the sentence important? 1 for YES, 0 for NO", 0]
    }

    for i, sentence in enumerate(summary):
        print(f"Sentence {(i+1)}: {sentence}.\n")
        row = [i]
        for metric in metrics:
            score = input(metrics[metric][0] + ": ")
            if score == "" and metric == "factual":
                continue
            metrics[metric][1] += int(score)
            row.append(int(score))
        print('\n\n')
        metric_logs.append(row)
    
    final_metrics = {}
    for metric in metrics:
        final_metrics[metric] = metrics[metric][1] / len(summary)
    
    granularity_prompt = f"Does the summary go deeper than the topic: {topic}?\n1. No, it doesn't take about the topic at all.\n2. No, it talks about stuff more vague than the topic.\n3. No, it talks just about the topic.\n4. Yes, but it does not go deep enough.\n5. Yes, it goes to the correct level.\nAnswer: "
    final_metrics['granularity'] = int(input(granularity_prompt))

    # completion
    completion_prompt = "Does the summary seem comprehensive and complete?\n1. No, the summary misses (MULTIPLE) major points.\n2.No, the summary misses a (SINGULAR) major point.\n3. Somewhat, the summary misses minor points.\n4. Yes, the summary covers the major points.\nAnswer: "
    final_metrics['completeness'] = int(input(completion_prompt))

    # sim and diffs
    similarities_prompt = f"Here are the similarities:\n{similarties}\n\nAre the similarities comprehensive?\n1. No, multiple similarities are problematic.\2. Somewhat, a singular similarity is problematic.\n3. Yes, all similarities are good.\nAnswer: "
    final_metrics['similarity_comprehension'] = int(input(similarities_prompt))

    differences_prompt = f"Here are the differences:\n{differences}\n\nAre the differences comprehensive?\n1. No, multiple differences are problematic.\2. Somewhat, a singular difference is problematic.\n3. Yes, all differences are good.\nAnswer: "
    final_metrics['difference_comprehension'] = int(input(differences_prompt))
    
    return final_metrics, metric_logs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--focus_paper", default="2406_11709")
    parser.add_argument("--cited_paper", default="2310_10648")
    parser.add_argument("--topic", default="helping students fix their mistakes")
    args = parser.parse_args()

    if os.path.exists('evaluation/'):
        os.mkdir('evaluation/')

    summary_files = glob.glob(f'logs/{args.focus_paper}-{args.cited_paper}/summary*.txt')

    for summary_file in summary_files:
        if os.path.exists(f'evaluation/{args.focus_paper}-{args.cited_paper}/'):
            continue

        os.mkdir(f'evaluation/{args.focus_paper}-{args.cited_paper}/')

        shorthand = summary_file.split('/')[-1][8:-4]

        with open(summary_file, 'r') as f:
            text = f.readline()
            similarities = f.readline()
            differences = f.readline()            
        
        text = [sent for sent in text.split('.') if len(sent) > 0]
        final_metrics, metric_logs = evaluate_one(args.topic, text, similarities, differences)


        with open(f'evaluation/{args.focus_paper}-{args.cited_paper}/summary_{shorthand}.txt', 'w+') as f:
            for metric in final_metrics.keys():
                f.write(f'{metric},{final_metrics[metric]}\n')
            for i, row in enumerate(metric_logs):
                f.write(f'{i},{str(row)[1:-1]}\n')