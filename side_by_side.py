import numpy as np
import argparse

def evaluate_one(summary):
    metrics = {
        "fidelity": ["Does the sentence maintain fidelity to the papers wrt the facts? i.e., are the facts NOT mixed up between papers? 1 for YES, 0 for NO", 0],
        "factual": ["Are the facts present in the sentence accurate? 2 for COMPLETELY ACCURATE, 1, 0 for COMPLETELY INACCURATE", 0],
        "importance": ["Are the facts present in the sentence important? 1 for YES, 0 for NO", 0]
    }

    for i, sentence in enumerate(summary):
        print(f"Sentence {(i+1)}: {sentence}.\n")
        for metric in metrics:
            score = int(input(metrics[metric][0] + ": "))
            metrics[metric][1] += score
        print('\n\n')
    
    final_metrics = []
    for metric in metrics:
        final_metrics.append(metrics[metric][1])
    
    return np.array(final_metrics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_one_file", default="logs/summary_one.txt")
    parser.add_argument("--summary_two_file", default="logs/summary_two.txt")
    args = parser.parse_args()

    with open(args.summary_one_file, 'r') as f:
        summary_one = f.read()
    
    with open(args.summary_two_file, 'r') as f:
        summary_two = f.read()
    
    summary_one = [sent for sent in summary_one.split('.') if len(sent) > 0]
    summary_two = [sent for sent in summary_two.split('.') if len(sent) > 0]

    single_eval_one = evaluate_one(summary_one)
    single_eval_two = evaluate_one(summary_two)

    print(single_eval_one - single_eval_two)