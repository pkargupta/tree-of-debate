import numpy as np
import argparse

def evaluate_one(summary):
    print(' '.join(summary))

    metrics = {
        "fidelity": ["Does the sentence maintain fidelity to the papers wrt the facts? i.e., are the facts NOT mixed up between papers? 1 for YES, 0 for NO", 0],
        "factual": ["Are the facts present in the sentence accurate? (Enter for no fact) 2 for COMPLETELY ACCURATE, 1, 0 for COMPLETELY INACCURATE", 0],
        # "importance": ["Are the facts present in the sentence important? 1 for YES, 0 for NO", 0]
    }

    for i, sentence in enumerate(summary):
        print(f"Sentence {(i+1)}: {sentence}.\n")
        for metric in metrics:
            score = input(metrics[metric][0] + ": ")
            if score == "" and metric == "factual":
                continue
            metrics[metric][1] += int(score)
        print('\n\n')
    
    final_metrics = []
    for metric in metrics:
        final_metrics.append(metrics[metric][1]) #TODO normalize
    
    granularity_prompt = "Does the summary go deeper than the topic: {topic}?\n1. No, it doesn't take about the topic at all.\n2. No, it talks about stuff more vague than the topic.\n3. No, it talks just about the topic.\n4. Yes, but it does not go deep enough.\n5. Yes, it goes to the correct level.\nAnswer: "
    final_metrics.append(int(input(granularity_prompt)))

    # completion
    num_topics = int(input("How many necessary topics should the summary cover?"))
    topics_cov = int(input("How many necessary topics are covered in the summary?"))
    final_metrics.append(float(topics_cov / num_topics))
    
    return np.array(final_metrics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_one_file", default="logs/summary_one.txt")
    parser.add_argument("--summary_two_file", default="logs/summary_two.txt")
    parser.add_argument("--topic", default="there is no topic")
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