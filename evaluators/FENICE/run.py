from metric.FENICE import FENICE
import pandas as pd
import argparse

fenice = FENICE()
def fact_check(document, summary):
    batch = [
        {"document": document, "summary": summary}
    ]
    results = fenice.score_batch(batch)
    return results[0]['score']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", default="/work/nvme/bcaq/shivama2/new_tod/tree-of-debate/baselines/results_prompt_intro.csv")
    args = parser.parse_args()

    scores = []
    data = pd.read_csv(args.results_path)
    for i,row in data.iterrows():
        summary = row['summary'] 
        document = row['document_f'] + "\n" + row['document_o']
        scores.append(fact_check(document,summary))
    print(f'scores={sum(scores) / len(scores)}')



