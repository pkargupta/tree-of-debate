import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import json  # To save opp_pap to disk
from utils import process_arxiv, extract_sections_from_markdown

def process_row(row):
    """
    Process a single row and return the processed data as a dictionary.
    """
    sample = row.to_dict()
    f_pap = process_arxiv(sample['focus_paper'])
    opp_pap = process_arxiv(sample['opp_paper'])
    f_tit = sample['title_focus']
    opp_tit = sample['title_opp']
    topic = sample['topic']
    
    f_abs = extract_sections_from_markdown(f_pap, 'Abstract')
    f_intro = extract_sections_from_markdown(f_pap, 'Introduction')
    o_abs = extract_sections_from_markdown(opp_pap, 'Abstract')
    o_intro = extract_sections_from_markdown(opp_pap, 'Introduction')
    
    return {
        'focus_paper': sample['focus_paper'],
        'opp_paper': sample['opp_paper'],
        'title_focus': f_tit,
        'title_opp': opp_tit,
        'topic': topic,
        'full_focus_paper': f_pap,
        'full_opp_paper': opp_pap,
        'f_abstract': f_abs,
        'f_intro': f_intro,
        'o_abstract': o_abs,
        'o_intro': o_intro,
    }

def main():
    dataset_path = '/work/nvme/bcaq/shivama2/new_tod/tree-of-debate/baselines/data2.csv'
    data = pd.read_csv(dataset_path)
    
    processed_data = []
    
    # Use ProcessPoolExecutor to parallelize processing with tqdm for progress tracking
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_row, [row for _, row in data.iterrows()]), total=len(data)))
        processed_data.extend(results)
    
    # Write the processed data to disk
    output_path = "opp_pap_data.json"
    with open(output_path, 'w') as f:
        json.dump(processed_data, f, indent=4)
    print(f"Data written to {output_path}")

if __name__ == "__main__":
    main()


# import pandas as pd
# import argparse
# from utils import process_arxiv,extract_sections_from_markdown


# # def process():

# op_data = []
# dataset_path = '/work/nvme/bcaq/shivama2/new_tod/tree-of-debate/baselines/data2.csv'
# data = pd.read_csv(dataset_path)

# for i,row in data.iterrows():
#     sample = row.to_dict()
#     f_pap, opp_pap, f_tit, opp_tit, topic = process_arxiv(sample['focus_paper']), process_arxiv(sample['opp_paper']), sample['title_focus'], sample['title_opp'], sample['topic']
#     f_abs, f_intro = extract_sections_from_markdown(f_pap,'Abstract'), extract_sections_from_markdown(f_pap,'Introduction')
#     o_abs, o_intro = extract_sections_from_markdown(opp_pap,'Introduction'), extract_sections_from_markdown(opp_pap,'Introduction')
#     j_dat = {'focus_paper': sample['focus_paper'],
#              'opp_paper': sample['opp_paper'],
#              'title_focus': sample['title_focus'],
#              'title_opp': sample['title_opp'],
#              'topic': sample['topic'],
#              'full_focus_paper': f_pap,
#              'full_opp_paper': opp_pap,
#              'f_abstract': f_abs,
#              'f_intro': f_intro,
#              'o_abstract': o_abs,
#              'o_intro': o_intro,
#              }
#     opp_pap.append(j_dat)




