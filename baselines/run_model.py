import pandas as pd
import json
import argparse
from vllm import LLM, SamplingParams
from utils import process_arxiv,extract_sections_from_markdown
from pydantic import BaseModel, StringConstraints, conlist
from typing_extensions import Annotated
from outlines.serve.vllm import JSONLogitsProcessor
import os

def process(s):
    # s = ''.join(s.split(' ')[:2])
    # s = [x for x in s if x.isalnum()]
    # return ''.join(s).lower()
    return s[s.rfind('/')+1:].replace('.', '_')

class string_schema(BaseModel):
    author_response: Annotated[str, StringConstraints(strip_whitespace=True)]

class summary_schema(BaseModel):
    similarities: conlist(Annotated[str, StringConstraints(strip_whitespace=True)], min_length=1,max_length=10) #Annotated[str, StringConstraints(strip_whitespace=True)]#
    differences: conlist(Annotated[str, StringConstraints(strip_whitespace=True)], min_length=1,max_length=10) #Annotated[str, StringConstraints(strip_whitespace=True)]#
    conclusion: Annotated[str, StringConstraints(strip_whitespace=True)]


def prompt_intro_abs(model,data):
    logits_processor = JSONLogitsProcessor(schema=summary_schema, llm=model.llm_engine)
    sampling_params = SamplingParams(max_tokens=3000, temperature=0.4, top_p=0.99,logits_processors=[logits_processor])
    prompts = []
    # document_f = []
    # document_o = []
    for i,sample in data.iterrows():
        sample = data.iloc[i]
        # sample = row.to_dict()
        f_abs, f_intro, f_tit, o_abs, o_intro, opp_tit, topic = sample['f_abstract'], sample['f_intro'], sample['title_focus'], sample['o_abstract'], sample['o_intro'], sample['title_opp'], sample['topic']
        # f_abs, f_intro = extract_sections_from_markdown(f_pap,'Abstract'), extract_sections_from_markdown(f_pap,'Introduction')
        # o_abs, o_intro = extract_sections_from_markdown(opp_pap,'Introduction'), extract_sections_from_markdown(opp_pap,'Introduction')
        prompt = f"""You are an helpful assistant. Given abstract and intros of two papers, write a finegrained comparative summary. <author 0> Title: {f_tit} Abstract: {f_abs}\n Introduction {f_intro} </author 0> <author 1> Title: {opp_tit} Abstract: {o_abs}\n Introduction {o_intro} </author 2>. Write a comparative summary between the papers based on the topic: {topic} with similarities, differences and conclusion. The comparative summary should be from the point of view of contrasting author 0's contrubutions against author 1. You must try to answer which components of author 0 are novel vs overlapping with author 1. Use author 0 and author 1 to mention while mentioning any claims. If there are no similarities or differences, mention it in the respective section. Format the output as the following JSON schema: {{
            "similarities": [should be a list of strings where the values are the similarities between the papers],
            "differences": [should be a list of strings where the values are the differences between the papers],
            "conclusion": <should be a strings where the value is the overall conclusion of the comparative summary between the papers>
        }}"""
        prompts.append(prompt)
        # document_f.append(f'Title: {f_tit} Abstract: {f_abs}\n Introduction {f_intro}')
        # document_o.append(f'Title: {opp_tit} Abstract: {o_abs}\n Introduction {o_intro}')

    summaries = model.generate(prompts,
                    sampling_params=sampling_params,
                    use_tqdm=True)
    # res = []
    # for i,sample in enumerate(data):
    #     sample['prompt_intro_abs'] = summaries[i].outputs[0].text
    comp_summaries = [json.loads(summaries[i].outputs[0].text) for i in range(len(summaries))]
    similarities = [row['similarities'] for row in comp_summaries]
    differences = [row['differences'] for row in comp_summaries]
    conclusion = [row['conclusion'] for row in comp_summaries]

    data['simi`larities'] = similarities
    data['differences'] = differences
    data['conclusion'] = conclusion
    return data
    # for i in summaries:
    #     res.append(i.outputs[0].text)
    # return res,document_f,document_o


def split_posthoc(model,data):
    sampling_params = SamplingParams(max_tokens=3000, temperature=0.4, top_p=0.99)
    prompts_pap1 = []
    prompts_pap2 = []
    # document_f = []
    # document_o = []
    for i,sample in data.iterrows():
        sample = data.iloc[i]
        f_abs, f_intro, f_tit, o_abs, o_intro, opp_tit, topic = sample['f_abstract'], sample['f_intro'], sample['title_focus'], sample['o_abstract'], sample['o_intro'], sample['title_opp'], sample['topic']
        f_prompt = f'You are an helpful assistant. Given abstract and intros, write a finegrained summary related to topic {topic} detailing key contributions, innovations and any other criteria you may find fit. <paper> Title: {f_tit} Abstract: {f_abs}\n Introduction {f_intro} </paper>'
        prompts_pap1.append(f_prompt)
        # document_f.append(f'Title: {f_tit} Abstract: {f_abs}\n Introduction {f_intro}')
        o_prompt = f'You are an helpful assistant. Given abstract and intros, write a finegrained summary detailing key contributions, innovations and any other criteria you may find fit. <paper> Title: {opp_tit} Abstract: {o_abs}\n Introduction {o_intro} </paper>'
        prompts_pap2.append(o_prompt)
        # document_o.append(f'Title: {opp_tit} Abstract: {o_abs}\n Introduction {o_intro}')
    
    # input(f'\n\n\n{i}, {len(data)}, {len(prompts_pap1)}, {len(prompts_pap2)}')
    f_summaries = model.generate(prompts_pap1,
                    sampling_params=sampling_params,
                    use_tqdm=True)
    f_res = []
    for i in f_summaries:
        f_res.append(i.outputs[0].text)
    o_summaries = model.generate(prompts_pap2,
                    sampling_params=sampling_params,
                    use_tqdm=True)
    o_res = []
    for i in o_summaries:
        o_res.append(i.outputs[0].text)
    logits_processor = JSONLogitsProcessor(schema=summary_schema, llm=model.llm_engine)
    sampling_params = SamplingParams(max_tokens=3000, temperature=0.4, top_p=0.99,logits_processors=[logits_processor])
    comp_prompts = []
    for i in range(len(o_res)):
        comp_prompt = f"""You are an helpful assistant. Given summaries of two papers write a finegrained comparative summary. <paper1> Summary: {f_res[i]} </paper1> <paper2> Summary: {o_res[i]} </paper2>. Write a comparative summary between the papers with similarities, differences and conclusion. The comparative summary shoudl be based on topic {topic} and should be from the point of view of contrasting author 0's contrubutions against author 1. You must try to answer which components of author 0 are novel vs overlapping with author 1. Use author 0 and author 1 to mention while mentioning any claims. If there are no similarities or differences, mention it in the respective section. Format the output as the following JSON schema: {{
            "similarities": [should be a list of strings where the values are the similarities between the papers],
            "differences": [should be a list of strings where the values are the differences between the papers>],
            "conclusion": <should be a strings where the value is the overall conclusion of the comparative summary between the papers>"""
        comp_prompts.append(comp_prompt)
    
    comp_summaries = model.generate(comp_prompts,
                    sampling_params=sampling_params,
                    use_tqdm=True)
    
    # for i,sample in enumerate(data):
    #     sample['split_posthoc'] = comp_summaries[i].outputs[0].text
    comp_summaries = [json.loads(comp_summaries[i].outputs[0].text) for i in range(len(comp_summaries))]
    similarities = [row['similarities'] for row in comp_summaries]
    differences = [row['differences'] for row in comp_summaries]
    conclusion = [row['conclusion'] for row in comp_summaries]

    data['similarities'] = similarities
    data['differences'] = differences
    data['conclusion'] = conclusion
    return data
    
    # c_res = []
    # for i in comp_summaries:
    #     c_res.append(i.outputs[0].text)
        
    # return c_res,document_f,document_o



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="baseline_data.tsv")
    parser.add_argument("--base_llm", default="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF")
    parser.add_argument("--baseline_type", default="prompt_intro")
    args = parser.parse_args()
    
    data = pd.read_csv(args.dataset_path, sep='\t')
    sampling_params = SamplingParams(max_tokens=3000, temperature=0.4, top_p=0.99, min_tokens=64)
    model_server = LLM(model=args.base_llm,tensor_parallel_size=2,max_num_seqs=100,enable_prefix_caching=True)

    if args.baseline_type=="prompt_intro":
        results = prompt_intro_abs(model_server,data)
    elif args.baseline_type=="split":
        results = split_posthoc(model_server,data)
    
    # output_path = f"opp_pap_data_{args.baseline_type}.json"
    # with open(output_path, 'w') as f:
    #     json.dump(results, f, indent=4)
    
    # print(f"Data written to {output_path}")

    # # print(len(results))
    # data['summary'] = results[0]
    # data['author 0'] = results[1]
    # data['author 1'] = results[2]

    # data.to_csv(f'results_{args.baseline_type}.tsv', sep='\t', index=False)
            # outputs = split_posthoc()
        
    for index, row in data.iterrows():
        shorthand = process(row['focus_paper']) + "-" + process(row['opp_paper'])
        if not os.path.exists(f'../logs/{shorthand}/'):
            os.mkdir(f'../logs/{shorthand}/')
        with open(f'../logs/{shorthand}/summary_{args.baseline_type}.txt', 'w+') as f:
            f.write(str(row['conclusion']))
            f.write(str(row['similarities']))
            f.write(str(row['differences']))
        with open(f'../logs/{shorthand}/evidence_{args.baseline_type}.txt', 'w+') as f:
            f.write(str(row['f_abstract']) + " " + str(row['f_intro']))
            f.write(str(row['o_abstract']) + " " + str(row['o_intro']))
