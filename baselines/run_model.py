import pandas as pd
import argparse
from vllm import LLM, SamplingParams
from utils import process_arxiv,extract_sections_from_markdown
from pydantic import BaseModel, StringConstraints, conlist
from typing_extensions import Annotated
from outlines.serve.vllm import JSONLogitsProcessor


class summary_schema(BaseModel):
    similarities: conlist(Annotated[str, StringConstraints(strip_whitespace=True)], min_length=1,max_length=10)
    differences: conlist(Annotated[str, StringConstraints(strip_whitespace=True)], min_length=1,max_length=10)
    conclusion: Annotated[str, StringConstraints(strip_whitespace=True)]


def prompt_intro_abs(model,data):
    logits_processor = JSONLogitsProcessor(schema=summary_schema, llm=model.llm_engine)
    sampling_params = SamplingParams(max_tokens=3000, temperature=0.4, top_p=0.99, min_tokens=64,logits_processor=[logits_processor])
    prompts = []
    document_f = []
    document_o = []
    for i,row in data.iterrows():
        sample = row.to_dict()
        f_pap, opp_pap, f_tit, opp_tit, topic = process_arxiv(sample['focus_paper']), process_arxiv(sample['opp_paper']), sample['title_focus'], sample['title_opp'], sample['topic']
        f_abs, f_intro = extract_sections_from_markdown(f_pap,'Abstract'), extract_sections_from_markdown(f_pap,'Introduction')
        o_abs, o_intro = extract_sections_from_markdown(opp_pap,'Introduction'), extract_sections_from_markdown(opp_pap,'Introduction')
        prompt = f"""You are an helpful assistant. Given abstract and intros of two papers, write a finegrained comparative summary. <paper1> Title: {f_tit} Abstract: {f_abs}\n Introduction {f_intro} </paper1> <paper2> Title: {opp_tit} Abstract: {o_abs}\n Introduction {o_intro} </paper2>. Write a comparative summary between the papers with similarities, differences and conclusion. If there are no similarities or differences, mention it in the respective section. Format the output as the following schema: {{
            "similarities": <should be a list of strings where the values are the similarities between the papers>,
            "differences": <should be a list of strings where the values are the differences between the papers>
            "conclusion": <should be a strings where the value is the overall conclusion of the comparative summary between the papers>
        }}"""
        prompts.append(prompt)
        document_f.append(f'Title: {f_tit} Abstract: {f_abs}\n Introduction {f_intro}')
        document_o.append(f'Title: {opp_tit} Abstract: {o_abs}\n Introduction {o_intro}')

    summaries = model.generate(prompts,
                    sampling_params=sampling_params,
                    use_tqdm=True)
    res = []
    for i in summaries:
        res.append(i.outputs[0].text)
    return res,document_f,document_o


def split_posthoc(model,data):
    sampling_params = SamplingParams(max_tokens=3000, temperature=0.4, top_p=0.99, min_tokens=64)
    prompts_pap1 = []
    prompts_pap2 = []
    document_f = []
    document_o = []
    for i,row in data.iterrows():
        sample = row.to_dict()
        f_pap, opp_pap, f_tit, opp_tit, topic = process_arxiv(sample['focus_paper']), process_arxiv(sample['opp_paper']), sample['title_focus'], sample['title_opp'], sample['topic']
        f_abs, f_intro = extract_sections_from_markdown(f_pap,'Abstract'), extract_sections_from_markdown(f_pap,'Introduction')
        o_abs, o_intro = extract_sections_from_markdown(opp_pap,'Introduction'), extract_sections_from_markdown(opp_pap,'Introduction')
        f_prompt = f'You are an helpful assistant. Given abstract and intros, write a finegrained summary detailing key contributions, innovations and any other criteria you may find fit. <paper> Title: {f_tit} Abstract: {f_abs}\n Introduction {f_intro} </paper>'
        prompts_pap1.append(f_prompt)
        document_f.append(f'Title: {f_tit} Abstract: {f_abs}\n Introduction {f_intro}')
        o_prompt = f'You are an helpful assistant. Given abstract and intros, write a finegrained summary detailing key contributions, innovations and any other criteria you may find fit. <paper> Title: {opp_tit} Abstract: {o_abs}\n Introduction {o_intro} </paper>'
        prompts_pap2.append(o_prompt)
        document_o.append(f'Title: {opp_tit} Abstract: {o_abs}\n Introduction {o_intro}')
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
    sampling_params = SamplingParams(max_tokens=3000, temperature=0.4, top_p=0.99, min_tokens=64, logits_processor=[logits_processor])
    comp_prompts = []
    for i in range(len(o_res)):
        comp_prompt = f"""You are an helpful assistant. Given summaries of two papers write a finegrained comparative summary. <paper1> Summary: {f_res[i]} </paper1> <paper2> Summary: {o_res[i]} </paper2>. Write a comparative summary between the papers with similarities, differences and conclusion. If there are no similarities or differences, mention it in the respective section. Format the output as the following schema: {{
            "similarities": <should be a list of strings where the values are the similarities between the papers>,
            "differences": <should be a list of strings where the values are the differences between the papers>
            "conclusion": <should be a strings where the value is the overall conclusion of the comparative summary between the papers>"""
        comp_prompts.append(comp_prompt)
    
    comp_summaries = model.generate(comp_prompts,
                    sampling_params=sampling_params,
                    use_tqdm=True)
    c_res = []
    for i in comp_summaries:
        c_res.append(i.outputs[0].text)
        
    return c_res,document_f,document_o



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="data.csv")
    parser.add_argument("--base_llm", default="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF")
    parser.add_argument("--baseline_type", default="prompt_intro")
    args = parser.parse_args()
    
    data = pd.read_csv(args.dataset_path)
    sampling_params = SamplingParams(max_tokens=3000, temperature=0.4, top_p=0.99, min_tokens=64)
    model_server = LLM(model=args.base_llm,tensor_parallel_size=4,max_num_seqs=100,enable_prefix_caching=True)

    if args.baseline_type=="prompt_intro":
        results = prompt_intro_abs(model_server,data)
    elif args.baseline_type=="split":
        results = split_posthoc(model_server,data)
    # print(len(results))
    data['summary'] = results[0]
    data['document_f'] = results[1]
    data['document_o'] = results[2]

    data.to_csv(f'results_{args.baseline_type}.csv', index=False)
            # outputs = split_posthoc()
        
