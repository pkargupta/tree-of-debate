import json
import argparse
from vllm import SamplingParams
from pydantic import BaseModel, StringConstraints, conlist
from typing_extensions import Annotated
from outlines.serve.vllm import JSONLogitsProcessor

class summary_schema(BaseModel):
    summary: Annotated[str, StringConstraints(strip_whitespace=True, min_length=50)]

##### SINGLE-STAGE PROMPTS #####
single_stage_prompt = lambda paper_0, paper_1: f"""
<paper_0>
Paper 0's Title: {paper_0['title']}

Paper 0's Abstract: {paper_0['abstract']}

Paper 0's Introduction: {paper_0['introduction']}
</paper_0>

<paper_1>
Paper 1's Title: {paper_1['title']}

Paper 1's Abstract: {paper_1['abstract']}

Paper 1's Introduction: {paper_1['introduction']}
</paper_1>

Output an approximately paragraph-long comparative summary of the similarities and differences between the two papers. Loosely structure your summary with initially their similarities (which ideas/aspects overlap between the two papers?) to their differences (what makes the papers unique) in novelties strictly based the information discussed within the input title, abstract, and introduction. Focus more on the differences than the similarities. Do not refer to the papers as Paper 0 or 1 in your output summary.

Format your output in the following JSON schema:
{{
    "summary": <5-20 sentence string to summarize the similarities and differences between the two papers>
}}"""

##### TWO-STAGE PROMPTS #####

two_stage_prompt_a = lambda paper: f"""
<paper>
Paper Title: {paper['title']}

Paper Abstract: {paper['abstract']}

Paper Introduction: {paper['introduction']}
</paper>

Output an approximately paragraph-long summary of the motivation and novelties of the paper (<paper>) based on its abstract and introduction.

Format your output in the following JSON schema:
{{
    "summary": <5-20 sentence string to summarize the similarities and differences between the two papers>
}}"""

two_stage_prompt_b = lambda paper_0, paper_1: f"""
<paper_0>
Paper 0's Title: {paper_0['title']}
Paper 0's Summary: {paper_0['summary']}
</paper_0>

<paper_1>
Paper 1's Title: {paper_1['title']}
Paper 1's Summary: {paper_1['summary']}
</paper_1>

For each of two papers, you are provided with a summary of their motivations and novelties. Output an approximately paragraph-long comparative summary of the similarities and differences between the two papers. Loosely structure your summary with initially their similarities (which ideas/aspects overlap between the two papers?) to their differences (what makes the papers unique) in novelties strictly based the information discussed within the input titles and summaries. Focus more on the differences than the similarities. Do not refer to the papers as Paper 0 or 1 in your output summary.

Format your output in the following JSON schema:
{{
    "summary": <5-20 sentence string to summarize the similarities and differences between the two papers>
}}"""


def run_baseline_code(args, paper_0, paper_1, model_server):
    if args.experiment == 'single':
        logits_processor = JSONLogitsProcessor(schema=summary_schema, llm=model_server.llm_engine)
        sampling_params = SamplingParams(max_tokens=3000, temperature=0.4, top_p=0.99,logits_processors=[logits_processor])
        output = model_server.generate([single_stage_prompt(paper_0, paper_1)], sampling_params=sampling_params, use_tqdm=False)[0].outputs[0].text
        summary = json.loads(output)['summary']

        with open(f'{args.log_dir}/{args.experiment}_summary.txt', 'w') as f:
            f.write(summary)
    
    elif args.experiment == 'two':
        # generate individual summaries
        logits_processor = JSONLogitsProcessor(schema=summary_schema, llm=model_server.llm_engine)
        sampling_params = SamplingParams(max_tokens=3000, temperature=0.4, top_p=0.99,logits_processors=[logits_processor])
        
        stage_a = model_server.generate([two_stage_prompt_a(paper_0), two_stage_prompt_a(paper_1)],
                                 sampling_params=sampling_params, use_tqdm=False)
        paper_0['summary'] = json.loads(stage_a[0].outputs[0].text)['summary']
        paper_1['summary'] = json.loads(stage_a[1].outputs[0].text)['summary']

        # generate joint comparative summary
        stage_b = model_server.generate([two_stage_prompt_b(paper_0, paper_1)], sampling_params=sampling_params, use_tqdm=False)[0].outputs[0].text
        summary = json.loads(stage_b)['summary']

        with open(f'{args.log_dir}/{args.experiment}_summary.txt', 'w', encoding="utf-8") as f:
            f.write(summary)
    
    else: 
        print("invalid experiment!")