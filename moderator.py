from vllm import SamplingParams
from outlines.serve.vllm import JSONLogitsProcessor
from unidecode import unidecode
from persona import log_llm
import json
from pydantic import BaseModel, StringConstraints, conlist
from typing_extensions import Annotated
from debate import DebateNode

class summary_schema(BaseModel):
    summary: Annotated[str, StringConstraints(strip_whitespace=True)]

class expansion_schema(BaseModel):
    explanation: Annotated[str, StringConstraints(strip_whitespace=True, min_length=5)]
    is_expand: bool

def arg_dict_to_str(args, arg_type=True):
    arguments = ""
    for i, key in enumerate(args.keys()):
        if arg_type:
            arguments += f"{(i+1)}. {args[key]['argument_title']}: {args[key]['description']}. "
        else:
            arguments += f"{(i+1)}. {args[key]['revised_argument_title']}: {args[key]['revised_argument_description']}. "

    return arguments.strip()

class Moderator:
    def __init__(self, model, log_dir):
        self.model = model # define model - Llama 3.
        self.log_dir = log_dir
    
    def is_expand(self, round: DebateNode):
        """
        Based on the previous arguments and the new arguments, determine if any progress has been made.
        """
        prev_args = arg_dict_to_str(round.init_arguments, True)
        new_args = arg_dict_to_str(round.final_arguments, False)
        logits_processor = JSONLogitsProcessor(schema=expansion_schema, llm=self.model.llm_engine)
        sampling_params = SamplingParams(max_tokens=1024, logits_processors=[logits_processor])

        prompt = f"""You are a moderator to a debate in which two scientific papers are being discussed. The topic of the debate is \"{round.round_topic}\". Below, you are given the previous set of arguments and the current set of arguments. 

\"previous arguments\": {prev_args}

\"current arguments\": {new_args}

You must determine whether progress is being made. DO NOT focus on the language being used. Focus on the content of the arguments. Specifically, are these arguments sufficiently different enough to necesitate further debate? Are there new, deeper concepts being discussed between the two sets of arguments?

Output your argument in the following JSON format: 
{{
    "explanation": <2-5 sentence string to explain your reasoning about whether further debate is necessary when comparing the \"previous arguments\" and the \"current arguments\">,
    "is_expand": <output a boolean; pick only one of "True" or "False" depending on the explanation above>
}}
"""
        # conversation = history.extend(conversation)
        outputs = unidecode(self.model.generate(prompt,
                    sampling_params=sampling_params,
                    use_tqdm=False)[0].outputs[0].text)
        print(f'IS EXPAND {outputs}')
        log_llm(self.log_dir, prompt, outputs)
        outputs = json.loads(outputs)

        return outputs['is_expand']

    def summarize_debate(self, conversation_history, similarities, differences):
        similarities = ",".join(similarities)
        differences = ",".join(differences)

        logits_processor = JSONLogitsProcessor(schema=summary_schema, llm=self.model.llm_engine)
        sampling_params = SamplingParams(max_tokens=1024, logits_processors=[logits_processor],min_tokens=50)

        prompt = f"""The authors of two papers have debated about the similarities and differences between their papers. Author 0 is the author of the main paper, while Author 1 is the author of the paper being compared to the main paper. Below, you are given the \"conversation_history\" between the authors, and the specific similarities and differences. The similarities and differences are from the point-of-view of Author 0.

\"conversation_history\":\n{conversation_history}

\"similarities\": {similarities}

\"differences\": {differences}

Your task is to write a synthesis of the debate that summarizes the similarities and differences between the papers. Focus more on the differences than the similarities. Format the output as a schema:
    {{
        "summary": <5-10 sentence string to summarize the similarities and differences between the two papesr>
    }}
"""
        # conversation = history.extend(conversation)
        outputs = unidecode(self.model.generate(prompt,
                    sampling_params=sampling_params,
                    use_tqdm=False)[0].outputs[0].text)
        log_llm(prompt, outputs)
        outputs = json.loads(outputs)['summary']
        return outputs
