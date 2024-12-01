from vllm import SamplingParams
from outlines.serve.vllm import JSONLogitsProcessor
from unidecode import unidecode
from persona import log_llm
import json
from pydantic import BaseModel, StringConstraints, conlist
from typing_extensions import Annotated
from debate import DebateNode

class summary_schema(BaseModel):
    summary: Annotated[str, StringConstraints(strip_whitespace=True, min_length=50)]

# class expansion_schema(BaseModel):
#     explanation: Annotated[str, StringConstraints(strip_whitespace=True, min_length=5)]
#     is_expand: bool

class expansion_schema(BaseModel):
    explanation: Annotated[str, StringConstraints(strip_whitespace=True, min_length=5)]
    progression_of_arguments: bool
    meaningful_questions: bool
    author_0_has_won: bool

def arg_dict_to_str(args, arg_type=True):
    arguments = ""
    for i, key in enumerate(args.keys()):
        if arg_type:
            arguments += f"Author {(i)}'s Initial Argument: {args[key]['argument_title']}: {args[key]['description']}. "
        else:
            arguments += f"Author {(i)}'s Revised Argument: {args[key]['revised_argument_title']}: {args[key]['revised_argument_description']}. "

    return arguments.strip()

class Moderator:
    def __init__(self, model, log_dir):
        self.model = model # define model - Llama 3.
        self.log_dir = log_dir
    
    def is_expand(self, round: DebateNode, history):
        """
        Based on the previous arguments and the new arguments, determine if any progress has been made.
        """
        prev_args = arg_dict_to_str(round.init_arguments, True)
        new_args = arg_dict_to_str(round.final_arguments, False)
        logits_processor = JSONLogitsProcessor(schema=expansion_schema, llm=self.model.llm_engine)
        sampling_params = SamplingParams(max_tokens=1024, logits_processors=[logits_processor])

        round_topic = round.round_topic


        prompt = f"""You are a moderator faciliating a debate in which a focus scientific paper (Author 0) is claiming that it makes the following novel contribution:\n\t- Topic: {round_topic['argument_title']}\n\t- Topic Description: {round_topic['description']}\n
Author 0's claim is that its paper's contributions towards the topic are all novel relative to Author 1's paper.
Author 1's claim is that Author 0's paper's contributions towards the topic are NOT novel relative to Author 1's own paper.

-------

Here is the debate history:
{history}

-------

Below, you are given the previous set of arguments and the current set of arguments. 

\"previous arguments\":\n{prev_args}

\"current arguments\":\n{new_args}

-------

You must determine whether progress is being made. DO NOT focus on the language being used. Focus on the content of the arguments. Specifically, determine the following (True or False for each):
1. progression_of_arguments: Are these arguments sufficiently different enough to necesitate further debate? Are there new, deeper concepts being discussed between the two sets of arguments?
2. meaningful_questions: Within the debate history, each author acknowledges each other's arguments and may ask clarifying questions accordingly. Do you believe that the clarifying questions have not been sufficiently addressed already and would be important to answer through further debate? If there are no questions raised in the debate history by either author, return False.
3. author_0_has_won: Do you believe that it is clear that Author 0's claim is completely novel and does not need to be further deconstructured (in order to determine which components within the claim are truly novel versus overlap with Author 1's paper)?

Output your argument in the following JSON format: 
{{
    "explanation": <2-5 sentence string to explain your reasoning about whether further debate is necessary when comparing the \"previous arguments\" and the \"current arguments\">,
    "progression_of_arguments": <output a boolean; pick only one of "True" or "False" depending on the history, arguments, and your explanation above>,
    "meaningful_questions": <output a boolean; pick only one of "True" or "False" depending on the history, arguments, and your explanation above>,
    "author_0_has_won": <output a boolean; pick only one of "True" or "False" depending on the history, arguments, and your explanation above>
}}
"""
        # conversation = history.extend(conversation)
        outputs = unidecode(self.model.generate(prompt,
                    sampling_params=sampling_params,
                    use_tqdm=False)[0].outputs[0].text)
        print(f'IS EXPAND {outputs}')
        log_llm(self.log_dir, prompt, outputs)
        outputs = json.loads(outputs)

        return (outputs['progression_of_arguments'] or outputs['meaningful_questions']) and (not outputs['author_0_has_won'])

    def summarize_debate(self, conversation_history, similarities, differences):
        similarities = ",".join(similarities)
        differences = ",".join(differences)

        logits_processor = JSONLogitsProcessor(schema=summary_schema, llm=self.model.llm_engine)
        sampling_params = SamplingParams(max_tokens=1024, logits_processors=[logits_processor])

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
        log_llm(self.log_dir, prompt, outputs)
        outputs = json.loads(outputs)['summary']
        return outputs
