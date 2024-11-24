from vllm import SamplingParams
from outlines.serve.vllm import JSONLogitsProcessor
from unidecode import unidecode
from persona import log_llm
import json
from pydantic import BaseModel, StringConstraints, conlist
from typing_extensions import Annotated
from debate import DebateNode

class expansion_schema(BaseModel):
    explanation: Annotated[str, StringConstraints(strip_whitespace=True)]
    is_expand: Annotated[str, StringConstraints(strip_whitespace=True)]

def arg_dict_to_str(args):
    arguments = ""
    for i, key in enumerate(args.keys()):
        arguments += f"{(i+1)}. {args[key]['argument_title']}. "

    return arguments.strip()

class Moderator:
    def __init__(self, model):
        self.model = model # define model - Llama 3.
    
    def is_expand(self, round: DebateNode):
        """
        Based on the previous arguments and the new arguments, determine if any progress has been made.
        """
        prev_args = arg_dict_to_str(round.init_arguments)
        new_args = arg_dict_to_str(round.final_arguments)
        logits_processor = JSONLogitsProcessor(schema=expansion_schema, llm=self.model.llm_engine)
        sampling_params = SamplingParams(max_tokens=1024, logits_processors=[logits_processor])
       
        prompt = f"""In this step you must strengthen your claims given the reponses of the opposition. Format the output as a schema: {{"argument_list":
                                                [
                                                    {{
                                                        "title": <should be a sentence-long string where the value is the high-level argument title>,
                                                        "description": <2-5 sentence string explaining the argument>
                                                    }}
                                                ]
                                            }}"""
        prompt = f"""You are a moderator to a debate in which two scientific papers are being discussed. The topic of the debate is \"{round.round_topic}\". Below, you are given the previous set of arguments and the current set of arguments. 

\"previous arguments\": {prev_args}

\"current arguments\": {new_args}

You must determine whether progress is being made. Specifically, are these arguments sufficiently different enough to necesitate further debate? Are there new, deeper concepts being discussed between the two sets of arguments? Format the output as a schema: {{"expansion":
                                                [
                                                    {{
                                                        "explanation": <2-5 sentence string explaining whether new concepts are being argued between the \"previous arguments\" and the \"current arguments\">,
                                                        "is_expand": <"True" or "False" depending on the explanation above>
                                                    }}
                                                ]
                                            }}"""
        # conversation = history.extend(conversation)
        outputs = unidecode(self.model.generate(prompt,
                    sampling_params=sampling_params,
                    use_tqdm=False)[0].outputs[0].text)
        print(f'IS EXPAND {outputs}')
        log_llm(prompt, outputs)
        outputs = json.loads(outputs)

        return outputs['is_expand'].lower() == "true"