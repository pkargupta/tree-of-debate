from paper_details import Paper
from vllm import SamplingParams
from outlines.serve.vllm import JSONLogitsProcessor
from unidecode import unidecode
from pydantic import BaseModel, StringConstraints, conlist
from typing_extensions import Annotated
import json
import re

def format_evidence(list_evi, author, ids=None):
    text_evi = ""
    idx = 1
    for counter, item in enumerate(list_evi):
        if (ids == None) or ((counter + 1) in ids):
            text_evi += f"\t- Author {author.id}'s Supporting Evidence #{idx}. {item}\n"
            idx += 1
    return text_evi

def format_preemption(author, list_pre):
    text_pre = f"\t- Author {1-author.id}'s relevant evidence to potentially counter the quality of this contribution:\n"
    for e_id, e in enumerate(list_pre):
        text_pre += f"\t\t- Author {1-author.id}'s Counter Evidence #{e_id+1}: The opposition states, \"{e}\"\n"
    return text_pre

def format_debate_context(you, opposition, parent_debate_node, debate_node):
    out = ""
    
    your_contributions = debate_node.round_topic['author_0_relevant_contributions'] if you.id == 0 else debate_node.round_topic['author_1_relevant_contributions']
    if len(your_contributions) > 0:
        out += f"""Here are your (Author {you.id}) claimed contributions towards the topic:\n"""
        for idx, cont in enumerate(your_contributions):
            arg = parent_debate_node.self_delib[you.id][cont-1]
            out += f"Author {you.id} Paper's Contribution #{idx+1}: {arg['argument_title']}: {arg['description']}\n"
            out += f"{format_evidence(parent_debate_node.evidence[you.id], you, arg['evidence'])}"
            arg_key = f"{arg['argument_title']}: {arg['description']}"
            out += f'{format_preemption(you, parent_debate_node.preemption[opposition.id][arg_key])}\n'
    else:
        out += f"""Here are some additional excerpts from your paper that may help you:\n{format_evidence(parent_debate_node.evidence[you.id], you)}\n"""
    
    opp_contributions = debate_node.round_topic['author_0_relevant_contributions'] if opposition.id == 0 else debate_node.round_topic['author_1_relevant_contributions']
    if len(opp_contributions) > 0:
        out += f"""Here are your opposition's (Author {opposition.id}) claimed contributions towards the topic:\n"""
        for idx, cont in enumerate(opp_contributions):
            arg = parent_debate_node.self_delib[opposition.id][cont-1]
            out += f"Author {opposition.id} Paper's Contribution #{idx+1}: {arg['argument_title']}: {arg['description']}\n"
            out += f"{format_evidence(parent_debate_node.evidence[opposition.id], opposition, arg['evidence'])}"
            arg_key = f"{arg['argument_title']}: {arg['description']}"
            out += f"{format_preemption(opposition, parent_debate_node.preemption[you.id][arg_key])}\n"
    else:
        out += f"""Here are some additional excerpts from the opposition's paper that may help you:\n{format_evidence(parent_debate_node.evidence[opposition.id], opposition)}\n"""
    
    return out

def format_args(list_arg):
    text_args = ""
    for counter, item in enumerate(list_arg):
        text_args += f"\t- Argument #{counter+1}. {item['argument_title']}: {item['description']}\n"
    return text_args

class revise_schema(BaseModel):
    revised_argument_title: Annotated[str, StringConstraints(strip_whitespace=True)]
    revised_argument_description: Annotated[str, StringConstraints(strip_whitespace=True, min_length=10)]

class argument_schema(BaseModel):
    argument_title: Annotated[str, StringConstraints(strip_whitespace=True)]
    description: Annotated[str, StringConstraints(strip_whitespace=True)]

class response_schema(BaseModel):
    author_response: Annotated[str, StringConstraints(strip_whitespace=True, min_length=50)]

class gen_argument_schema(BaseModel):
    argument_title: Annotated[str, StringConstraints(strip_whitespace=True)]
    description: Annotated[str, StringConstraints(strip_whitespace=True)]
    
class argument_list_schema(BaseModel):
    argument_list : conlist(gen_argument_schema, min_length=1,max_length=10)

def log_llm(log_dir, prompt, output):
    with open(f'{log_dir}/llm_calls.txt', 'a+') as f:
        f.write('--------------------------------------------\n')
        f.write(f'PROMPT: {prompt}\n')
        f.write(f'OUTPUT: {output}\n')
        f.write('--------------------------------------------\n\n')

class PaperAuthor:
    def __init__(self, model, id, paper: Paper, focus, log_dir):
        self.model = model # define model - Llama 3.1
        self.paper = paper
        self.focus = focus
        self.id = id
        self.log_dir = log_dir

    def generate_arguments(self, topic, evidence=False, temperature=0.3, top_p=0.99, k=3):
        """
        Given topic and evidence, generate k arguments. 
        If the paper is a focus paper, and the debate round is round #1, the topic should be "I am great".
        If the paper is NOT a focus paper, the topic should be the focus paper's arguments.
        """
        logits_processor = JSONLogitsProcessor(schema=argument_list_schema, llm=self.model.llm_engine)
        sampling_params = SamplingParams(max_tokens=2000, logits_processors=[logits_processor], temperature=temperature, top_p=top_p)

        prompt = f"You are the author of the paper, '{self.paper.title}'. The abstract of your work is:\n{self.paper.abstract}.\n The introduction of your work is: {self.paper.introduction}.\n\nYou are debating another author on the novel contributions that your work makes towards the following topic:\n"

        if topic['topic_title'] == topic['topic_description']:
            # if it's the root node:
            prompt += f"{topic['topic_title']}\n"
        else:
            # if it's a debate node, then topic is the focus paper's claim
            prompt += f"{topic['topic_title']}: {topic['topic_description']}\n"

        prompt += f"""Based on your paper's title, abstract, and introduction, output a list of 1 to {k} DIVERSE, specific arguments for your position. Each argument should have a corresponding "argument_title", which is a brief statement of your argument (e.g., Better Efficiency for Training) and a "description" explaining your argument and mentioning specific excerpts from your papers. Each argument should make a unique point.
        
Output your list of arguments in the following JSON format:
{{
    "argument_list":
        [
            {{
                "argument_title": <should be a brief, 10-15 word string where the value is the argument_title>,
                "description": <1-2 sentence string explaining the argument, including specific excerpts from the evidence pool>
            }}
        ]
}}
"""
        outputs = unidecode(self.model.generate(prompt,
                    sampling_params=sampling_params)[0].outputs[0].text)
        log_llm(self.log_dir, prompt, outputs)
        return json.loads(outputs)['argument_list']

    def present_argument(self, debate_node, parent_debate_node, opposition, temperature=0.1, top_p=0.99):
        """
        Generate an argument based on your claims and evidences and other paper's claims and evidences.
        """
        logits_processor = JSONLogitsProcessor(schema=argument_schema, llm=self.model.llm_engine)
        sampling_params = SamplingParams(max_tokens=3000, logits_processors=[logits_processor], temperature=temperature, top_p=top_p)

        round_topic = debate_node.round_topic

        prompt = f"You are the author of the paper, '{self.paper.title}'. The abstract of your work is:\n{self.paper.abstract}\n\nHere is the introduction to your paper:\n{self.paper.introduction}\n\nYou are debating another author (Opposition), whose work is titled, '{opposition.paper.title}', and abstract is:\n{opposition.paper.abstract}\n\n"

        prompt += f"""You are debating the other author on how and why your paper makes a better contribution towards the following topic:\n\t- Topic: {round_topic['topic_title']}\n\t- Topic Description: {round_topic['topic_description']}

{format_debate_context(self, opposition, parent_debate_node, debate_node)}

Given the above, make an argument for a specific reason why your contributions towards the topic, {round_topic['topic_title']}, are better than the opposition's. If you feel that you do not contribute to the given topic or your contributions ARE NOT better than the opposition's, then state so by conceding to the opposition (e.g., 'I do not believe my paper makes a better contribution than yours') and explain why. 

Output your argument in the following JSON format:

{{
    "argument_title": <should be a brief, 10-15 word string where the value is the argument_title>,
    "description": <2-3 sentence string explaining your argument>
}}
"""     
        print(prompt + '\n\n')
        outputs = unidecode(self.model.generate(prompt,
                    sampling_params=sampling_params,
                    use_tqdm=False)[0].outputs[0].text)
        log_llm(self.log_dir, prompt, outputs)

        return json.loads((outputs))

    
    def respond_to_argument(self, history, debate_node, parent_debate_node, opposition, temperature=0.4, top_p=0.99):
        """
        Respond to the paper given the current state of debate.
        """
        
        logits_processor = JSONLogitsProcessor(schema=response_schema, llm=self.model.llm_engine)
        sampling_params = SamplingParams(max_tokens=3000, logits_processors=[logits_processor], temperature=temperature, top_p=top_p)

        last_occurrence_index = history.rfind("-Opposition:")
        
        # Check if "-Opposition:" exists in the string
        if last_occurrence_index != -1:
            history = history[:last_occurrence_index] + "<respond_to_this>\n" + history[last_occurrence_index:]+"</respond_to_this>"
        print(f'HISTORY {history}')
        
        round_topic = debate_node.round_topic

        prompt = f"You are the author of the paper, '{self.paper.title}'. The abstract of your work is:\n{self.paper.abstract}\n\nYou are debating another author (Opposition), whose work is titled, '{opposition.paper.title}', and abstract is:\n{opposition.paper.abstract}\n\n"

        prompt += f"""You are debating the other author on how and why your paper makes a better contribution towards the following topic:\n\t- Topic: {round_topic['topic_title']}\n\t- Topic Description: {round_topic['topic_description']}

{format_debate_context(self, opposition, parent_debate_node, debate_node)}
"""
        
        prompt+= f"""Here is your conversation debate history with the opposition paper. You must respond to the last argument presented by your opposition in debate (tagged between <respond_to_this> and </respond_to_this>). A response may consist of (1) an acknowledgement of the opposition's previous response, (2) answering any of the questions about your paper brought up by the opposition, (3) asking any clarifying questions based on the opposition's claims and reasoning, (4) any clarifications of your own presented arguments based on the opposition, and/or (5) if you feel that the opposition's claim is strong and you do not have sufficient grounds to refute it, then a concession to your opposition.\n\n""" + history
        
        prompt += f"""\nOutput your new response in the following JSON format:
{{
    "author_response": <2-3 sentence string response to the opposition's last turn with an explanation behind your reasoning (tagged between <respond_to_this> and </respond_to_this>)>
}}
"""
        print(prompt + '\n\n')
        # conversation = history.extend(conversation)
        outputs = unidecode(self.model.generate(prompt,
                    sampling_params=sampling_params,
                    use_tqdm=False)[0].outputs[0].text)
        
        log_llm(self.log_dir, prompt, outputs)
        return json.loads(outputs)
    
    def revise_argument(self, history, debate_node, parent_debate_node, opposition, temperature=0.45, top_p=0.99):
        """
        Strengthen the final argument at the debate node for a paper.
        """

        logits_processor = JSONLogitsProcessor(schema=revise_schema, llm=self.model.llm_engine)
        sampling_params = SamplingParams(max_tokens=3000, logits_processors=[logits_processor], temperature=temperature, top_p=top_p)

        round_topic = debate_node.round_topic
        
        prompt = f"You are the author of the paper, '{self.paper.title}'. The abstract of your work is:\n{self.paper.abstract}\n\nYou are debating another author (Opposition), whose work is titled, '{opposition.paper.title}', and abstract is:\n{opposition.paper.abstract}\n\n"

        prompt += f"""You are debating the other author on how and why your paper makes a better contribution towards the following topic:\n\t- Topic: {round_topic['topic_title']}\n\t- Topic Description: {round_topic['topic_description']}

{format_debate_context(self, opposition, parent_debate_node, debate_node)}
"""
        
        prompt+= f"""Based on the debate history and your/your opposition's arguments and evidence, you must construct a new, stronger argument related to the topic. This consists of an argument that addresses/is robust to any doubts or clarifying questions made by the opposition which you feel are valid. If based on the debate, you feel that you do not contribute to the given topic or your contributions ARE NOT better than the opposition's, then state so by conceding to the opposition (e.g., 'I do not believe my paper makes a better contribution than yours') and explain why. \n\n""" + history
        
        prompt += f"""\nOutput your new, revised argument in the following JSON format:
{{
    "revised_argument_title": <should be a brief, 10-15 word string where the value is your revised argument on your paper's novelty toward the "topic" based on your debate with the opposition>,
    "revised_argument_description": <2-3 sentence string explaining your new argument>
}}
"""
        print(prompt + '\n\n')
        
        # conversation = history.extend(conversation)
        outputs = unidecode(self.model.generate(prompt,
                    sampling_params=sampling_params,
                    use_tqdm=False)[0].outputs[0].text)
        
        log_llm(self.log_dir, prompt, outputs)
        return json.loads(outputs)
