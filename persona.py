from paper_details import Paper
from vllm import SamplingParams
from outlines.serve.vllm import JSONLogitsProcessor
from unidecode import unidecode
from pydantic import BaseModel, StringConstraints, conlist
from typing_extensions import Annotated
import json

def format_args(list_arg):
    text_args = ""
    for counter, item in enumerate(list_arg):
        text_args += f"\t- Argument #{counter+1}. {item['title']}: {item['description']}\n"
    return text_args

def format_evidence(list_evi):
    text_evi = ""
    for counter, item in enumerate(list_evi):
        text_evi += f"\t- Evidence #{counter+1}. {item}\n"
    return text_evi

def format_preemption(list_pre):
    text_pre = ""
    claim_no = 1
    for question, evidence_list in list_pre.items():
        mod_question = question.replace('the claim', 'the opposition\'s claim')
        text_pre += f"\t- Opposition Claim #{claim_no}: {mod_question}\n"
        for counter, e in enumerate(evidence_list):
            text_pre += f"\t- Your Counter Evidence #{counter+1}: {e}\n"
        text_pre += f"\n"
        claim_no += 1
    return text_pre

class argument_schema(BaseModel):
    title: Annotated[str, StringConstraints(strip_whitespace=True)]
    description: Annotated[str, StringConstraints(strip_whitespace=True)]


class argument_list_schema(BaseModel):
    argument_list : conlist(argument_schema, min_length=1,max_length=10)

def log_llm(prompt, output):
    with open('logs/llm_calls.txt', 'a+') as f:
        f.write('--------------------------------------------\n')
        f.write(f'PROMPT: {prompt}\n')
        f.write(f'OUTPUT: {output}\n')
        f.write('--------------------------------------------\n\n')

class PaperAuthor:
    def __init__(self, model, id, paper: Paper, focus):
        self.model = model # define model - Llama 3.1
        self.paper = paper
        self.focus = focus
        self.id = id

    def gather_evidence(self, topic, k=2, return_scores=True):
        """
        Use paper chunks to get relevant segments to the topic.
        """

        retrievals = self.paper.retrieve_top_k(topic, k=k)
        evidence, scores = [], []
        for retrieval in retrievals:
            evidence.append(retrieval[0])
            scores.append(retrieval[1])
        if return_scores:
            return evidence, scores
        return evidence

    def generate_arguments(self, topic, evidence=False, k=2):
        """
        Given topic and evidence, generate k arguments. 
        If the paper is a focus paper, and the debate round is round #1, the topic should be "I am great".
        If the paper is NOT a focus paper, the topic should be the focus paper's arguments.
        """
        logits_processor = JSONLogitsProcessor(schema=argument_list_schema, llm=self.model.llm_engine)
        sampling_params = SamplingParams(max_tokens=2000, logits_processors=[logits_processor])

        # TODO: distinct prompts between focus and cited papers
        prompt = f"You are a helpful assistant. You are an author of a paper that is trying to convince others of your paper's contributions towards {topic}."

        formatted_evidence = "\n".join([f'Retrieved Evidence #{idx+1}: {e}' for idx, e in enumerate(evidence)])
        prompt += f"""Below is a list of relevant evidence retrieved from your paper:\n\n{formatted_evidence}\nBased on the evidence, output a list of {k} diverse, specific arguments on your paper's major unique contributions towards {topic}, that are all supported by the evidence. Each argument should make a unique point. Output your list of arguments in the following JSON format:
{{"argument_list":
    [
        {{
            "title": <should be a brief, 10-15 word string where the value is the argument title>,
            "description": <1-2 sentence string explaining the argument>
        }}
    ]
}}"""
        outputs = unidecode(self.model.generate(prompt,
                    sampling_params=sampling_params)[0].outputs[0].text)
        log_llm(prompt, outputs)
        return json.loads(outputs)['argument_list']

    def preempt_arguments(self, counter_claims):
        """
        gathers evidence for why self is better than paper_b wrt paper_b's arguments/evidences.
        """
        # generate template to combine counter_claims, counter_evidence
        #   Does my paper also include or address a similar claim/idea?
        #   Does my paper propose a better claim/idea to address the problem solved by p_i's claim?
        extended_pool = {}

        for c in counter_claims:
            augmented_topic = f'Does my paper also address the claim, \"{c.lower()}\"?'
            extended_pool[augmented_topic] = self.gather_evidence(augmented_topic, return_scores=False)

            # augmented_topic = f'Does my paper propose a better claim/idea than the claim, \"{c}\"?'
            # extended_pool[augmented_topic] = self.gather_evidence(augmented_topic)

        return extended_pool
    
    def present_argument(self, debate_node, parent_debate_node, temperature=0.1, top_p=0.99):
        """
        Generate an argument based on your claims and evidences and other paper's claims and evidences.
        """
        logits_processor = JSONLogitsProcessor(schema=argument_schema, llm=self.model.llm_engine)
        sampling_params = SamplingParams(max_tokens=1024, logits_processors=[logits_processor], temperature=temperature, top_p=top_p)

        round_topic = debate_node.round_topic

        if self.focus:
            claim = "your paper's contributions towards the topic are all novel relative to the other paper"
            prompt = f"You are an author of a paper that is debating another author about the following topic:\n\t- Topic: {round_topic['title']}\n\t- Topic Description: {round_topic['description']}\n\nYour debate claim is that {claim}. Refer to their arguments and presented evidence, as well as your own paper's segments as evidence when refining your arguments.\n\n"

        else:
            claim = "the other paper's contributions towards the topic are not novel relative to your own paper"
            prompt = f"You are an author of a paper that is debating another author about their claimed novelty:\n\t- Novelty Claim: {round_topic['title']}\n\t- Novelty Claim Description: {round_topic['description']}\n\nYour debate claim is that {claim}. Refer to their arguments and presented evidence, as well as your own paper's segments as evidence when refining your arguments."
            
            prompt+=f"""\n\nYou initially argued that your own paper has the following novelties:\n{format_args(parent_debate_node.self_delib[self.id])}\n\n"""
            
        prompt+= f"""You used the following evidence to support your arguments:
{format_evidence(parent_debate_node.evidence[self.id])}
You also have preemptively collected some counter evidence from your own paper based on the opposing author's claimed points of novelty:
{format_preemption(parent_debate_node.preemption[self.id])}Given the above (your initial argument, your evidence, the opposition paper's claimed points of novelty, and your counter evidence), make an argument for a specific reason why {claim}, with respect to the topic, {round_topic['title']}. 

Output your argument in the following JSON format:

{{
    "title": <should be a brief, 10-15 word string where the value is the argument title>,
    "description": <2-3 sentence string explaining the argument>
}}
"""
            
        print(prompt)
        outputs = unidecode(self.model.generate(prompt,
                    sampling_params=sampling_params,
                    use_tqdm=False)[0].outputs[0].text)
        print(f'PRESENTING ARGUMENTS FOR {self.focus} PAPER:\n\n{outputs}')
        log_llm(prompt, outputs)

        return json.loads((outputs))

    def respond_to_argument(self, history, author_type):#round_topic, claim, evidence, counter_claim, counter_evidence):
        """
        Respond to the paper given the current state of debate.
        """
        k=3
        # augmented_topic = "" # TODO: SHIVAM (write a prompt, write the output json format)
        # argument = self.generate_arguments(augmented_topic, evidence)
        logits_processor = JSONLogitsProcessor(schema=argument_list_schema, llm=self.model.llm_engine)
        sampling_params = SamplingParams(max_tokens=1024, logits_processors=[logits_processor])
        
        if author_type=="focus paper":
            
            prompt = f"""Your claims: {history['focus_arg']}, Your evidences: {history['f_evidence']}. Opposition claims: {history['cited_arg']}. Oppositions evidence: {history['c_claim']}."""
        else:
            prompt = f"""Your claims: {history['cited_arg']}, Your evidences: {history['c_claim']}. Opposition claims: {history['focus_arg']}. Oppositions evidence: {history['f_evidence']}."""
        prompt = prompt + f"""In this step you must respond to claims of the opposition given your claims and evidences along with the claims and evidences of the opposition paper. Format the output as a schema: {{"argument_list":
                                                [
                                                    {{
                                                        "title": <should be a sentence-long string where the value is the high-level argument title>,
                                                        "description": <2-5 sentence string explaining the argument>
                                                    }}
                                                ]
                                            }}"""
        # conversation = history.extend(conversation)
        outputs = unidecode(self.model.generate(prompt,
                    sampling_params=sampling_params,
                    use_tqdm=False)[0].outputs[0].text)
        print(f'RESPONDING TO ARGUMENTS{outputs}')
        log_llm(prompt, outputs)
        return json.loads(outputs)
    
    def revise_argument(self, history,author_type):
        """
        Strengthen the final argument at the debate node for a paper.
        """
        k=3
        # augmented_topic = "" # TODO: SHIVAM (write a prompt, write the output json format)
        # argument = self.generate_arguments(augmented_topic, evidence)
        logits_processor = JSONLogitsProcessor(schema=argument_list_schema, llm=self.model.llm_engine)
        sampling_params = SamplingParams(max_tokens=1024, logits_processors=[logits_processor])
       
        if author_type=="focus paper":
            
            prompt = f"""Your claims: {history['focus_arg']}, Your evidences: {history['f_evidence']}, Your response: {history['focus_response']}. Opposition claims: {history['cited_arg']}. Oppositions evidence: {history['c_claim']}. Opposition response: {history['cited_response']}"""
        else:
            prompt = f"""Your claims: {history['cited_arg']}, Your evidences: {history['c_claim']}. Your responses: {history['cited_response']}. Opposition claims: {history['focus_arg']}. Oppositions evidence: {history['f_evidence']}. Opposition response: {history['cited_response']}"""
        prompt = prompt  + f"""In this step you must strengthen your claims given the reponses of the opposition. Format the output as a schema: {{"argument_list":
                                                [
                                                    {{
                                                        "title": <should be a sentence-long string where the value is the high-level argument title>,
                                                        "description": <2-5 sentence string explaining the argument>
                                                    }}
                                                ]
                                            }}"""
        # conversation = history.extend(conversation)
        outputs = unidecode(self.model.generate(prompt,
                    sampling_params=sampling_params,
                    use_tqdm=False)[0].outputs[0].text)
        print(f'REVISING ARGUMENTS {outputs}')
        log_llm(prompt, outputs)
        return json.loads(outputs)
