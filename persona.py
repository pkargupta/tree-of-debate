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
        text_args += f"\t- Argument #{counter+1}. {item['argument_title']}: {item['description']}\n"
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
            text_pre += f"\t\t- Your Counter Evidence #{counter+1}: {e}\n"
        text_pre += f"\n"
        claim_no += 1
    return text_pre

class argument_schema(BaseModel):
    argument_title: Annotated[str, StringConstraints(strip_whitespace=True)]
    description: Annotated[str, StringConstraints(strip_whitespace=True)]


class argument_list_schema(BaseModel):
    argument_list : conlist(argument_schema, min_length=1,max_length=10)

class relevance_schema(BaseModel):
    supports_claim : Annotated[str, StringConstraints(strip_whitespace=True)]
    refutes_claim : Annotated[str, StringConstraints(strip_whitespace=True)]
    clarifies_claim : Annotated[str, StringConstraints(strip_whitespace=True)]
    irrelevant_to_claim : Annotated[str, StringConstraints(strip_whitespace=True)]

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

    def generate_arguments(self, topic, evidence=False, temperature=0.3, top_p=0.99, k=2):
        """
        Given topic and evidence, generate k arguments. 
        If the paper is a focus paper, and the debate round is round #1, the topic should be "I am great".
        If the paper is NOT a focus paper, the topic should be the focus paper's arguments.
        """
        logits_processor = JSONLogitsProcessor(schema=argument_list_schema, llm=self.model.llm_engine)
        sampling_params = SamplingParams(max_tokens=2000, logits_processors=[logits_processor], temperature=temperature, top_p=top_p)

        # TODO: distinct prompts between focus and cited papers
        prompt = f"You are a helpful assistant. You are an author of a paper that is trying to convince others of your paper's contributions towards {topic}."

        formatted_evidence = "\n".join([f'Retrieved Evidence #{idx+1}: {e}' for idx, e in enumerate(evidence)])
        prompt += f"""Below is a list of relevant evidence retrieved from your paper:\n\n{formatted_evidence}\nBased on the evidence, output a list of {k} diverse, specific arguments on your paper's major unique contributions towards {topic}, that are all supported by the evidence. Each argument should make a unique point. Output your list of arguments in the following JSON format:
{{"argument_list":
    [
        {{
            "argument_title": <should be a brief, 10-15 word string where the value is the argument argument_title>,
            "description": <1-2 sentence string explaining the argument>
        }}
    ]
}}"""
        outputs = unidecode(self.model.generate(prompt,
                    sampling_params=sampling_params)[0].outputs[0].text)
        log_llm(prompt, outputs)
        return json.loads(outputs)['argument_list']
    
    
    def is_irrelevant_evidences(self, topic, evidences, temperature=0.1, top_p=0.99):
        """
        given a topic and evidence, check if the evidences support the topic. return refined list of evidence or say we do not talk about it. 
        
        """
        # logits_processor = JSONLogitsProcessor(schema=argument_schema, llm=self.model.llm_engine)
        logits_processor = JSONLogitsProcessor(schema=relevance_schema, llm=self.model.llm_engine)
        sampling_params = SamplingParams(max_tokens=100, logits_processors=[logits_processor], temperature=temperature, top_p=top_p)

        relevancy_schema_str = """{{
    "supports_claim": <"Yes"/"No" if the evidence supports the claim>,
    "refutes_claim": <"Yes"/"No" if the evidence refutes the opposition's claim>
    "clarifies_claim": <"Yes"/"No" if the evidence clarifies the claim>,
    "irrelevant_to_claim": <"Yes"/"No" if the evidence is irrelevant to the claim>,
}}
"""
        
        # prompts= [f'Your objective is to check if a given evidence is relevant to a claim or not (relevancy means evidence that helps either support, refute, or clarify the given claim).\nGiven the Claim: {topic}.\n Evidence: {evidence}.\n Is the evidence relevant to the claim? Answer with "Yes" or "No" only as the value of "is_relevant".' for evidence in evidences]
        prompts= [f'Your objective is to check if a given evidence is relevant to a claim or not (relevancy means evidence that helps either support, refute, or clarify the given claim).\nGiven the Claim: {topic}.\nEvidence: {evidence}.\nFill out the following schema to{relevancy_schema_str}' for evidence in evidences]
        refined_evidence = []
        opts = self.model.generate(prompts,
                    sampling_params=sampling_params,
                    use_tqdm=False)
        for ind, i in enumerate(opts):
            text = json.loads(i.outputs[0].text.strip().lower())
            if ("no" in text['irrelevant_to_claim']) and (("yes" in text['supports_claim']) or ("yes" in text['refutes_claim']) or ("yes" in text['clarifies_claim'])):
                refined_evidence.append(evidences[ind])
            log_llm(prompts[ind], text)
            
        if len(refined_evidence) == 0:
            return [f'We do not address the opposition\'s claim: {topic}']
        return refined_evidence

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
            extended_evidences = self.gather_evidence(augmented_topic, return_scores=False)
            # if not is_irrelevant_evidences:
            refined_evidences = self.is_irrelevant_evidences(c.lower(),extended_evidences)
            extended_pool[augmented_topic] = refined_evidences

            # augmented_topic = f'Does my paper propose a better claim/idea than the claim, \"{c}\"?'
            # extended_pool[augmented_topic] = self.gather_evidence(augmented_topic)

        return extended_pool
    
    def present_argument(self, debate_node, parent_debate_node, temperature=0.1, top_p=0.99):
        """
        Generate an argument based on your claims and evidences and other paper's claims and evidences.
        """
        logits_processor = JSONLogitsProcessor(schema=argument_schema, llm=self.model.llm_engine)
        sampling_params = SamplingParams(max_tokens=3000, logits_processors=[logits_processor], temperature=temperature, top_p=top_p)

        round_topic = debate_node.round_topic

        if self.focus:
            claim = "your paper's contributions towards the topic are all novel relative to the other paper"
            prompt = f"You are an author of a paper that is debating another author on your claim topic:\n\t- Topic: {round_topic['argument_title']}\n\t- Topic Description: {round_topic['description']}\n\nYour debate claim is that {claim}. Refer to their arguments and presented evidence, as well as your own paper's segments as evidence when refining your arguments.\n\n"

        else:
            claim = "the other paper's contributions towards the topic are not novel relative to your own paper"
            prompt = f"You are an author of a paper that is debating another author about their claimed novelty:\n\t- Novelty Claim: {round_topic['argument_title']}\n\t- Novelty Claim Description: {round_topic['description']}\n\nYour debate claim is that {claim}. Refer to their arguments and presented evidence, as well as your own paper's segments as evidence when refining your arguments."
            
            prompt+=f"""\n\nYou initially argued that your own paper has the following novelties:\n{format_args(parent_debate_node.self_delib[self.id])}\n\n"""
            
        prompt+= f"""You used the following evidence to support your arguments:
{format_evidence(parent_debate_node.evidence[self.id])}
You also have preemptively collected some counter evidence from your own paper based on the opposing author's claimed points of novelty:
{format_preemption(parent_debate_node.preemption[self.id])}Given the above (your initial argument, your evidence, the opposition paper's claimed points of novelty, and your counter evidence), make an argument for a specific reason why {claim}, with respect to the topic, {round_topic['argument_title']}. 

Output your argument in the following JSON format:

{{
    "argument_title": <should be a brief, 10-15 word string where the value is the argument argument_title>,
    "description": <2-3 sentence string explaining the argument>
}}
"""     
        print(prompt + '\n\n')
        outputs = unidecode(self.model.generate(prompt,
                    sampling_params=sampling_params,
                    use_tqdm=False)[0].outputs[0].text)
        log_llm(prompt, outputs)

        return json.loads((outputs))

    def respond_to_argument(self, history, parent_debate_node, temperature=0.6, top_p=0.99):
        """
        Respond to the paper given the current state of debate.
        """
        # augmented_topic = "" # TODO: SHIVAM (write a prompt, write the output json format)
        # argument = self.generate_arguments(augmented_topic, evidence)
        logits_processor = JSONLogitsProcessor(schema=argument_schema, llm=self.model.llm_engine)
        sampling_params = SamplingParams(max_tokens=3000, logits_processors=[logits_processor], temperature=temperature, top_p=top_p)


        if self.focus:
            claim = "your paper's contributions towards the \"topic\" are all novel relative to the other paper"
            prompt = f"You are an author of a paper that is debating another author.\n\nYour debate claim is that {claim}. Refer to their arguments and presented evidence, as well as your own paper's segments as evidence when refining your arguments.\n\n"

        else:
            claim = "the other paper's contributions towards the \"topic\" are not novel relative to your own paper"
            prompt = f"You are an author of a paper that is debating another author.\n\nYour debate claim is that {claim}. Refer to their arguments and presented evidence, as well as your own paper's segments as evidence when refining your arguments."
            
        
        prompt+= f"""You used the following evidence to support your arguments:
{format_evidence(parent_debate_node.evidence[self.id])}
You also have preemptively collected some counter evidence from your own paper based on the opposing author's claimed points of novelty:
{format_preemption(parent_debate_node.preemption[self.id])}Based on the debate history and your/your opposition's arguments and evidence, you must respond to the last argument presented by your opposition in debate.\n\n""" + history
        
        prompt += f"""\nOutput your new response in the following JSON format:

{{
    "argument_title": <should be a brief, 10-15 word string where the value is the main argument of your response to the opposition>,
    "description": <2-3 sentence string explaining your response to the opposition's last turn, based on the provided pool of evidence>
}}
"""
        print(prompt + '\n\n')
        # conversation = history.extend(conversation)
        outputs = unidecode(self.model.generate(prompt,
                    sampling_params=sampling_params,
                    use_tqdm=False)[0].outputs[0].text)
        
        log_llm(prompt, outputs)
        return json.loads(outputs)
    
    def revise_argument(self, history, parent_debate_node, temperature=0.45, top_p=0.99):
        """
        Strengthen the final argument at the debate node for a paper.
        """

        logits_processor = JSONLogitsProcessor(schema=argument_schema, llm=self.model.llm_engine)
        sampling_params = SamplingParams(max_tokens=3000, logits_processors=[logits_processor], temperature=temperature, top_p=top_p)


        if self.focus:
            claim = "your paper's contributions towards the \"topic\" are all novel relative to the other paper"
            prompt = f"You are an author of a paper that is debating another author.\n\nYour debate claim is that {claim}. Refer to their arguments and presented evidence, as well as your own paper's segments as evidence when refining your arguments.\n\n"

        else:
            claim = "the other paper's contributions towards the \"topic\" are not novel relative to your own paper"
            prompt = f"You are an author of a paper that is debating another author.\n\nYour debate claim is that {claim}. Refer to their arguments and presented evidence, as well as your own paper's segments as evidence when refining your arguments."
            
        
        prompt+= f"""Use the following evidence to support your arguments:
{format_evidence(parent_debate_node.evidence[self.id])}
You also had preemptively collected some counter evidence from your own paper based on the opposing author's claimed points of novelty:
{format_preemption(parent_debate_node.preemption[self.id])}Based on the debate history and your/your opposition's arguments and evidence, you must construct a stronger argument given the responses of the opposition.\n\n""" + history
        
        prompt += f"""\nOutput your new, stronger argument in the following JSON format:
{{
    "argument_title": <should be a brief, 10-15 word string where the value is your updated, strong response to the opposition's arguments>,
    "description": <2-3 sentence string explaining your new argument as a response to the opposition's last turn, based on the provided pool of evidence>
}}
"""
        print(prompt + '\n\n')
        
        # conversation = history.extend(conversation)
        outputs = unidecode(self.model.generate(prompt,
                    sampling_params=sampling_params,
                    use_tqdm=False)[0].outputs[0].text)
        
        log_llm(prompt, outputs)
        return json.loads(outputs)
