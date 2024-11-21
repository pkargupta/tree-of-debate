from paper_details import Paper
from vllm import SamplingParams
from outlines.serve.vllm import JSONLogitsProcessor
from unidecode import unidecode
from pydantic import BaseModel, conset, StringConstraints, Field, conlist
from typing_extensions import Annotated
import json


class argument_schema(BaseModel):
    title: Annotated[str, StringConstraints(strip_whitespace=True)]
    description: Annotated[str, StringConstraints(strip_whitespace=True)]
    evidence: conset(str, min_length=1, max_length=10)


class argument_list_schema(BaseModel):
    argument_list : conlist(argument_schema, min_length=1,max_length=10)


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
        sampling_params = SamplingParams(max_tokens=1024, logits_processors=[logits_processor])

        # TODO: distinct prompts between focus and cited papers
        prompt = "You are a helpful assistant. You are an author of a a paper debating about your paper with someone else."
        prompt += f"""Below is a list of evidence:\n {evidence}\nOutput a list of {k} arguments about the topic: {topic}. Format the output as a schema: {{"argument_list":
                                                [
                                                    {{
                                                        "title": <should be a sentence-long string where the value is the high-level argument title>,
                                                        "evidence": <a list of 1-10 segments from the input list of evidence that supports your argument>,
                                                        "description": <2-5 sentences explaining the argument>
                                                    }}
                                                ]
                                            }}"""
        outputs = self.model.generate(prompt,
                    sampling_params=sampling_params)
        print(outputs[0].outputs[0].text)
        return json.loads(unidecode(outputs[0].outputs[0].text))

    def preempt_arguments(self, counter_claims, counter_evidence):
        """
        gathers evidence for why self is better than paper_b wrt paper_b's arguments/evidences.
        """
        # generate template to combine counter_claims, counter_evidence
        #   Does my paper also include or address a similar claim/idea?
        #   Does my paper propose a better claim/idea to address the problem solved by p_i's claim?
        extended_pool = {}

        for c, e in zip(counter_claims, counter_evidence):
            augmented_topic = f'Does my paper also include or address {c}?'
            extended_pool[augmented_topic] = self.gather_evidence(augmented_topic)

            augmented_topic = f'Does my paper propose a better claim/idea than "{c}"?'
            extended_pool[augmented_topic] = self.gather_evidence(augmented_topic)

        return extended_pool
    
    def present_argument(self, round_topic, f_claim, f_evidence, counter_claim, counter_evidence,k,author_type):
        """
        Generate an argument based on your claims and evidences and other paper's claims and evidences.
        """
        logits_processor = JSONLogitsProcessor(schema=argument_list_schema, llm=self.model.llm_engine)
        sampling_params = SamplingParams(max_tokens=1024, logits_processors=[logits_processor])
      
        prompt = f"You are a helpful assistant. You are an author of a {author_type}."
        prompt+=f"""In this step you have your claims and evidences along with the claims and evidences of the opposition paper. Your claims: {f_claim}. Your evidences: {f_evidence}. Opposition claims: {counter_claim} Counter evidence {counter_evidence}. Given the list of yours and the oppositions claims and evidences, refine your claims about {round_topic} and present {k} arguments. Format the output as a schema: {{"argument_list":
                                                [
                                                    {{
                                                        "title": single sentence high level argument title,
                                                        "evidence": list of 1-10 evidences unique to that specific argument,
                                                        "description": 2-5 sentences explaining the argument
                                                    }}
                                                ]
                                            }}"""
        outputs = self.model.generate(prompt,
                    sampling_params=sampling_params,
                    use_tqdm=False)
        print(f'PRESENTING ARGUMENTS {outputs[0].outputs[0].text}')
        return json.loads(unidecode(outputs[0].outputs[0].text))
        # prompt = ""
        # argument = self.model.generate() # TODO: SHIVAM (write a prompt, write the output json format)
        # # parse argument

        # return argument

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
                                                        "title": single sentence high level argument title,
                                                        "evidence": list of 1-10 evidences unique to that specific argument,
                                                        "description": 2-5 sentences explaining the argument
                                                    }}
                                                ]
                                            }}"""
        # conversation = history.extend(conversation)
        outputs = self.model.generate(prompt,
                    sampling_params=sampling_params,
                    use_tqdm=False)
        print(f'RESPONDING TO ARGUMENTS{outputs[0].outputs[0].text}')
        return json.loads(unidecode(outputs[0].outputs[0].text))
    
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
                                                        "title": single sentence high level argument title,
                                                        "evidence": list of 1-10 evidences unique to that specific argument,
                                                        "description": 2-5 sentences explaining the argument
                                                    }}
                                                ]
                                            }}"""
        # conversation = history.extend(conversation)
        outputs = self.model.generate(prompt,
                    sampling_params=sampling_params,
                    use_tqdm=False)
        print(f'REVISING ARGUMENTS {outputs[0].outputs[0].text}')

        return json.loads(unidecode(outputs[0].outputs[0].text))
