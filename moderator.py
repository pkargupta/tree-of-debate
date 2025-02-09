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
    explanation: Annotated[str, StringConstraints(strip_whitespace=True)]
    progression_of_arguments: Annotated[str, StringConstraints(strip_whitespace=True)]
    meaningful_questions: Annotated[str, StringConstraints(strip_whitespace=True)]
    clear_winner: Annotated[str, StringConstraints(strip_whitespace=True)]

class subtopic_schema(BaseModel):
    topic_title: Annotated[str, StringConstraints(strip_whitespace=True)]
    topic_description: Annotated[str, StringConstraints(strip_whitespace=True)]
    author_0_relevant_contributions: conlist(int, min_length=0,max_length=10)
    author_1_relevant_contributions: conlist(int, min_length=0,max_length=10)
    
    
class subtopic_list_schema(BaseModel):
    subtopic_list : conlist(subtopic_schema, min_length=1, max_length=10)

def arg_dict_to_str(args, arg_type=True):
    arguments = ""
    for i, key in enumerate(args.keys()):
        if arg_type:
            arguments += f"Author {(i)}'s Initial Argument: {args[key]['argument_title']}: {args[key]['description']}\n\n"
        else:
            arguments += f"Author {(i)}'s Revised Argument: {args[key]['revised_argument_title']}: {args[key]['revised_argument_description']}\n\n"

    return arguments.strip()

def format_evidence(list_evi, author, ids=None):
    text_evi = ""
    idx = 1
    for counter, item in enumerate(list_evi):
        if (ids == None) or ((counter + 1) in ids):
            text_evi += f"\t- Author {author.id}'s Supporting Evidence #{idx}. {item}\n"
            idx += 1
    return text_evi

def format_preemption(author, list_pre):
    text_pre = f"\t- Author {1-author.id}'s relevant evidence to potentially counter the novelty of this contribution:\n"
    for e_id, e in enumerate(list_pre):
        text_pre += f"\t\t- Author {1-author.id}'s Counter Evidence #{e_id+1}: The opposition states, \"{e}\"\n"
    return text_pre

def format_self_deliberation(debate_node, paper_authors):
    out = ""
    for author in paper_authors:
        out += f"Author {author.id} Paper Title: {author.paper.title}\n"
        out += f"Author {author.id} Paper Abstract: {author.paper.abstract}\n\n"

        for no, arg in enumerate(debate_node.self_delib[author.id]):
            out += f"Author {author.id} Paper's Contribution #{no+1}: {arg['argument_title']}: {arg['description']}\n"
            out += f"{format_evidence(debate_node.evidence[author.id], author, arg['evidence'])}"
            arg_key = f"{arg['argument_title']}: {arg['description']}"
            out += f"{format_preemption(author, debate_node.preemption[1-author.id][arg_key])}\n"
    
    return out

class Moderator:
    def __init__(self, model, log_dir):
        self.model = model # define model - Llama 3.
        self.log_dir = log_dir

    def generate_topics(self, round: DebateNode, parent_topic, paper_authors, k=5, temperature=0.3, top_p=0.99):
        topic_title = parent_topic['topic_title']
        prompt = f"""You are a fair and balanced moderator of a debate between two authors determining their respective novel contributions towards the following topic:
Topic: {parent_topic['topic_title']}
Topic Description: {parent_topic['topic_title']}

Here are the two papers and their claimed novel contributions with corresponding evidence:

{format_self_deliberation(round, paper_authors)}

Based on each of the author's claimed novelties, evidence, and counter-evidence to each other's arguments, you must determine the most meaningful, diverse set of subtopics within the parent topic, {topic_title}, which best cover the types of contributions each of the papers make. Remember that for each of your selected topics, the papers will be debating which of them makes the better contribution towards the topic. Hence, for each of your subtopics, cite the integer IDs of any relevant contributions from Author 0 (author_0_relevant_contributions) or Author 1 (author_1_relevant_contributions). At least one of these lists should be non-empty. Overall, our goal is to identify how novel Author 0's paper's contributions towards topic {topic_title} are by individually considering their contributions towards your subtopics. 

Output your subtopics (up to {k}) in the following JSON format: 
{{"subtopic_list":
    [
        {{
            "topic_title": <should be a brief, 10-15 word string where the value is the title of your subtopic>,
            "topic_description": <1-2 sentence string explaining the subtopic and what you feel would be most helpful for the papers to debate within the subtopic>,
            "author_0_relevant_contributions": <list of integer IDs citing which contribution(s) from Author 0 would be most relevant to this subtopic; can be empty>,
            "author_1_relevant_contributions": <list of integer IDs citing which contribution(s) from Author 1 would be most relevant to this subtopic; can be empty>
        }},
        ...
    ]
}}

"""
        logits_processor = JSONLogitsProcessor(schema=subtopic_list_schema, llm=self.model.llm_engine)
        sampling_params = SamplingParams(max_tokens=2000, logits_processors=[logits_processor], temperature=temperature, top_p=top_p)
        
        outputs = unidecode(self.model.generate(prompt,
                    sampling_params=sampling_params,
                    use_tqdm=False)[0].outputs[0].text)
        
        log_llm(self.log_dir, prompt, outputs)
        outputs = json.loads(outputs)

        return outputs['subtopic_list']
    
    def is_expand(self, round: DebateNode, history, temperature=0.1, top_p=0.99):
        """
        Based on the previous arguments and the new arguments, determine if any progress has been made.
        """
        prev_args = arg_dict_to_str(round.init_arguments, True)
        new_args = arg_dict_to_str(round.final_arguments, False)
        logits_processor = JSONLogitsProcessor(schema=expansion_schema, llm=self.model.llm_engine)
        sampling_params = SamplingParams(max_tokens=1024, logits_processors=[logits_processor], temperature=temperature, top_p=top_p)

        round_topic = round.round_topic
        
        prompt = f"""You are a moderator faciliating a debate in which two paper are debating who makes the better contribution towards the following topic:\n\t- Topic: {round_topic['topic_title']}\n\t- Topic Description: {round_topic['topic_description']}\n
-------

{history}

-------

Below, you are given the previous set of arguments and the current set of arguments. 

\"previous arguments\":\n{prev_args}

\"current arguments\":\n{new_args}

-------

You must determine whether progress is being made. DO NOT focus on the language being used. Focus on the content of the arguments. Specifically, determine the following (True or False for each):
1. progression_of_arguments: Are these arguments sufficiently different enough to necesitate further debate? Are there new, deeper concepts being discussed between the two sets of arguments? Return "Yes" or "No".
2. meaningful_questions: Within the debate history, each author acknowledges each other's arguments and may ask clarifying questions accordingly. Do you believe that the clarifying questions have not been sufficiently addressed already and would be important to answer through further debate? If there are no questions raised in the debate history by either author, return "No", otherwise "Yes".
3. clear_winner: Do you believe that it is clear that one author has won the debate, and it does not need to be further deconstructured (in order to determine which components within each author's contributions are truly better)?  Return "Yes" or "No".

Output your argument in the following JSON format: 
{{
    "explanation": <2-5 sentence string to explain your reasoning about whether further debate is necessary>,
    "progression_of_arguments": <pick only one of "Yes" or "No" depending on the history, arguments, and your explanation above>,
    "meaningful_questions": <pick only one of "Yes" or "No" depending on the history, arguments, and your explanation above>,
    "clear_winner": <pick only one of "Yes" or "No" depending on the history, arguments, and your explanation above>
}}
"""
        outputs = unidecode(self.model.generate(prompt,
                    sampling_params=sampling_params,
                    use_tqdm=False)[0].outputs[0].text).lower()
        print(f'IS EXPAND {outputs}')
        log_llm(self.log_dir, prompt, outputs)
        outputs = json.loads(outputs)

        return (("yes" in outputs['progression_of_arguments']) or ("yes" in outputs['meaningful_questions'])) and ("no" in outputs['clear_winner'])

    def summarize_debate(self, conversation_history, similarities, differences, temperature=0.4, top_p=0.99):
        similarities = ",".join(similarities)
        differences = ",".join(differences)

        logits_processor = JSONLogitsProcessor(schema=summary_schema, llm=self.model.llm_engine)
        sampling_params = SamplingParams(max_tokens=1024, logits_processors=[logits_processor], temperature=temperature, top_p=top_p)

        prompt = f"""The authors of two papers have debated about the similarities and differences between their papers. Author 0 is the author of the main paper, while Author 1 is the author of the paper being compared to the main paper. Below, you are given the \"conversation_history\" between the authors, and the specific similarities and differences. The similarities and differences are from the point-of-view of Author 0.

\"conversation_history\":\n{conversation_history}

\"similarities\": {similarities}

\"differences\": {differences}

Your task is to write a synthesis of the debate that summarizes the similarities and differences between the papers. Focus more on the differences than the similarities. Format the output as a schema:
    {{
        "summary": <5-10 sentence string to summarize the similarities and differences between the two papers>
    }}
"""
        # conversation = history.extend(conversation)
        outputs = unidecode(self.model.generate(prompt,
                    sampling_params=sampling_params,
                    use_tqdm=False)[0].outputs[0].text)
        log_llm(self.log_dir, prompt, outputs)
        outputs = json.loads(outputs)['summary']
        return outputs

    def summarize_path_debate(self, paper_authors, root_topic, tree, temperature=0.4, top_p=0.99):
        
        logits_processor = JSONLogitsProcessor(schema=summary_schema, llm=self.model.llm_engine)
        sampling_params = SamplingParams(max_tokens=2000, logits_processors=[logits_processor], temperature=temperature, top_p=top_p)

        prompt = f"""Two authors are debating their respective novelties with respect to the following topic:
Topic: {root_topic['topic_title']}

Author 0's paper title is: {paper_authors[0].paper.title}
Author 1's paper title is: {paper_authors[1].paper.title}

Here is a breakdown of their debates in a tree dictionary format. The debate tree can be interpreted as a debate starting from the root node topic and branching out into different child nodes based on different arguments brought up by each author. The debate path ends when either there is no progression in the authors' arguments or an author has clearly won with respect to novelty. At each tree node, we provide the topic, 'description' of the topic, Author 0's corresponding argument (author_0_argument), and Author 1's corresponding argument (author_0_argument) regarding the topic:

{tree}

Based on the debate breakdown, output an approximately paragraph-long synthesis of the debate which summarizes the similarities and differences between the papers. Loosely structure your summary with initially their similarities (which ideas/aspects overlap between the two papers?) to their differences (what makes the papers unique) in novelties strictly based the information discussed within the debate. Focus more on the differences than the similarities. ENSURE that your output summary is specific and detailed-- no high-level, loose claims unsupported by evidence. Write it as if you were an expert on the topic.

Format your output in the following JSON schema:
{{
    "summary": <5-20 sentence string to summarize the similarities and differences between the two papers identified within the debate tree>
}}
"""
        # conversation = history.extend(conversation)
        outputs = unidecode(self.model.generate(prompt,
                    sampling_params=sampling_params,
                    use_tqdm=False)[0].outputs[0].text)
        log_llm(self.log_dir, prompt, outputs)
        outputs = json.loads(outputs)['summary']
        return outputs