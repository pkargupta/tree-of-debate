
from vllm import LLM, SamplingParams
from pydantic import BaseModel, StringConstraints, conlist
from typing_extensions import Annotated
from outlines.serve.vllm import JSONLogitsProcessor
import json
import argparse
import json


class summary_schema(BaseModel):
    claims: conlist(Annotated[str, StringConstraints(strip_whitespace=True)], min_length=1,max_length=10) 

class factuality(BaseModel):
    explanation: Annotated[str, StringConstraints(strip_whitespace=True)]
    supports: Annotated[str, StringConstraints(strip_whitespace=True)]
    refutes: Annotated[str, StringConstraints(strip_whitespace=True)]
    irrelevant: Annotated[str, StringConstraints(strip_whitespace=True)]



model = LLM(model="meta-llama/Llama-3.1-8B-Instruct",tensor_parallel_size=4,max_num_seqs=100,enable_prefix_caching=True)
logits_processor = JSONLogitsProcessor(schema=summary_schema, llm=model.llm_engine)
sampling_params = SamplingParams(max_tokens=3000, temperature=0.4, top_p=0.99,logits_processors=[logits_processor])

def evaluate(document, summary, author_name):
    logits_processor = JSONLogitsProcessor(schema=summary_schema, llm=model.llm_engine)
    sampling_params = SamplingParams(max_tokens=3000, temperature=0.4, top_p=0.99,logits_processors=[logits_processor])
    prompt_claim_extract = f"""Your task is to extract claims by {author_name} in a summary and decontextualize those claims from the rest of the summary. Summary: {summary}
        Each string should be a context-independent claim by {author_name}, representing one atomic fact. For claims that are common to both authors or papers, you must include them in the list. Output the list of claims as using the following JSON schema : {{
    "claims": [item 1, item 2, item 3, ...] <should be list of context independent claims by an author>
}} """
    
    opts = model.generate(prompt_claim_extract,
                    sampling_params=sampling_params,
                    use_tqdm=True)[0].outputs[0].text
    claims = json.loads(opts)['claims']
    # print(claims)
    # exit(0)
    logits_processor = JSONLogitsProcessor(schema=factuality, llm=model.llm_engine)
    sampling_params = SamplingParams(max_tokens=3000, temperature=0.4, top_p=0.99,logits_processors=[logits_processor])
    prompts = [f"""Given a claim from a paper's summary and the corresponding paper, check if the claim is supported, refuted or is irrelevant to the paper. <{author_name}> Paper: {document} </{author_name}>. Claim: {c}. We are checking claim about {author_name} about the paper's summary, using the corresponding paper, check if the claim is supported, refuted or is irrelevant to the paper. Format the output as the following JSON schema {{
    "supports": <"Yes"/"No" if the paper supports the claim>,
    "refutes": <"Yes"/"No" if the evidence refutes the opposition's claim>
    "irrelevant": <"Yes"/"No" if the evidence is irrelevant to the claim>,
    "explanation": <explanation of the choice>
}} """for c in claims]
    opts = model.generate(prompts,
                    sampling_params=sampling_params,
                    use_tqdm=True)
    num_claims = len(claims)
    num_errors = 0
    op_res = []
    for i in opts:
        res = i.outputs[0].text
        res = json.loads(res)
        op_res.append(res)
        print(res)
        if 'yes' in res['refutes'].lower() or 'yes' in res['irrelevant'].lower():
            num_errors += 1
    # avg_err = num_errors/num_claims
    # print(avg_err)
    
    return num_claims,num_errors, claims, op_res

        



    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", default="/work/nvme/bcaq/shivama2/new_tod/tree-of-debate/baselines/results_prompt_intro.csv")
    parser.add_argument("--model_type", default="prompt_intro_abs")
    # parser.add_argument("--author_name")
    args = parser.parse_args()
    
    with open(args.results_path, 'r') as f:
        data = json.load(f)
    num_claims = 0
    num_errors = 0
    for d in data:
        # author_name = "paper 0"
        try:
            f_document = d['title_focus'].lower()+d['f_abstract'].lower()+d['f_intro'].lower()
            o_document = d['title_opp'].lower()+d['o_abstract'].lower()+d['o_intro'].lower()
            summary = d[args.model_type].lower().replace('author 0','paper 0').replace('author 1','paper 1')
        except:
            print('SOME ENTRIES IN INTRO OR ABSTRACT ARE NULL')
            continue
        f_author = 'paper 0'
        o_author = 'paper 1'

        f_num_claims, f_num_errors, f_claims, f_fact_measure = evaluate(f_document,summary,f_author)
        o_num_claims, o_num_errors, o_claims, o_fact_measure = evaluate(o_document,summary,o_author)
        num_claims +=  f_num_claims+o_num_claims
        num_errors += f_num_errors+o_num_errors
        d['f_claims'] = f_claims
        d['f_fact_measure'] = f_fact_measure
        d['o_claims'] = o_claims
        d['o_fact_measure'] = o_fact_measure
    avg_score = num_errors/num_claims
    print(f'ERR: for {args.model_type} is {avg_score}')
    output_path = f"opp_pap_data_{args.model_type}_factuality.json"
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Data written to {output_path}")
    
    # exit(0)/

    
#     summary = """{""similarities"": [""Both papers utilize Large Language Models (LLMs) for educational purposes"", ""Both papers aim to improve the learning experience for students through conversational interactions"", ""Both papers evaluate their approaches through human evaluations and quantitative analysis""], ""differences"": [""Author 0 focuses on code debugging, while Author 1 focuses on math remediation"", ""Author 0 proposes a novel state space-based planning algorithm for multi-turn Socratic questioning, whereas Author 1 leverages cognitive task analysis to elicit expert thought processes"", ""Author 0 constructs a multi-bug debugging dataset, whereas Author 1 constructs a dataset of real-world tutoring conversations for math remediation""], ""conclusion"": ""While both papers share similarities in utilizing LLMs for educational purposes and evaluating their approaches through human evaluations, they differ significantly in their focus areas, methodologies, and contributions. Author 0's novel state space-based planning algorithm and multi-bug debugging dataset are distinct from Author 1's cognitive task analysis and math remediation focus. Author 0's work is more specialized in code debugging, whereas Author 1's work is more generalizable to various subjects through its expert-guided decision-making model. Overall, both papers contribute to the growing field of AI-powered education, but with unique strengths and applications.""}""" #row['summary'] 
    
#     document = "Instruct, Not Assist: LLM-based Multi-Turn Planning and Hierarchical Questioning for Socratic Code Debugging" +\
#                     """"Title: Instruct, Not Assist: LLM-based Multi-Turn Planning and Hierarchical Questioning for Socratic Code Debugging Abstract: Socratic questioning is an effective teaching strategy, encouraging critical thinking and problem-solving. The conversational capabilities of large language models (LLMs) show great potential for providing scalable, real-time student guidance. However, current LLMs often give away solutions directly, making them ineffective instructors. We tackle this issue in the code debugging domain with TreeInstruct , an Instructor agent guided by a novel state space-based planning algorithm. TreeInstruct asks probing questions to help students independently identify and resolve errors. It estimates a student's conceptual and syntactical knowledge to dynamically construct a question tree based on their responses and current knowledge state, effectively addressing both independent and dependent mistakes concurrently in a multi-turn interaction setting. In addition to using an existing single-bug debugging benchmark, we construct a more challenging multi-bug dataset of 150 coding problems, incorrect solutions, and bug fixes- all carefully constructed and annotated by experts. Extensive evaluation shows TreeInstruct's state-ofthe-art performance on both datasets, proving it to be a more effective instructor than baselines. Furthermore, a real-world case study with five students of varying skill levels further demonstrates TreeInstruct's ability to guide students to debug their code efficiently with minimal turns and highly Socratic questioning.
#  Introduction With the rapidly expanding conversational and reasoning abilities of large language models (LLMs), there has been a substantial rise in demand for exploiting their capabilities within a multitude of educational applications (Kasneci et al., 2023) in order to widen accessibility via personalized feedback. Specifically, several recent works explore

# Figure 1: The Instructor's goal is to generate multi-turn Socratic questions while guiding the Student towards the correct solution.

# <!-- image -->

# the use of LLMs for providing feedback and guidance to students (Wang et al., 2023; Kazemitabaar et al., 2024; Sheese et al., 2024; Lyu et al., 2024). However, LLMs are typically optimized to generate customer-serving, assistant-like responses, which also translates into the types of questions asked. Especially for educational domains, this style of questioning can be suboptimal (Cotton, 1988; Sahamid, 2016; Yang et al., 2005; Wilson, 1987). For instance, if a student is seeking help from an instructor for correcting their mistakes (e.g., debugging their buggy code), we consider two forms of potential responses: assistant-like and instructor-like . As shown in Figure 1, an assistant-like response would not be a successful educational interaction, as it leads to the Assistant directly providing an answer. On the other hand, an Instructor-like response reflects the educational philosophy of Socratic questioning .

# Socratic questioning is a teaching strategy where the Student independently solves their problem by answering guiding questions, instead of being

# given the solution directly (Wilson, 1987). This is a more effective learning strategy because the weight of learning falls on the Student as they must put in effort to answer a question as opposed to solely relying on the model (Cotton, 1988; Kasneci et al., 2023). Therefore, we aim to re-orient an LLM to be an Instructor, not an assistant, by asking Socratic questions that (1) help the Student understand their mistakes, and (2) do not directly provide the answer. To tackle these challenges, we propose TreeInstruct based on the following principles:

# - 1. State space estimation: An Instructor plans its conversation with a Student based on the ""distance"" between their initial answer and the optimal, correct answer within the estimated state space. In other words, it tracks the knowledge state of the Student within this space throughout the Instructor-Student interactions.
# - 2. Tree-based Socratic questioning: An Instructor generates turn-level Socratic questions conditioned on both the Student's current knowledge state and misunderstanding(s), the latter derived from their responses to the Instructor's questions. This step dynamically constructs a Socratic question tree.
# - 3. Adaptive conversation restructuring: An Instructor updates their initial conversation plan based on how the Student is progressing in the conversation, as reflected by updates (or lack thereof) to the Student's knowledge state. This planning can include both questioning and teaching actions.

# While these principles can apply to many educational domains, this paper focuses on code debugging, which presents unique challenges. Realworld code debugging often involves multiple, potentially interdependent conceptual and syntactical bugs. For instance, Figure 1 shows that first resolving the Student's conceptual misunderstanding of recursion in Fibonacci helps them identify their recursive syntactical bug (Figure 1). However, existing work fails to account for such nuances and assumes single-turn feedback (Kazemitabaar et al., 2024; Wang et al., 2023; Lyu et al., 2024). This ignores the sub-steps required for the Student to understand each bug.

# In contrast, TreeInstruct constructs a multi-turn debugging plan ( state representation ), defined as the set of Student misunderstandings and mistakes ( state variables ) to be resolved in order to comprehend and correct their bug(s). We define all

# potential paths to complete these tasks as the state space . We traverse the space using Socratic questions and trace which variables have been resolved, grounded based on the Student's responses.

# While existing LLM-based tutors are effective in fixing the Student's code with high success, they are either prone to directly revealing code answers or cannot be adapted to new Student responses. For example, CodeAid (Kazemitabaar et al., 2024) (specifically, the ""Help Fix Code"" and ""Question from Code"" modules, as these are most similar to our setting) directly provides code or pseudocode 57% of the time, and achieves a mere 55% rate of helpfulness. On the other hand, TreeInstruct exploits the state space to dynamically construct a tree of questions based on (1) incorrect Student responses, or (2) gaps in the Student's knowledge. The sibling and parent-child relationships between questions reflect the manner in which they traverse the state space. Finally, it exploits both the Student's knowledge state and any proposed bug fixes to serve as the dynamic stopping condition. Overall, TreeInstruct takes a more structured approach to multi-turn conversational feedback, as (1) grounding the conversation on the state space representation ensures that all bugs are sufficiently addressed, and (2) constructing a tree based on the Student's current level of understanding allows for more relevant and personalized question generation.

# We summarize our contributions below:

# - · To the best of our knowledge, TreeInstruct is the first work to explore state space estimation and dynamic tree-based questioning for multi-turn Socratic instruction.
# - · We construct a novel multi-bug debugging dataset with 150 expert-annotated, challenging conceptual and syntactical bugs and their fixes.
# - · Extensive experiments on an existing benchmark and our constructed dataset demonstrate that TreeInstruct can be universally applied to both open and closed source-settings. We also showcase that TreeInstruct's strong Socratic questioning abilities widely outperform all baselines through both (1) rigorous quantitative and qualitative expert evaluation (on average, preferred 78.43% of the time; Student fixes code 24.55% more ) and (2) real-world interactions with students of varying coding abilities."""

#     author_name = "paper 0"
#     document = document.lower()
#     summary = summary.lower().replace('author 0','paper 0').replace('author 1','paper 1')

#     evaluate(document,summary,author_name)

    
    # data = pd.read_csv(args.results_path)

