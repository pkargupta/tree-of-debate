from typing import List

from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
from metric.utils.utils import chunks, split_into_sentences, distinct

from vllm import LLM, SamplingParams
from pydantic import BaseModel, StringConstraints, conlist
from typing_extensions import Annotated
from outlines.serve.vllm import JSONLogitsProcessor
import json
import outlines


class summary_schema(BaseModel):
    claims: conlist(Annotated[str, StringConstraints(strip_whitespace=True)], min_length=1,max_length=10) 


class ClaimExtractor:
    def __init__(
        self,
        author_name,
        model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        device: str = "cuda:0",
        batch_size: int = 5,
    ):
        # self.device = device
        
        # load model from HF
        # self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
        # self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = outlines.models.transformers(model_name)#LLM(model=model_name,tensor_parallel_size=4,max_num_seqs=100,enable_prefix_caching=True)
        self.generator = outlines.generate.json(self.model,summary_schema)
        self.batch_size = batch_size
        self.author_name = author_name
        # self.logits_processor = JSONLogitsProcessor(schema=summary_schema, llm=self.model.llm_engine)
        # self.sampling_params = SamplingParams(max_tokens=3000, temperature=0.4, top_p=0.99,logits_processors=[self.logits_processor])

    def process_batch(self, batch: List[str]) -> List[List[str]]:
        predictions = []
        batches = list(chunks(batch, self.batch_size))
        # prompt = 
        
        for b in tqdm(batches, desc="Extracting claims..."):
            prompts = [f"""Your task is to extract claims by {self.author_name} in a summary and decontextualize those claims from the rest of the summary. Summary: {i}
        Each string should be a context-independent claim by {self.author_name}, representing one atomic fact. For claims that are common to both authors or papers, you must include them in the list. Output the list of claims as using the following JSON schema : {{
    "claims": [item 1, item 2, item 3, ...] <should be list of context independent claims by an author>
}} """ for i in b]
            # opts = self.model.generate(prompts,
            #         sampling_params=self.sampling_params,
            #         use_tqdm=True)
            claims = [self.generator(pro).claims for pro in prompts]
            print(claims)
            # claims = []
            # for ind, i in enumerate(opts):
            #     print()
            #     exit(0)
            #     text = json.loads(i)#.outputs[0].text.strip().lower())
            #     print(text)
            #     exit(0)
            #     claims.append(text['claims'])
            
            
            # tok_input = self.tokenizer.batch_encode_plus(
            #     b, return_tensors="pt", padding=True
            # ).to(self.device)
            # claims = self.model.generate(**tok_input)
            # claims = self.tokenizer.batch_decode(claims, skip_special_tokens=True)
            # claims = [split_into_sentences(c) for c in claims]
            # claims = [distinct(c) for c in claims]
            predictions.extend(claims)
        return predictions
