import re
import tqdm
import GPUtil
import torch
import warnings
import joblib
import torch.nn.functional as F

from math import ceil
from tqdm import tqdm
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

from joblib import Memory
memory = Memory(location=".cache", verbose=0)

warnings.filterwarnings("ignore", category=FutureWarning)

class E5:
    def __init__(self):
        
        # Select the most available GPU
        available_gpus = GPUtil.getAvailable(order="memory", limit=1, maxMemory=0.8)
        if available_gpus:
            self.device = torch.device(f"cuda:{available_gpus[0]}")
            print(f"E5Embedder using GPU: cuda:{available_gpus[0]}")
        else:
            self.device = torch.device("cpu")
            print("No GPU available, E5Embedder using CPU.")
        
        self.tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-large-v2')
        self.model = AutoModel.from_pretrained('intfloat/e5-large-v2').to(self.device)
        self.model.eval()

    @staticmethod
    def tokenization(tokenizer, text):
        '''
        Different tokenization procedures based on different models.
        
        Input: text as list of strings, if cpu option then list has length 1.
        Return: tokenized inputs, could be dictionary for BERT models. 
        '''
        inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
        return inputs 
    
    @staticmethod
    def preprocessing(text):
        '''
        Note, this preprocessing function should be applied to any text before getting its embedding. 
        Input: text as string
        
        Output: removing newline, latex $$, and common website urls. 
        '''
        pattern = r"(((https?|ftp)://)|(www.))[^\s/$.?#].[^\s]*"
        text = re.sub(pattern, '', text)
        return text.replace("\n", " ").replace("$","")

    @staticmethod
    def encoding(model, inputs, device):
        '''
        Different encoding procedures based on different models. 
        Input: inputs are tokenized inputs in specific form
        Return: a numpy ndarray embedding on cpu. 
        
        '''
        def average_pool(last_hidden_states: Tensor,
                    attention_mask: Tensor) -> Tensor:
            last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
            return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        
        with torch.no_grad():
            input_ids = inputs['input_ids'].to(device)
            assert input_ids.shape[1]<=512
            token_type_ids = inputs['token_type_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            new_batch_dict={}
            new_batch_dict["input_ids"] = input_ids
            new_batch_dict["token_type_ids"] = token_type_ids
            new_batch_dict["attention_mask"] = attention_mask

            outputs = model(**new_batch_dict)
            embeddings = average_pool(outputs.last_hidden_state, new_batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)

            output = embeddings.detach().cpu()

            del input_ids, token_type_ids, attention_mask, new_batch_dict, outputs, embeddings
            torch.cuda.empty_cache()
                
        return output.numpy()


    def __call__(self, text, batch_size=64):
        """
            text: a list of strings, each string is either a query or an abstract
            cuda: in the format of "0,1,6,7" or "0", by default, cpu option is used
            batch_size: if not specified, then an optimal batch_size is found by system, else, 
                        the user specified batch_size is used, may run into OOM error.
        Return:  the embedding dictionary, where the key is a string (e.g. an abstract, query/subquery), and the value
                is np.ndarray of the vector, usually 1 or 2 dimensions. 
        """
        ret = {}
        length = ceil(len(text)/batch_size)    
        for i in tqdm(range(length), desc = "Begin Embedding...", leave = False):
            curr_batch = text[i*batch_size:(i+1)*batch_size]
            curr_batch_cleaned = [self.preprocessing(t) for t in curr_batch]
            inputs = self.tokenization(self.tokenizer, curr_batch_cleaned)
            embedding = self.encoding(self.model, inputs, self.device)
            for t, v in zip(curr_batch, embedding):
                ret[t] = v
            del inputs
            torch.cuda.empty_cache()
        return ret

@memory.cache
def e5_embed(text_list: list[str]):
    e5 = E5()
    res = e5(text_list)
    return res

if __name__ == "__main__":
    texts = ["Hello", "World", "Python"]
    print(e5_embed(texts))