from arxiv2text import arxiv_to_text
from docling.document_converter import DocumentConverter
from unidecode import unidecode
import string
import json
import re


import argparse
import os
import csv
import string
import numpy as np
import re
from tqdm import tqdm
import json
from nltk import word_tokenize

def extract_text(pdf_url):
    raw_extracted_text = arxiv_to_text(pdf_url).strip()
    raw_extracted_text = unidecode(raw_extracted_text)

    printable = set(string.printable)
    raw_extracted_text = ''.join(filter(lambda x: x in printable, raw_extracted_text)).replace('\r', '')
    raw_extracted_text = re.sub(r'\/uni\w{4,8}', "", raw_extracted_text)
    raw_extracted_text = raw_extracted_text.split('\n')
    
    extracted_text = [""]
    for text in raw_extracted_text:
        try:
            float(text)
            continue
        except:
            if text == "\n":
                if extracted_text[-1] != "\n":
                    extracted_text.append(text)
            elif len(text) < 4:
                extracted_text[-1] += text
            else:
                extracted_text.append(text)
                

    extracted_text = " ".join(extracted_text).replace('\n', ' ') # remove new lines
    extracted_text = extracted_text.replace("- ", "") # remove justified text errors that result in half words ("arbi-\ntrary")
    extracted_text = " ".join(extracted_text.split()) # remove unnecessary whitespace in between
    return extracted_text[:extracted_text.find("References")] # only take the text before the references section

def parse_papers(focus_paper, cited_paper):
    with open(os.path.join("abstracts", focus_paper + ".json"), 'r') as file:
        focus_data = json.load(file)
    with open(os.path.join("abstracts", cited_paper + ".json"), 'r') as file:
        cited_data = json.load(file)
    
    focus = extract_text(f"https://arxiv.org/pdf/{focus_data['arxiv_key'].replace('_', '.')}")
    cited = extract_text(f"https://arxiv.org/pdf/{cited_data['arxiv_key'].replace('_', '.')}")

    data = []
    data.append({'focus':{'title':unidecode(focus_data['title']), 'abstract':unidecode(focus_data['abstract']), 'introduction': unidecode(focus_data['introduction']), 'full_text':focus},
                 'cited':{'title':unidecode(cited_data['title']), 'abstract':unidecode(cited_data['abstract']), 'introduction': unidecode(cited_data['introduction']), 'full_text':cited}})

    with open('data.json', 'w') as file:
        json.dump(data, file)

def parse_papers_docling(focus_url, cited_url):
    converter = DocumentConverter()
    focus = converter.convert(focus_url).document.export_to_dict()
    cited = converter.convert(cited_url).document.export_to_dict()

    data = []
    data.append({'focus':focus,'cited':cited})

    with open('data.json', 'w') as file:
        json.dump(data, file)