from arxiv2text import arxiv_to_text
from unidecode import unidecode
import string
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
    try:
        raw_extracted_text = arxiv_to_text(pdf_url).strip()
    except:
        raise Exception(f"PDF Link INVALID! {pdf_url}")
        
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
    # extracted_text = extracted_text.replace("- ", "") # remove justified text errors that result in half words ("arbi-\ntrary")
    extracted_text = " ".join(extracted_text.split()) # remove unnecessary whitespace in between
    return extracted_text[:extracted_text.find("References")] # only take the text before the references section


def parse_papers_url(focus_url, cited_url):
    focus_text = extract_text(focus_url)
    cited_text = extract_text(cited_url)

    return focus_text, cited_text