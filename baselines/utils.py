from docling.document_converter import DocumentConverter
from docling_core.types.doc import (
    DocItem,
    DocItemLabel,
    DoclingDocument,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
)

import re

def extract_sections_from_markdown(content,section):

    text = extract_section(content, section)
    # Extract Introduction
    # introduction = extract_section(content, "Introduction")
    
    return text

def extract_section(content, section_name):
    # Regular expression to match headers with optional numbering
    pattern = rf"(?:^|\n)(?:#+\s*)(?:\d+\.?\s+)?{section_name}\s*\n(.*?)(?=\n#+\s*(?:\d+\.?\s+)?\w+|\Z)"
    match = re.search(pattern, content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def process_arxiv(link):
    converter = DocumentConverter()
    result = converter.convert(link)
    # result.document.print_element_tree()
    # for item, level in result.document.iterate_items():
    #     if isinstance(item, TextItem):
    #         print(item)
    #         input()
    #         # exit(0)
    return result.document.export_to_markdown() 


