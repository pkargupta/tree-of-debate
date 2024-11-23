from persona import PaperAuthor
from typing import List
from collections import defaultdict
import os

def collect_arguments(arguments):
    text_args = ""
    counter = 1
    for args in arguments:
        for a in args['argument_list']:
            text_args += f"{counter}. {a['title']}. "
    return text_args


class DebateNode:
    def __init__(self, round_topic, parent=None) -> None:
        self.children = []
        self.content = []

        self.self_delib = {} # paper_id: [] (format for all below)
        self.evidence = {}
        self.preemption = {}
        self.init_arguments = {}
        self.final_arguments = {}

        self.parent = parent
        self.round_topic = round_topic

    def __repr__(self):
        return self.round_topic['title']

    def conduct_self_deliberation(self, topic, paper_authors: List[PaperAuthor], log=None, num_evidence=5, num_arg=2):
        focus_paper = None
        for paper_author in paper_authors:
            # gather evidence
            evidence, scores = paper_author.gather_evidence(topic, k=num_evidence, return_scores=True)

            if paper_author.id not in self.evidence.keys(): self.evidence[paper_author.id] = []
            self.evidence[paper_author.id].extend(evidence)

            # develop k arguments
            if paper_author.id not in self.self_delib.keys(): self.self_delib[paper_author.id] = []
            author_args = paper_author.generate_arguments(topic, evidence, k=num_arg)
            self.self_delib[paper_author.id].extend(author_args)

            # check if paper is the focus
            if paper_author.focus:
                focus_paper = paper_author

            # logging
            if log is not None:
                with open(os.path.join(log, 'self_deliberation.txt'), 'a') as f:
                    f.write(f'Topic: {topic}\n\n')
                    f.write(f'Gather Evidence:\n\n')
                    temp = "\n".join([f'{s} - {e}' for s, e in zip(scores, evidence)])
                    f.write(f'{paper_author.focus} paper:\n{temp}\n\n')

                    f.write(f'Develop Arguments:\n\n')
                    temp = ""
                    for i, arg in enumerate(author_args):
                        temp += f"Argument #{i+1} - {arg['title']}.\n\t{arg['description']}\n"
                    f.write(f'{paper_author.focus} paper:\n{temp}\n\n')
        
        # preemption
        ## for each other paper author, collect their respective arguments/evidence
        for i in range(len(paper_authors)):
            other_arguments = []
            for j in range(len(paper_authors)):
                if j != i:
                    other_arguments.extend([a['title'] for a in self.self_delib[paper_authors[j].id]])

            if paper_authors[i].id not in self.preemption.keys(): self.preemption[paper_authors[i].id] = {}
            self.preemption[paper_authors[i].id].update(paper_authors[i].preempt_arguments(other_arguments))
            
            # logging
            if log is not None:
                with open(os.path.join(log, 'self_deliberation.txt'), 'a') as f:
                    f.write(f'Preemption:\n\n')
                    temp = ""
                    for key in self.preemption.keys():
                        temp += f"\t{key}\n"
                        for j, p in enumerate(self.preemption[key]):
                            temp += f"\t\tPreemptive Arg #{j+1}: {p}\n"
                        temp += "\n"
                    f.write(f'{paper_authors[i].focus} paper:\n{temp}\n\n')       

        for child_topic in self.self_delib[focus_paper.id]:
            self.children.append(DebateNode(child_topic, parent=self))
        return self.children
        

    def conduct_debate(self, focus_paper: PaperAuthor, cited_paper: PaperAuthor):
        # focus paper presents argument
        f_claim = self.parent.arguments[focus_paper.id]
        f_evidence = self.parent.evidence[focus_paper.id]
        c_claim = self.parent.arguments[cited_paper.id]
        c_evidence = self.parent.evidence[cited_paper.id]

        # TODO: f_evidence includes ALL of the retrieved segments relevant to ALL of the 
        
        conversation = defaultdict()
        # add moderator response
        conversation['f_evidence'] = f_evidence
        conversation['c_claim'] = c_claim
        # each paper presents their arguments
        focus_arg = focus_paper.present_argument(self.round_topic, collect_arguments(f_claim), f_evidence, collect_arguments(c_claim), c_evidence, k=3, author_type='focus paper')
        conversation['focus_arg'] = focus_arg

        cited_arg = cited_paper.present_argument(self.round_topic, f_claim, f_evidence, c_claim, c_evidence, k=3, author_type='opposition paper')
        conversation['cited_arg'] = cited_arg
        # history = focus_arg.extend(cited_arg)

        # each paper responds to opposing side's arguments
        focus_response = focus_paper.respond_to_argument(conversation, author_type='focus paper')#self.round_topic, f_claim, f_evidence, c_claim, c_evidence)
        # history.extend(focus_response)
        conversation['focus_response'] = focus_response

        cited_response = cited_paper.respond_to_argument(conversation, author_type='opposition paper')#self.round_topic, f_claim, f_evidence, c_claim, c_evidence)
        # history.extend(focus_response)
        conversation['cited_response'] = cited_response
        # history.extend(cited_response)
        
        # each paper revises their arguments
        new_focus_arg = focus_paper.revise_argument(conversation, author_type='focus paper')#, self.round_topic, f_claim, f_evidence, c_claim, c_evidence)
        conversation['new_focus_arg'] = new_focus_arg

        new_cited_arg = cited_paper.revise_argument(conversation, author_type='opposition paper')#, self.round_topic, f_claim, f_evidence, c_claim, c_evidence)            
        conversation['new_cited_arg'] = new_cited_arg
        # history.extend(new_focus_arg)
        # history.extend(new_cited_arg)

        conv_list = [f"Round 1: {self.round_topic}"]
        for key in conversation.keys():
            conv_list.append(f"\t{key}: {conversation[key]}\n")

        return conv_list, new_focus_arg, new_cited_arg
    
    def expand_node(self, parent_node, new_node):
        parent_node.children.append(new_node)
        new_node.parents = parent_node

