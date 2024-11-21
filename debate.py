from persona import PaperAuthor
from typing import List
from collections import defaultdict
from unidecode import unidecode
import os

class DebateNode:
    def __init__(self, round_topic, parent=None) -> None:
        self.children = []
        self.content = []

        self.arguments = {} # paper_id: []
        self.evidence = {} # paper_id: []
        self.preemption = {} # paper_id: []

        self.parent = parent
        self.round_topic = round_topic

    def conduct_self_deliberation(self, topic, paper_authors: List[PaperAuthor], log=None):
        focus_paper = None
        for paper_author in paper_authors:
            # gather evidence
            evidence, scores = paper_author.gather_evidence(topic, k=5, return_scores=True)

            if paper_author.id not in self.evidence.keys(): self.evidence[paper_author.id] = []
            self.evidence[paper_author.id].append(evidence)

            # develop k arguments
            if paper_author.id not in self.arguments.keys(): self.arguments[paper_author.id] = []
            author_args = paper_author.generate_arguments(topic, evidence,k=2)
            self.arguments[paper_author.id].append(author_args)

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
                    for i, arg in enumerate(author_args['argument_list']):
                        temp += f"Argument #{i+1} - {arg['title']}.\n\t{arg['description']}\n"
                    f.write(f'{paper_author.focus} paper:\n{temp}\n\n')
        
        # preemption
        for i in range(len(paper_authors)):
            other_arguments = [self.arguments[paper_authors[j].id] for j in range(len(paper_authors)) if j != i]
            other_evidence = [self.evidence[paper_authors[j].id] for j in range(len(paper_authors)) if j != i]

            preemption = paper_authors[i].preempt_arguments(other_arguments, other_evidence)
            
            # logging
            if log is not None:
                with open(os.path.join(log, 'self_deliberation.txt'), 'a') as f:
                    f.write(f'Preemption:\n\n')
                    f.write(f'{paper_author.focus} paper:\n{preemption}\n\n')

            self.evidence[paper_authors[i].id].append(preemption)         

        for child_topic in self.arguments[focus_paper.id]:
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
        focus_arg = focus_paper.present_argument(self.round_topic, f_claim, f_evidence, c_claim, c_evidence, k=3,author_type='focus paper')
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

        return conversation, new_focus_arg, new_cited_arg
    
    def expand_node(self, parent_node, new_node):
        parent_node.children.append(new_node)
        new_node.parents = parent_node

