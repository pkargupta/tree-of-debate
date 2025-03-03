from no_tree.persona import PaperAuthor
from typing import List
from collections import defaultdict
import os

def collect_arguments(arguments):
    text_args = ""
    counter = 1
    for args in arguments:
        for a in args['argument_list']:
            text_args += f"{counter}. {a['argument_title']}. "
    return text_args

def topic_dict_to_str(topic):
    if topic['topic_title'] == topic['topic_description']:
        return topic['topic_title']
    else:
        return f"{topic['topic_title']}: {topic['topic_description']}." # keeping this in case we want to represent topics with the title and description

class DebateNode:
    def __init__(self, round_topic: str, parent=None) -> None:
        self.children = []
        self.content = []

        self.self_delib = {} # paper_id: [] (format for all below)
        self.evidence = {}
        self.preemption = {}
        self.topics = {}
        self.init_arguments = {}
        self.response = {}
        self.final_arguments = {}

        self.parent = parent
        self.round_topic = round_topic
    

    def __repr__(self):
        return topic_dict_to_str(self.round_topic)

    def conduct_self_deliberation(self, topic, paper_authors: List[PaperAuthor], moderator, log=None, num_evidence=5, num_arg=3):
        focus_paper = None
        for paper_author in paper_authors:
            # gather evidence
            evidence, scores = paper_author.gather_evidence(topic_dict_to_str(topic), k=num_evidence, return_scores=True)

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
                        temp += f"Argument #{i+1} - {arg['argument_title']}.\n\t{arg['description']}\n\t{arg['evidence']}\n"
                    f.write(f'{paper_author.focus} paper:\n{temp}\n\n')
        
        # preemption
        ## for each other paper author, collect their respective arguments/evidence
        for i in range(len(paper_authors)):
            other_arguments = []
            for j in range(len(paper_authors)):
                if j != i:
                    other_arguments.extend([a for a in self.self_delib[paper_authors[j].id]])

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

        self.topics = moderator.generate_topics(round=self, parent_topic=topic, paper_authors=paper_authors)

        # for subtopic in self.topics:
        #     self.children.append(DebateNode(subtopic, parent=self))

        
        self.children.append(DebateNode(self.topics, parent=self))
        return self.children
        

    def conduct_debate(self, paper_authors: List[PaperAuthor]):

        convo_history = f"Debate Topic Information:\n\t- Topics: {self.round_topic['topic_title']}\n\t- Topic Description: {self.round_topic['topic_description']}\n\n"

        # each paper presents their arguments
        convo_history = "Debate History:\n\n"
        for author in paper_authors:
            opposition = paper_authors[1-author.id]
            
            print(f"\nPRESENT ARGUMENT FOR AUTHOR {author.id}:\n")
            author_arg = author.present_argument(debate_node=self, parent_debate_node=self.parent, opposition=opposition)
            self.init_arguments[author.id] = author_arg
            convo_history += f"\t-Author {author.id}: I argue that {author_arg['argument_title'].lower()}. {author_arg['description']}\n"

        convo_history += "\n"
        # each paper responds to opposing side's arguments
        for author in paper_authors:
            opposition = paper_authors[1-author.id]
            
            print(f"\nRESPOND ARGUMENT FOR AUTHOR {author.id}:\n")
            author_history = convo_history.replace(f'Author {author.id}:', "You:").replace(f'Author {1-author.id}:', "Opposition:")
            
            author_response = author.respond_to_argument(author_history, debate_node=self, parent_debate_node=self.parent, opposition=opposition)
            self.response[author.id] = author_response
            convo_history += f"\t-Author {author.id}: {author_response['author_response']}\n"

        convo_history += "\n"
        # each paper revises their arguments
        for author in paper_authors:
            opposition = paper_authors[1-author.id]
            
            print(f"\nREVISE ARGUMENT FOR AUTHOR {author.id}:\n")
            author_history = convo_history.replace(f'Author {author.id}:', "You:").replace(f'Author {1-author.id}:', "Opposition:")
            author_revision = author.revise_argument(author_history, debate_node=self, parent_debate_node=self.parent, opposition=opposition)
            self.final_arguments[author.id] = author_revision
            convo_history += f"\t-Author {author.id}: I argue that {author_revision['revised_argument_title'].lower()}. {author_revision['revised_argument_description']}\n"

        return convo_history
    
    def expand_node(self, parent_node, new_node):
        parent_node.children.append(new_node)
        new_node.parents = parent_node

