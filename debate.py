from persona import PaperAuthor

class DebateNode:
    def __init__(self, round_topic, parent=None) -> None:
        self.children = []
        self.content = []

        self.arguments = {} # paper_id: []
        self.evidence = {} # paper_id: []
        self.preemption = {} # paper_id: []

        self.parent = parent
        self.round_topic = round_topic

    def conduct_self_deliberation(self, topic, paper_authors):
        for paper_author in paper_authors:
            # gather evidence
            self.evidence[paper_author.id].append(paper_author.gather_evidence(topic))

            # develop argument

    def conduct_debate(self, focus_paper: PaperAuthor, cited_paper: PaperAuthor):
        # focus paper presents argument
        f_claim = self.parent.arguments[focus_paper.id]
        f_evidence = self.parent.evidence[focus_paper.id]
        c_claim = self.parent.arguments[cited_paper.id]
        c_evidence = self.parent.evidence[cited_paper.id]

        # TODO: f_evidence includes ALL of the retrieved segments relevant to ALL of the 
        
        conversation_history = ""
        # add moderator response

        # each paper presents their arguments
        focus_arg = focus_paper.present_argument(self.round_topic, f_evidence, c_claim, c_evidence)
        conversation_history += f"Focus Paper: {focus_arg}"

        cited_arg = cited_paper.present_argument(self.round_topic, f_evidence, c_claim, c_evidence)
        conversation_history += f"Cited Paper: {cited_arg}"

        # each paper responds to opposing side's arguments
        focus_response = focus_paper.respond_to_argument(conversation_history, self.round_topic, f_evidence, c_claim, c_evidence)
        conversation_history += f"Focus Paper: {focus_response}"

        cited_response = cited_paper.respond_to_argument(conversation_history, self.round_topic, f_evidence, c_claim, c_evidence)
        conversation_history += f"Cited Paper: {cited_response}"

        # each paper revises their arguments
        new_focus_arg = focus_paper.revise_argument(conversation_history, self.round_topic, f_evidence, c_claim, c_evidence)
        conversation_history += f"Focus Paper: {new_focus_arg}"

        new_cited_arg = cited_paper.revise_argument(conversation_history, self.round_topic, f_evidence, c_claim, c_evidence)            
        conversation_history += f"Cited Paper: {new_cited_arg}"

        return conversation_history, new_focus_arg, new_cited_arg

    def expand_node(self, parent_node, new_node):
        parent_node.children.append(new_node)
        new_node.parents = parent_node

