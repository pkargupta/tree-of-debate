from debate import DebateNode
from paper_details import Paper
from persona import PaperAuthor
from moderator import Moderator
from typing import List

focus_paper = PaperAuthor(
    model_id = "Llama 3.1",
    paper = Paper("TreeInstruct is great. TreeInstruct is great. TreeInstruct is great. TreeInstruct is great. TreeInstruct is great. TreeInstruct is great."),
    focus=True,
    id=0
)

cited_paper = PaperAuthor(
    model_id = "Llama 3.1",
    paper = Paper("Knowledge tracing is ok. Knowledge tracing is ok. Knowledge tracing is ok. Knowledge tracing is ok. Knowledge tracing is ok. Knowledge tracing is ok. "),
    focus=False,
    id=1
)

moderator = Moderator("Llama 3.1")

paper_authors = [focus_paper, cited_paper]
leaf_node_label = "Educational Conversations"

# each node has a topic
root_node = DebateNode(leaf_node_label)
subtrees = root_node.conduct_self_deliberation(leaf_node_label, paper_authors) # k new, finer topics to discuss

"""
TI and knowledge tracing
args: (1) tree and (2) sse

root: educational convos
    root_args: tree + sse

level 1: tree || sse (2 nodes)
    tree_args: tree is better than not
    sse_args: sse is good

level 2: tree_is_good | tree_is_bad || sse_with_nl | sse_with_vector

"""

conversation_history = []

queue_of_rounds: List[DebateNode] = []
queue_of_rounds.extend(subtrees)

while len(queue_of_rounds) > 0:
    round = queue_of_rounds.pop(0)
    conversation, new_focus_arg, new_cited_arg = round.conduct_debate(focus_paper, cited_paper)
    conversation_history.extend(conversation)
    if moderator.is_expand(round.arguments, [new_focus_arg, new_cited_arg]):
        new_subtrees = round.conduct_self_deliberation(round.round_topic, paper_authors)
        queue_of_rounds.extend(new_subtrees)

with open('conversation_history.txt', 'w+') as f:
    f.write('\n'.join(conversation_history))