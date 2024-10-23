from debate import DebateNode, DebateTree
from paper_details import Paper
from persona import PaperAuthor

paper_authors = []

# each node has a topic
root_node = DebateNode("leaf_node_label")
subtrees = root_node.conduct_self_deliberation # k new, finer topics to discuss

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

queue_of_rounds = []
queue_of_rounds.extend(subtrees)

while len(queue_of_rounds) > 0:
    subtree = queue_of_rounds.popleft()
    round = DebateNode(subtree)
    is_expand = round.conduct_debate()
    if is_expand:
        subtrees = round.conduct_self_deliberation()
        queue_of_rounds.extend(subtrees)
