from paper_details import Paper

class PaperAuthor:
    def __init__(self, model_id, id, paper: Paper, focus) -> None:
        self.model = None # define model
        self.paper_details = paper
        self.focus = focus
        self.id = None

    def gather_evidence(self, topic):
        """
        Use paper chunks to get relevant segments to the topic.
        """
        raise NotImplemented

    def generate_arguments(self, topic, evidence=False, k=2):
        """
        Given topic and evidence, generate k arguments. 
        If the paper is a focus paper, and the debate round is round #1, the topic should be "I am great".
        If the paper is NOT a focus paper, the topic should be the focus paper's arguments.
        """
        raise NotImplemented

    def preempt_arguments(self, counter_claims, counter_evidence):
        """
        gathers evidence for why self is better than paper_b wrt paper_b's arguments/evidences.
        """
        # generate template to combine counter_claims, counter_evidence
        #   Does my paper also include or address a similar claim/idea?
        #   Does my paper propose a better claim/idea to address the problem solved by p_i's claim?

        augmented_topic = ""
        return self.gather_evidence(augmented_topic)
    
    def present_argument(self, round_topic, f_claim, f_evidence, counter_claim, counter_evidence):
        """
        Generate an argument based on your claims and evidences and other paper's claims and evidences.
        """

        prompt = ""
        argument = self.model.generate()
        # parse argument

        return argument

    def respond_to_argument(self, other_argument, claim, evidence, counter_claim, counter_evidence):
        """
        Respond to the paper given the current state of debate.
        """

        augmented_topic = ""
        argument = self.generate_arguments(augmented_topic, evidence)
        return argument
    
    def revise_argument(self, self_argument, other_argument, claim, evidence, counter_claim, counter_evidence):
        """
        Strengthen the final argument at the debate node for a paper.
        """

        debate_template = ""
        self.model.generate(debate_template)