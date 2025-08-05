class Agent:
    def __init__(self, name, llm_client=None):
        self.name = name
        self.llm_client = llm_client

    def execute(self, *args, **kwargs):
        raise NotImplementedError("This method should be overridden by subclasses.")
