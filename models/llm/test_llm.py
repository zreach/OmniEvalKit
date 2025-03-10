from models.base import ModelWrapper

class MyDummyModel:
    def __init__(self):
        self.config = None
    def to(self, device):
        pass
    def eval(self):
        pass

class TestLLM(ModelWrapper):
    def __init__(self, model_path, model_args, tokenizer_args, **kwargs):
        super().__init__()
        self.model = MyDummyModel()
        self.tokenizer = None
        self.force_use_generate = True

    def generate_text_only(self, conversation, **kwargs):
        return "Test of response 1.2.\nLine 1.\nLine 2."

model_core = TestLLM
