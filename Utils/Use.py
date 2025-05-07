
class UseModel:
    def __init__(self, model):
        self.model = model

    def __enter__(self):
        return self.model

    def __exit__(self, exc_type, exc_value, traceback):
        self.model.close()
        