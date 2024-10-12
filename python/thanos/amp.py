import os

class autocast:
    def __enter__(self):
        os.environ["USE_AMP"] = "1"
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        os.environ["USE_AMP"] = "0"
        if exc_type:
            print(f"An exception occurred: {exc_value}")
        return False
