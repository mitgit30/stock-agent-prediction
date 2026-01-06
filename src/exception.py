class ApiError(Exception):
    "Custom exception class for API errors."
    def __init__(self, message):
        self.message = message
        
class PipelineError(Exception):
    "Custom exception class for pipeline errors."
    def __init__(self, message):
        self.message = message