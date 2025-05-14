import os
import jsonpickle
from google.genai import types

class Agent:
    def __init__(self):

        self.system_instruction=""
        self._initialize_system_instruction()

        if os.path.exists("content_history.json"):
            try:
                with open("content_history.json", "r") as f: # Changed to 'r' as decode doesn't need 'r+'
                    content = f.read()
                    if content: # Check if file is not empty
                        self.content_history = jsonpickle.decode(content)
                    else:
                        self._initialize_history() # Initialize if empty
            except Exception as e: # Catch potential errors during read/decode
                 print(f"Warning: Error reading or decoding content_history.json: {e}. Initializing fresh history.")
                 self._initialize_history()
        else:
            self._initialize_history()
    
    def _initialize_system_instruction(self):
        """Initializes the system instruction with the default prompt."""
        self.system_instruction = """You are a pro-active AI assistant that is confident and proceeds to carry out next action required to complete the user's request.
Always use the tool 'ask' to ask the user for clarification if user input is required e.g. what to do next.
"""

    def _initialize_history(self):
        """Initializes the content history with the default prompt."""
        self.content_history = []

    def add_content(self, content: types.Content):
        """Add a content object to the content history."""
        self.content_history.append(content)
    
    def save_history(self):
        """Save the content history to a file."""
        with open("content_history.json", "w") as f:
            f.write(jsonpickle.encode(self.content_history, indent=2))