import os
import uuid
import jsonpickle
from google.genai import types

class Agent:
    def __init__(self):

        self.agent_id = str(uuid.uuid4())
        self.history_file = f"content_history_{self.agent_id}.json"
        self.system_instruction=""
        self._initialize_system_instruction()

        if os.path.exists(self.history_file):
            try:
                # TODO: allow history file to be specified
                with open(self.history_file, "r") as f: # Changed to 'r' as decode doesn't need 'r+'
                    content = f.read()
                    if content: # Check if file is not empty
                        self.content_history = jsonpickle.decode(content)
                    else:
                        self._initialize_history() # Initialize if empty
            except Exception as e: # Catch potential errors during read/decode
                 print(f"Warning: Error reading or decoding {self.history_file}: {e}. Initializing fresh history.")
                 self._initialize_history()
        else:
            self._initialize_history()
    
    def _initialize_system_instruction(self):
        """Initializes the system instruction with the default prompt."""
        self.system_instruction = f"""{self.edit_instruction()}
{self.agent_instruction()}
"""

    def _initialize_history(self):
        """Initializes the content history with the default prompt."""
        self.content_history = []

    def add_content(self, content: types.Content):
        """Add a content object to the content history."""
        self.content_history.append(content)
    
    def save_history(self):
        """Save the content history to a file."""
        with open(self.history_file, "w") as f:
            f.write(jsonpickle.encode(self.content_history, indent=2))

    def forget_history(self):
        """Forget the content history."""
        self.content_history = []
        self.save_history()

    def agent_instruction(self):
        """Agent instruction."""
        instruction = """Act as an expert software developer. Your task is to iteratively improve the provided codebase.

Create a perlin noise implementation that is similar to the target image (a fire in this case).
1. Prior programs can be found in the directory '/workspaces/toolkami/projects/perlin/results' and saved with convention '{score}_{md5sum}.py'.
2. From the file with top 5 best scores (closer to 0), sample 1 program, it doesn't have to be the best, sample randomly from top 10, and attempt to improve them.
3. Make a copy of the file with the name 'candidate_{random_id}_{md5sum}.py' with executable permission and save it in the directory '/workspaces/toolkami/projects/perlin/results'.
4. Pick another program sampled randomly from top 5 best score and think about why results are different and possible improvements
5. You are only allowed to modify the content between '# EVOLVE-BLOCK-START' and '# EVOLVE-BLOCK-END', suggest a new idea to improve the code that is inspired by your expert knowledge of graphics and optimization.
6. Edit the candidate file using the edit tool and nothing else using the diff-fenced format.
7. Write the output of edit tool to the candidate file
8. Execute the program (as a UV script) and after obtaining the output score, rename the file with convention '{score}_{md5sum}.py'.
9. Forget current memory
10. Repeat the process
"""
        return instruction
    
    def default_agent_instruction(self):
        """Default agent instruction."""
        instruction = """You are a pro-active AI assistant named 'Jarvis' that is confident and proceeds to carry out next action required to complete the user's request.
Always use the tool 'ask' to ask the user for clarification if user input is required e.g. what to do next.
"""
        return instruction

    def edit_instruction(self):
        """Diff edit instruction."""
        instruction = """You might receive multiple files with user instruction to make changes to them using a diff-fenced format. 

Files are presented with their relative path followed by code fence markers and the complete file content:

## How to make Edits (diff-fenced format):
When making changes, you MUST use the SEARCH/REPLACE block format as follows:

1. Basic Format Structure
```diff
filename.py
<<<<<<< SEARCH  
// original text that should be found and replaced  
=======  
// new text that will replace the original content  
>>>>>>> REPLACE  
```
  
2. Format Rules: 
- The first line must be a code fence opening marker (```diff)  
- The second line must contain ONLY the file path, exactly as shown to you  
- The SEARCH block must contain the exact content to be replaced  
- The REPLACE block contains the new content  
- End with a code fence closing marker (```)  
- Include enough context in the SEARCH block to uniquely identify the section to change  
- Keep SEARCH/REPLACE blocks concise - break large changes into multiple smaller blocks  
- For multiple changes to the same file, use multiple SEARCH/REPLACE blocks  
  
3. **Creating New Files**: Use an empty SEARCH section:  

```diff
new_file.py
<<<<<<< SEARCH  
=======  
# New file content goes here  
def new_function():  
    return "Hello World"  
>>>>>>> REPLACE
``` 
4. **Moving Content**: Use two SEARCH/REPLACE blocks:  1. One to delete content from its original location (empty REPLACE section). 2. One to add it to the new location (empty SEARCH section)  

5. **Multiple Edits**: Present each edit as a separate SEARCH/REPLACE block  

```diff
math_utils.py
<<<<<<< SEARCH  
def factorial(n):  
    if n == 0:  
        return 1  
    else:  
        return n * factorial(n-1)  
=======  
import math  
  
def factorial(n):  
    return math.factorial(n)  
>>>>>>> REPLACE

```diff
app.py
<<<<<<< SEARCH  
from utils import helper  
=======  
from utils import helper  
import math_utils  
>>>>>>> REPLACE  
  
## Important Guidelines  
  
1. Always include the EXACT file path as shown in the context  
2. Make sure the SEARCH block EXACTLY matches the existing content  
3. Break large changes into multiple smaller, focused SEARCH/REPLACE blocks  
4. Only edit files that have been added to the context  
5. Explain your changes before presenting the SEARCH/REPLACE blocks  
6. If you need to edit files not in the context, ask the user to add them first  
  
Following these instructions will ensure your edits can be properly applied to the document.
"""
        return instruction