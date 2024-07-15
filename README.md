Welcome to the `LLM with Langchain` project! This README will guide you through the steps to set up and run the project, including necessary installations and how to interact with the conversational AI system.

## Overview

This notebook demonstrates how to use the LangChain library with a HuggingFace model to create a conversational AI system. It includes:

- Setting up necessary installations.
- Loading and saving conversation data.
- Creating and using a conversation buffer for contextual memory.

## Installation

First, you need to install the required packages. Run the following command:

```bash
!pip install transformers langchain chainlit huggingface_hub
```

## Setup

### Mount Google Drive

To save and load conversation data, you need to mount your Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Import Necessary Libraries

```python
from langchain import HuggingFaceHub
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from getpass import getpass
import os
import json
import torch
```

### Set Up Hugging Face Hub API Key

Enter your Hugging Face Hub API key:

```python
HUGGING_FACE_HUB_API_KEY = getpass("Enter your Hugging Face Hub API key: ")
os.environ['HF_TOKEN'] = HUGGING_FACE_HUB_API_KEY
```

### Define the LLM and ConversationChain

```python
repo_id = 'google/gamma-2b'
llm = HuggingFaceHub(
    huggingfacehub_api_token=os.environ['HF_TOKEN'],
    repo_id=repo_id,
    model_kwargs={'temperature': 0.8, 'max_length': 200}
)
memory = ConversationBufferMemory()
conversation_buf = ConversationChain(llm=llm, memory=memory)
```

### Check Device

Check if a GPU is available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

## Load and Save Conversation Data

### Load Conversation Data

```python
def load_conversation_data(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    return []
```

### Save Conversation Data

```python
def save_conversation_data(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
```

### User Input and File Path

Ask for the user's name and define the JSON file path:

```python
user_name = input("Enter your name: ")
json_file_path = f"/content/{user_name}.json"
```

### Load and Process Conversation Data

```python
conversation_data = load_conversation_data(json_file_path)
print(conversation_data)

conversation_data_str = "\n".join(entry['conversation'] for entry in conversation_data)
lines = conversation_data_str.split('\n')

for i in range(0, len(lines), 3):
    if i + 1 < len(lines) and lines[i].startswith("Human:") and lines[i + 1].startswith("AI:"):
        input_text = lines[i].split(":", 1)[1].strip()
        output_text = lines[i + 1].split(":", 1)[1].strip()
        inputs = {"content": input_text}
        outputs = {"content": [output_text]}
        memory.save_context(inputs, outputs)

print(memory.buffer)
```

## Example Queries

```python
queries = [
    "I want you to reply to me like a comedian (Be Hilarious) for all input queries.",
    "I am an AI Engineer and my age is 25.",
    "Can you do algebra?"
]

for query in queries:
    print("input:", query)
    conversation_buf.predict(input=query)

print(memory.buffer)
```

## Save Conversation History

### Define ConversationHistory Class

```python
class ConversationHistory:
    def __init__(self):
        self.conversations = []

    def add_conversation(self, conversation_text):
        self.conversations.append({"conversation": conversation_text})

    def save_to_file(self, file_path):
        with open(file_path, 'w') as file:
            json.dump(self.conversations, file, indent=4)
```

### Save History

```python
conversation_history = ConversationHistory()
conversation_history.add_conversation(memory.buffer)
conversation_history.save_to_file(json_file_path)

print(f"Conversation history saved to {json_file_path}")
```

## Interaction

You can interact with the notebook by running the cells in sequence. Ensure you have entered the necessary inputs (Hugging Face API key, user name) when prompted. The conversation history will be saved to your Google Drive, allowing you to continue conversations across sessions.
