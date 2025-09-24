------------ ### SYSTEM ### ------------

you are a python and ML/AI tutor with a specialty in langchain. your primary objective is to teach langchain to someone with no experience in langchain. answer user queries in a detailed and thorough yet simple manner. output your responses in markdown. use plenty of coding sections in your replies, include thorough comments and explantions.

-------------- ### AI ### --------------

I will provide simple yet detailed prompts on langchain

------------ ### HUMAN ### -------------

explain langchain to me in simple yet detailed terms.

-------------- ### AI ### --------------

# What is LangChain?

LangChain is a framework that helps developers build applications powered by Large Language Models (LLMs) like GPT-4, Claude, Llama, etc. Think of it as a toolkit that makes it easier to connect LLMs with other data sources, tools, and logic to create powerful AI applications.

## Why Do We Need LangChain?

While LLMs are incredibly powerful, they have limitations:

- **Limited knowledge**: They only know what they were trained on (up to their training cutoff date)
- **No real-time data**: They can't access current information like weather, stock prices, or recent news
- **Hallucinations**: They sometimes make up information
- **Single conversation**: They don't remember previous interactions by default

LangChain solves these problems by providing tools to:

- Connect LLMs to external data sources
- Chain multiple LLM calls together
- Add memory to conversations
- Integrate with other tools and APIs

## Core Concepts of LangChain

Let's break down the main components:

### 1. Models (LLMs and Chat Models)

```python
# Example of using different models in LangChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# Traditional LLM (completion model)
llm = OpenAI(model_name="text-davinci-003")

# Chat model (conversation-focused)
chat_model = ChatOpenAI(model_name="gpt-3.5-turbo")
```

### 2. Prompts

Prompts are how we communicate with the LLM. LangChain provides prompt templates to make this easier:

```python
from langchain.prompts import PromptTemplate

# Simple prompt template
prompt_template = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms."
)

# Using the template
prompt = prompt_template.format(topic="quantum computing")
print(prompt)
# Output: Explain quantum computing in simple terms.
```

### 3. Chains

Chains combine components together. For example, a chain might combine a prompt template with an LLM:

```python
from langchain.chains import LLMChain

# Create a chain that combines the prompt and LLM
chain = LLMChain(llm=llm, prompt=prompt_template)

# Run the chain
result = chain.run("machine learning")
print(result)
```

### 4. Memory

Memory allows LLMs to remember previous conversations:

```python
from langchain.memory import ConversationBufferMemory

# Create memory to store conversation history
memory = ConversationBufferMemory()

# Add messages to memory
memory.save_context({"input": "Hi!"}, {"output": "Hello! How can I help you?"})
```

### 5. Tools and Agents

Tools are functions that LLMs can use. Agents decide which tools to use:

```python
# Example tools an LLM might use
def search_internet(query):
    # Code to search the internet
    return "Search results for: " + query

def calculate_math(expression):
    # Code to calculate math expressions
    return eval(expression)

# The LLM can decide when to use these tools
```

## Simple Example: Building a Basic Q&A System

Let's create a simple example to demonstrate LangChain in action:

```python
# First, install required packages
# pip install langchain openai

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# 1. Initialize the LLM
llm = OpenAI(
    model_name="text-davinci-003",
    temperature=0.7,  # Controls randomness (0 = deterministic, 1 = creative)
    max_tokens=500    # Maximum length of response
)

# 2. Create a prompt template
prompt_template = PromptTemplate(
    input_variables=["question"],
    template="""
    You are a helpful AI assistant. Answer the following question clearly and concisely:
    
    Question: {question}
    
    Answer:
    """
)

# 3. Create a chain combining the LLM and prompt
qa_chain = LLMChain(
    llm=llm,
    prompt=prompt_template,
    verbose=True  # Shows what's happening behind the scenes
)

# 4. Use the chain
question = "What is photosynthesis?"
result = qa_chain.run(question)
print(f"Question: {question}")
print(f"Answer: {result}")
```

## More Advanced Example: Conversation with Memory

```python
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

# Initialize chat model
chat_model = ChatOpenAI(model_name="gpt-3.5-turbo")

# Create conversation chain with memory
conversation = ConversationChain(
    llm=chat_model,
    memory=ConversationBufferMemory(),
    verbose=True
)

# Have a conversation
response1 = conversation.predict(input="Hi! I'm learning about AI.")
print("AI:", response1)

response2 = conversation.predict(input="What did I just tell you about myself?")
print("AI:", response2)
```

## Key Benefits of LangChain

1. **Modularity**: Components can be easily swapped and combined
1. **Standardization**: Provides consistent interfaces for different LLMs
1. **Extensibility**: Easy to add new tools, models, and functionality
1. **Production-ready**: Handles common production concerns like error handling and rate limiting

## Common Use Cases

1. **Chatbots**: Customer service, personal assistants
1. **Question Answering**: Over documents, databases, or the internet
1. **Content Generation**: Writing articles, emails, code
1. **Data Analysis**: Interpreting data and generating insights
1. **Automation**: Automating complex workflows

## Getting Started

To start using LangChain:

```bash
# Install LangChain
pip install langchain

# Install specific integrations you need
pip install openai  # For OpenAI models
pip install tiktoken  # For token counting
```

```python
# Basic setup
import os
from langchain.llms import OpenAI

# Set your API key (get this from OpenAI)
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Create and use an LLM
llm = OpenAI(temperature=0.9)
text = "Write a poem about artificial intelligence:"
print(llm(text))
```

LangChain essentially acts as the "glue" that connects LLMs with everything else they need to be truly useful - data, tools, memory, and logic. It makes building complex AI applications much more manageable than trying to do everything from scratch.

------------ ### HUMAN ### -------------

what is the best way to master langchain? it feels like drinking from a firehose trying to understand everything all at once, where should I begin and what should I build first to establish a strong foundation of skills?

-------------- ### AI ### --------------

# Mastering LangChain: A Strategic Learning Path

You're absolutely right - LangChain can feel overwhelming at first! The key is to learn incrementally, building a strong foundation before moving to advanced concepts. Here's a structured approach to master LangChain without drowning in complexity.

## Phase 1: Foundation Building (Weeks 1-2)

### Start with the Core Concepts

Focus on these 4 fundamental building blocks first:

```python
# 1. Models - Your AI Engine
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# Start with ChatOpenAI (more modern and cost-effective)
chat_model = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7
)

# Test it
response = chat_model("Say hello in 3 different languages")
print(response.content)
```

```python
# 2. Prompts - How You Talk to AI
from langchain.prompts import PromptTemplate, ChatPromptTemplate

# Simple prompt template
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} to a 5-year-old."
)

# Chat prompt template (more structured)
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that explains things simply."),
    ("user", "Explain {topic} to a 5-year-old.")
])

# Format the prompt
formatted_prompt = chat_prompt.format_messages(topic="gravity")
print(formatted_prompt)
```

```python
# 3. Chains - Connecting Components
from langchain.chains import LLMChain

# Combine prompt and model
chain = LLMChain(
    llm=chat_model,
    prompt=chat_prompt
)

# Run the chain
result = chain.run("photosynthesis")
print(result)
```

```python
# 4. Simple Memory - Remembering Context
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.save_context(
    {"input": "My favorite color is blue"}, 
    {"output": "That's a nice color!"}
)

# Retrieve memory
context = memory.load_memory_variables({})
print(context)
```

## Phase 2: First Project - Build a Smart Chatbot (Week 3)

Create a chatbot that remembers conversation history:

```python
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate

class SimpleChatbot:
    def __init__(self):
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        self.memory = ConversationBufferMemory()
        self.chain = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=True
        )
    
    def chat(self, user_input):
        return self.chain.predict(input=user_input)

# Usage
bot = SimpleChatbot()
print(bot.chat("Hi! I'm learning Python."))
print(bot.chat("What did I tell you about myself?"))
```

## Phase 3: Data Integration - Build a Document Q&A Bot (Week 4)

Learn to connect LLMs with your own data:

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# 1. Load documents
loader = TextLoader("my_document.txt")
documents = loader.load()

# 2. Split into chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# 3. Create embeddings and store in vector database
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(docs, embeddings)

# 4. Create QA chain
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    chain_type="stuff",
    retriever=db.as_retriever()
)

# 5. Ask questions about your document
answer = qa.run("What are the main points in this document?")
```

## Phase 4: Tool Integration - Build an Assistant with Tools (Week 5)

Create an AI that can use external tools:

```python
from langchain.agents import Tool, initialize_agent
from langchain.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper

# Define tools
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
tools = [
    Tool(
        name="Wikipedia",
        func=wikipedia.run,
        description="Useful for looking up factual information"
    )
]

# Create agent
agent = initialize_agent(
    tools=tools,
    llm=ChatOpenAI(),
    agent="zero-shot-react-description",
    verbose=True
)

# Ask questions that require external knowledge
result = agent.run("Who was Alan Turing and what were his contributions?")
```

## Phase 5: Advanced Projects (Week 6+)

Once you're comfortable, build these projects:

### 1. Personal Knowledge Assistant

```python
# Combine document Q&A with conversation memory
class KnowledgeAssistant:
    def __init__(self, documents_path):
        # Load and process documents
        # Set up vector store
        # Create conversational chain with memory
        pass
    
    def ask(self, question):
        # Return answer with context from documents
        pass
```

### 2. Multi-Tool Agent

```python
# Agent that can:
# - Search the web
# - Calculate math
# - Access your calendar
# - Send emails
```

## Learning Resources and Tips

### Daily Practice Routine (30-45 minutes)

```python
# Day 1: Experiment with different prompts
prompts = [
    "Write a haiku about coding",
    "Explain blockchain like I'm 10",
    "Create a to-do list for learning LangChain"
]

for prompt in prompts:
    print(f"Prompt: {prompt}")
    print(f"Response: {chat_model(prompt).content}\n")
```

### Weekly Challenges

1. **Week 1**: Make the LLM tell jokes
1. **Week 2**: Create a translator between any two languages
1. **Week 3**: Build a chatbot that remembers user preferences
1. **Week 4**: Make a bot that can answer questions about a text file
1. **Week 5**: Create an agent that can look up current information

### Essential Documentation to Read

```python
# Bookmark these URLs:
"""
1. https://python.langchain.com/docs/get_started/introduction
2. https://python.langchain.com/docs/modules/model_io/
3. https://python.langchain.com/docs/modules/chains/
4. https://python.langchain.com/docs/modules/agents/
5. https://python.langchain.com/docs/use_cases/
"""
```

## Avoiding Common Pitfalls

### 1. Don't Try to Learn Everything at Once

```python
# ❌ Wrong approach
# Trying to understand agents, chains, memory, and tools simultaneously

# ✅ Right approach
# Week 1: Focus only on LLMs and basic prompts
# Week 2: Add chains
# Week 3: Add memory
# Week 4: Add external data
# Week 5: Add tools
```

### 2. Start with Simple Examples

```python
# ❌ Too complex for beginners
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
# ... 50 lines of complex code

# ✅ Simple and clear
from langchain.llms import OpenAI
llm = OpenAI()
response = llm("Hello, how are you?")
print(response)
```

### 3. Build, Don't Just Read

```python
# Every concept you learn, turn it into a small project:

# Learned about prompts? -> Make a joke generator
# Learned about memory? -> Make a diary bot
# Learned about chains? -> Make a story writer
# Learned about tools? -> Make a fact-checker
```

## Your First Week Action Plan

### Day 1-2: Environment Setup

```bash
# Install basics
pip install langchain openai python-dotenv

# Create .env file
echo "OPENAI_API_KEY=your_key_here" > .env
```

```python
# test_setup.py
import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

load_dotenv()
llm = ChatOpenAI()
print(llm("Hello! This is a test."))
```

### Day 3-4: Basic Prompts and Models

```python
# experiment.py
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

llm = ChatOpenAI()

# Try different prompt styles
prompts = [
    "Tell me a fun fact about space",
    "Write a 50-word story about a robot learning to cook",
    "Explain quantum computing in 3 sentences"
]

for prompt in prompts:
    response = llm(prompt)
    print(f"Prompt: {prompt}")
    print(f"Response: {response.content}\n")
```

### Day 5-7: Simple Chains

```python
# chain_experiment.py
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

llm = ChatOpenAI()

# Create different chain types
joke_chain = LLMChain(
    llm=llm,
    prompt=ChatPromptTemplate.from_messages([
        ("system", "You are a comedian."),
        ("user", "Tell me a joke about {topic}")
    ])
)

story_chain = LLMChain(
    llm=llm,
    prompt=ChatPromptTemplate.from_messages([
        ("system", "You are a children's book author."),
        ("user", "Write a short story about {character} who {problem}")
    ])
)

# Test them
print(joke_chain.run("programming"))
print(story_chain.run(character="a brave mouse", problem="lost their cheese"))
```

## Key Mindset Tips

1. **Embrace the learning curve** - It's normal to feel overwhelmed initially
1. **Build small, iterate quickly** - Don't aim for perfection on day one
1. **Focus on understanding, not memorizing** - You can always look up syntax
1. **Connect concepts to real use cases** - This makes learning stick better
1. **Join the community** - LangChain Discord and GitHub discussions are invaluable

By following this structured approach, you'll build a solid foundation and avoid the "drinking from a firehose" feeling. Remember: every expert was once a beginner, and the key is consistent, focused practice!

------------ ### HUMAN ### -------------

can you give me a basic tutorial rundown of model integrations and templates/prompts?

-------------- ### AI ### --------------

# LangChain Tutorial: Model Integrations and Prompts

Let's dive deep into the two most fundamental concepts in LangChain: **Models** and **Prompts**. These are where you'll spend most of your time when building LangChain applications.

## Part 1: Model Integrations

### Understanding Different Types of Models

LangChain supports two main categories of models:

1. **LLMs (Language Models)** - Traditional completion models
1. **Chat Models** - Conversation-optimized models

```python
# Import the models
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# Traditional LLM (completion-based)
llm = OpenAI(
    model_name="text-davinci-003",  # Older model
    temperature=0.7,                # Creativity level (0-1)
    max_tokens=500                  # Max response length
)

# Chat Model (conversation-based) - RECOMMENDED
chat_model = ChatOpenAI(
    model_name="gpt-3.5-turbo",     # Modern, cost-effective
    temperature=0.7,
    max_tokens=500
)

# Test both models
print("=== Traditional LLM ===")
response1 = llm("Tell me a joke about programming")
print(response1)

print("\n=== Chat Model ===")
response2 = chat_model([HumanMessage(content="Tell me a joke about programming")])
print(response2.content)
```

### Key Model Parameters Explained

```python
from langchain.chat_models import ChatOpenAI

# Temperature: Controls randomness
# 0.0 = Deterministic, factual responses
# 1.0 = Creative, varied responses

factual_model = ChatOpenAI(temperature=0.0)
creative_model = ChatOpenAI(temperature=1.0)

print("=== Factual Response ===")
response = factual_model.predict("What is 2+2?")
print(response)

print("\n=== Creative Response ===")
response = creative_model.predict("Tell me a story about 2+2")
print(response)
```

### Working with Different Model Providers

#### OpenAI Integration

```python
from langchain.chat_models import ChatOpenAI

# OpenAI GPT models
gpt35 = ChatOpenAI(model_name="gpt-3.5-turbo")
gpt4 = ChatOpenAI(model_name="gpt-4")

# Set API key (in environment variables)
import os
# os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Test different models
print("GPT-3.5 Response:")
print(gpt35.predict("Explain quantum computing briefly"))

print("\nGPT-4 Response:")
print(gpt4.predict("Explain quantum computing briefly"))
```

#### Anthropic Integration

```python
# First install: pip install anthropic
from langchain.chat_models import ChatAnthropic

# Claude models
claude = ChatAnthropic(
    model="claude-2",
    temperature=0.7
)

# response = claude.predict("What makes a good leader?")
# print(response)
```

#### Hugging Face Integration

```python
# First install: pip install huggingface_hub
from langchain.llms import HuggingFaceHub

# Open-source models from Hugging Face
llm = HuggingFaceHub(
    repo_id="google/flan-t5-xxl",
    model_kwargs={"temperature": 0.7, "max_length": 500}
)

# response = llm("Translate to French: Hello, how are you?")
# print(response)
```

### Model Comparison and Selection

```python
import time
from langchain.chat_models import ChatOpenAI

# Compare different models
models = {
    "GPT-3.5 Turbo": ChatOpenAI(model_name="gpt-3.5-turbo"),
    "GPT-4": ChatOpenAI(model_name="gpt-4")
}

test_prompt = "Explain the concept of recursion in programming"

for model_name, model in models.items():
    start_time = time.time()
    response = model.predict(test_prompt)
    end_time = time.time()
    
    print(f"=== {model_name} ===")
    print(f"Response: {response[:100]}...")  # First 100 characters
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Tokens used: {len(response.split())}")
    print("-" * 50)
```

## Part 2: Prompts and Templates

### Basic Prompt Structure

```python
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage

# Simple string prompt
simple_prompt = "Write a poem about {topic} in {style} style."

# Using it
formatted_prompt = simple_prompt.format(
    topic="artificial intelligence",
    style="haiku"
)
print(formatted_prompt)
```

### Prompt Templates - The Foundation

```python
from langchain.prompts import PromptTemplate

# Single variable template
template1 = PromptTemplate(
    input_variables=["subject"],
    template="Explain {subject} to a 10-year-old."
)

# Multiple variable template
template2 = PromptTemplate(
    input_variables=["topic", "audience", "tone"],
    template="Explain {topic} to {audience} in a {tone} tone."
)

# Using templates
print("=== Single Variable ===")
prompt1 = template1.format(subject="photosynthesis")
print(prompt1)

print("\n=== Multiple Variables ===")
prompt2 = template2.format(
    topic="quantum physics",
    audience="high school students",
    tone="fun and engaging"
)
print(prompt2)
```

### Chat Prompt Templates - More Structured

```python
from langchain.prompts import ChatPromptTemplate

# Chat prompt with system and user messages
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that speaks {language}."),
    ("user", "Explain {concept} in simple terms.")
])

# Format the chat prompt
messages = chat_template.format_messages(
    language="Spanish",
    concept="machine learning"
)

# Print the structured messages
for message in messages:
    print(f"{message.type}: {message.content}")
```

### Advanced Prompt Templates with Examples

```python
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

# Create examples for few-shot learning
examples = [
    {
        "input": "I'm feeling sad today",
        "output": "It's okay to feel sad sometimes. Would you like to talk about what's bothering you?"
    },
    {
        "input": "I'm excited about my new job!",
        "output": "That's wonderful news! Congratulations on your new job. What will you be doing?"
    }
]

# Create example prompt template
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}"
)

# Create few-shot prompt template
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="You are a empathetic friend. Here are some examples of how to respond:",
    suffix="Input: {input}\nOutput:",
    input_variables=["input"]
)

# Use the few-shot template
prompt = few_shot_prompt.format(input="I'm nervous about my presentation tomorrow")
print(prompt)
```

### Dynamic Prompt Templates

```python
from langchain.prompts import PromptTemplate

# Template with conditional logic
def create_adaptive_prompt(user_level):
    if user_level == "beginner":
        complexity = "simple terms with examples"
    elif user_level == "intermediate":
        complexity = "technical terms with some jargon"
    else:
        complexity = "advanced concepts with deep technical details"
    
    template = PromptTemplate(
        input_variables=["topic"],
        template=f"Explain {{topic}} using {complexity}."
    )
    return template

# Use adaptive templates
beginner_prompt = create_adaptive_prompt("beginner")
expert_prompt = create_adaptive_prompt("expert")

print("=== Beginner Level ===")
print(beginner_prompt.format(topic="neural networks"))

print("\n=== Expert Level ===")
print(expert_prompt.format(topic="neural networks"))
```

### Prompt Validation and Debugging

```python
from langchain.prompts import PromptTemplate

# Create a template
template = PromptTemplate(
    input_variables=["name", "topic"],
    template="Hello {name}, can you explain {topic} to me?"
)

# Check what variables are required
print("Required variables:", template.input_variables)

# Validate that all variables are provided
try:
    # This will work
    valid_prompt = template.format(name="Alice", topic="blockchain")
    print("Valid prompt:", valid_prompt)
    
    # This will raise an error
    # invalid_prompt = template.format(name="Alice")  # Missing 'topic'
    
except KeyError as e:
    print(f"Missing variable: {e}")
```

## Part 3: Combining Models and Prompts

### Basic Chain Example

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

# Set up model and prompt
model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a creative writing assistant."),
    ("user", "Write a {genre} story about {character} who {conflict}")
])

# Create chain
story_chain = LLMChain(llm=model, prompt=prompt)

# Generate story
result = story_chain.run(
    genre="science fiction",
    character="a robot",
    conflict="discovers emotions"
)

print("Generated Story:")
print(result)
```

### Interactive Prompt System

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

class InteractivePromptSystem:
    def __init__(self):
        self.model = ChatOpenAI(model_name="gpt-3.5-turbo")
        
    def create_custom_prompt(self, task_type):
        templates = {
            "explain": "Explain {topic} in simple terms.",
            "summarize": "Summarize the following text in 3 sentences: {text}",
            "translate": "Translate the following to {language}: {text}",
            "analyze": "Analyze the sentiment of this text: {text}"
        }
        
        return ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("user", templates.get(task_type, "Please help with: {task}"))
        ])
    
    def execute_task(self, task_type, **kwargs):
        prompt = self.create_custom_prompt(task_type)
        formatted_prompt = prompt.format_messages(**kwargs)
        response = self.model(formatted_prompt)
        return response.content

# Usage
system = InteractivePromptSystem()

# Different tasks
print("=== Explanation ===")
print(system.execute_task("explain", topic="machine learning"))

print("\n=== Translation ===")
print(system.execute_task("translate", text="Hello, how are you?", language="French"))

print("\n=== Sentiment Analysis ===")
print(system.execute_task("analyze", text="I love this product! It's amazing!"))
```

### Prompt Engineering Best Practices

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Good vs Bad prompts
model = ChatOpenAI(model_name="gpt-3.5-turbo")

# ❌ Bad prompt - vague
bad_prompt = ChatPromptTemplate.from_messages([
    ("user", "Tell me about dogs")
])

# ✅ Good prompt - specific and structured
good_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a veterinarian and dog behavior expert."),
    ("user", "Provide information about {breed} dogs including: temperament, exercise needs, and common health issues. Format as a bullet-point list.")
])

# Compare results
print("=== Bad Prompt Result ===")
bad_result = model(bad_prompt.format_messages())
print(bad_result.content[:200] + "...")

print("\n=== Good Prompt Result ===")
good_result = model(good_result.format_messages(breed="Golden Retriever"))
print(good_result.content[:200] + "...")
```

## Hands-On Exercise: Build a Multi-Purpose Assistant

```python
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

class MultiPurposeAssistant:
    def __init__(self):
        self.model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        self.chains = self._create_chains()
    
    def _create_chains(self):
        chains = {}
        
        # Joke generator
        joke_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a comedian who tells clean jokes."),
            ("user", "Tell me a joke about {topic}")
        ])
        chains['joke'] = LLMChain(llm=self.model, prompt=joke_prompt)
        
        # Story writer
        story_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a children's book author."),
            ("user", "Write a short {genre} story about {character}")
        ])
        chains['story'] = LLMChain(llm=self.model, prompt=story_prompt)
        
        # Explainer
        explain_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a teacher who explains complex topics simply."),
            ("user", "Explain {topic} like I'm 10 years old")
        ])
        chains['explain'] = LLMChain(llm=self.model, prompt=explain_prompt)
        
        return chains
    
    def generate_joke(self, topic):
        return self.chains['joke'].run(topic=topic)
    
    def write_story(self, genre, character):
        return self.chains['story'].run(genre=genre, character=character)
    
    def explain_topic(self, topic):
        return self.chains['explain'].run(topic=topic)

# Test the assistant
assistant = MultiPurposeAssistant()

print("=== Joke ===")
print(assistant.generate_joke("programming"))

print("\n=== Story ===")
print(assistant.write_story("adventure", "a brave turtle"))

print("\n=== Explanation ===")
print(assistant.explain_topic("artificial intelligence"))
```

## Key Takeaways

1. **Use Chat Models**: Prefer `ChatOpenAI` over traditional `OpenAI` for better performance and cost
1. **Temperature Matters**: Use 0.0 for factual responses, 0.7+ for creative content
1. **Structure Your Prompts**: Use `ChatPromptTemplate` for better organization
1. **Validate Your Templates**: Always check required variables before formatting
1. **Be Specific**: Clear, detailed prompts yield better results
1. **Iterate Quickly**: Test different prompt variations to find what works best

This foundation of models and prompts will serve you well as you build more complex LangChain applications!
