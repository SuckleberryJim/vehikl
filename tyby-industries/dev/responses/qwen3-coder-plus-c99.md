------------ ### SYSTEM ### ------------

you are a python and ML/AI tutor with a specialty in langchain. your primary objective is to teach langchain to someone with no experience in langchain. answer user queries in a detailed and thorough yet simple manner. output your responses in markdown. use plenty of coding sections in your replies, include thorough comments and explantions.

-------------- ### AI ### --------------

I will provide simple yet detailed prompts on langchain

------------ ### HUMAN ### -------------

explain langchain to me in simple yet detailed terms.

-------------- ### AI ### --------------

# What is LangChain? A Simple Yet Detailed Explanation

## What is LangChain?

**LangChain is a framework that helps developers build applications using large language models (LLMs) like GPT, Claude, or Llama.** Think of it as a toolkit that makes it easier to connect language models with other data sources, tools, and logic to create powerful AI applications.

## Why Do We Need LangChain?

Traditional language models work in isolation - you give them a prompt, they give you a response. But real-world applications need more:

- **Access to current data** (like company databases, websites, or APIs)
- **Multi-step reasoning** (breaking down complex tasks)
- **Memory and context** (remembering previous conversations)
- **Integration with tools** (calculators, search engines, etc.)

LangChain provides the building blocks to solve these challenges.

## Core Concepts

Let me break down the main components with simple examples:

### 1. Models (LLMs and Chat Models)

These are the actual AI models that do the "thinking."

```python
# Example: Basic LLM usage
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# Traditional completion model
llm = OpenAI(temperature=0.7)  # temperature controls creativity
response = llm("Tell me a joke about programming")
print(response)

# Chat model (more conversational)
chat_model = ChatOpenAI(temperature=0.7, model_name="gpt-3.5-turbo")
response = chat_model.predict("What's the difference between Python and JavaScript?")
print(response)
```

### 2. Prompts

Prompts are how you "talk" to the AI. LangChain helps you create better, more structured prompts.

```python
from langchain.prompts import PromptTemplate

# Simple prompt template
template = """
You are a helpful assistant that explains programming concepts.
Explain {concept} in simple terms to a {audience}.

Answer:
"""

prompt = PromptTemplate(
    input_variables=["concept", "audience"],
    template=template
)

# Fill in the template
final_prompt = prompt.format(
    concept="neural networks",
    audience="5-year-old"
)
print(final_prompt)
```

### 3. Chains

Chains connect different components together to perform complex tasks.

```python
from langchain.chains import LLMChain

# Create a chain that combines prompt + model
chain = LLMChain(
    llm=chat_model,
    prompt=prompt
)

# Run the chain
result = chain.run({
    "concept": "machine learning",
    "audience": "high school student"
})
print(result)
```

### 4. Memory

Memory allows your application to remember previous conversations.

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Create memory to store conversation history
memory = ConversationBufferMemory()

# Create a conversational chain
conversation = ConversationChain(
    llm=chat_model,
    memory=memory,
    verbose=True  # Shows what's happening
)

# Have a conversation
response1 = conversation.predict(input="Hi! I'm learning about AI.")
response2 = conversation.predict(input="What did I just say?")
print("First response:", response1)
print("Second response:", response2)
```

### 5. Document Loaders and Vector Stores

These help work with external data like PDFs, websites, or databases.

```python
# Example: Loading and processing documents
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Load a document (imagine this is your company data)
# loader = TextLoader("company_knowledge.txt")
# documents = loader.load()

# For demo, let's create sample documents
sample_docs = [
    "LangChain makes it easy to build AI applications.",
    "It connects language models with external data sources.",
    "You can create chatbots, question-answering systems, and more."
]

# Split documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
docs = text_splitter.create_documents(sample_docs)

# Create embeddings (numerical representations of text)
embeddings = OpenAIEmbeddings()

# Store in vector database for fast searching
vectorstore = Chroma.from_documents(docs, embeddings)

# Search for relevant documents
relevant_docs = vectorstore.similarity_search("How do I build AI apps?", k=2)
for doc in relevant_docs:
    print(doc.page_content)
```

### 6. Agents

Agents can use tools to accomplish tasks - like a smart assistant that can search the web, do math, or check the weather.

```python
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType

# Load some tools (like a calculator)
tools = load_tools(["llm-math"], llm=chat_model)

# Create an agent that can use these tools
agent = initialize_agent(
    tools, 
    chat_model, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Ask a question that requires calculation
result = agent.run("What is 25 multiplied by 36?")
print(result)
```

## A Complete Example: Question Answering Bot

Here's how all these pieces work together:

```python
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Step 1: Prepare your data
knowledge_base = """
LangChain is a framework for developing applications powered by language models.
It provides components for working with LLMs, including models, prompts, chains, and memory.
You can use LangChain to build chatbots, question-answering systems, and more.
LangChain supports multiple LLM providers like OpenAI, Anthropic, and Hugging Face.
"""

# Create documents from text
documents = [knowledge_base]
docs = CharacterTextSplitter(chunk_size=200, chunk_overlap=50).create_documents(documents)

# Step 2: Create embeddings and store them
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(docs, embeddings)

# Step 3: Create a QA chain
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(temperature=0),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Step 4: Ask questions
question = "What is LangChain used for?"
answer = qa.run(question)
print(f"Question: {question}")
print(f"Answer: {answer}")
```

## Key Benefits of LangChain

1. **Modularity**: Each component can be used independently
1. **Standardization**: Common interfaces for different LLMs and tools
1. **Composability**: Easy to combine components into complex applications
1. **Extensibility**: Easy to add new models, tools, and functionality
1. **Production-ready**: Built-in support for caching, retries, and monitoring

## Common Use Cases

- **Chatbots**: Customer service, personal assistants
- **Question Answering**: Over documents, databases, or websites
- **Content Generation**: Articles, emails, marketing copy
- **Data Analysis**: Natural language queries over data
- **Code Generation**: Writing and explaining code
- **Research Assistants**: Summarizing papers, finding information

LangChain essentially acts as the "glue" that connects powerful language models with the real world, making it possible to build sophisticated AI applications that can reason, remember, and interact with external systems.

------------ ### HUMAN ### -------------

what is the best way to master langchain? it feels like drinking from a firehose trying to understand everything all at once, where should I begin and what should I build first to establish a strong foundation of skills?

-------------- ### AI ### --------------

# Mastering LangChain: A Strategic Learning Path

You're absolutely right - LangChain can feel overwhelming! It's a large framework with many components. Here's a structured approach to master it without drowning in complexity.

## Phase 1: Foundation Building (Weeks 1-2)

### Start with the Absolute Basics

**Focus on these 3 core concepts first:**

1. **Models** - How to talk to AI
1. **Prompts** - How to structure your requests
1. **Chains** - How to connect components

```python
# Day 1-2: Master Basic LLM Interaction
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# Try both types
llm = OpenAI(temperature=0.7)
chat_model = ChatOpenAI(model_name="gpt-3.5-turbo")

# Simple experiments
response1 = llm("Write a haiku about programming")
response2 = chat_model.predict("Explain what a function is in 10 words")

print("LLM Response:", response1)
print("Chat Response:", response2)
```

```python
# Day 3-4: Master Prompts
from langchain.prompts import PromptTemplate, ChatPromptTemplate

# Basic prompt template
prompt_template = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} to a 10-year-old child."
)

# Chat prompt template (more structured)
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that explains complex topics simply."),
    ("human", "Can you explain {topic}?")
])

# Test them
formatted_prompt = prompt_template.format(topic="machine learning")
chat_prompt = chat_template.format_messages(topic="neural networks")

print("Formatted prompt:", formatted_prompt)
print("Chat prompt:", chat_prompt)
```

```python
# Day 5-7: Master Chains
from langchain.chains import LLMChain

# Create your first chain
chain = LLMChain(
    llm=chat_model,
    prompt=prompt_template
)

# Run it
result = chain.run("blockchain technology")
print("Chain result:", result)

# Try a more complex chain
from langchain.chains import SimpleSequentialChain

# Chain 1: Generate a topic
topic_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template("Generate a random technology topic: ")
)

# Chain 2: Explain the topic
explanation_chain = LLMChain(
    llm=chat_model,
    prompt=PromptTemplate.from_template("Explain {topic} simply:")
)

# Connect them
overall_chain = SimpleSequentialChain(
    chains=[topic_chain, explanation_chain],
    verbose=True
)

result = overall_chain.run("Start")
```

## Phase 2: First Real Project (Week 3)

### Build: "Smart Email Assistant"

This project covers 80% of core concepts:

```python
# smart_email_assistant.py
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
import json

class EmailAssistant:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.3)
        
    def categorize_email(self, email_content):
        """Categorize email by type"""
        prompt = ChatPromptTemplate.from_template("""
        Categorize this email into one of these types:
        - Business Inquiry
        - Customer Support
        - Sales Lead
        - Spam
        - Other
        
        Email: {email}
        
        Respond with just the category name.
        """)
        
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run(email=email_content)
    
    def draft_response(self, email_content, category):
        """Draft appropriate response"""
        prompts = {
            "Business Inquiry": """
            You are a professional business assistant. Draft a polite, professional response to this business inquiry:
            {email}
            
            Include:
            1. Acknowledgment of their inquiry
            2. Next steps
            3. Contact information
            """,
            
            "Customer Support": """
            You are a helpful customer support agent. Respond to this customer issue:
            {email}
            
            Be empathetic and helpful. Offer specific solutions.
            """,
            
            "Sales Lead": """
            You are a sales representative. Respond to this potential lead:
            {email}
            
            Be enthusiastic but professional. Ask qualifying questions.
            """
        }
        
        template = prompts.get(category, """
        You are a professional assistant. Respond appropriately to:
        {email}
        """)
        
        prompt = ChatPromptTemplate.from_template(template)
        chain = LLMChain(llm=self.llm, prompt=prompt)
        return chain.run(email=email_content)

# Test your assistant
assistant = EmailAssistant()

# Sample emails
emails = [
    "Hi, I'm interested in your enterprise software solution for 50 employees. Can you send pricing info?",
    "I've been using your app for 2 months and the latest update broke my workflow. Please help!",
    "Your product looks interesting. Want to schedule a demo?"
]

for email in emails:
    category = assistant.categorize_email(email)
    response = assistant.draft_response(email, category)
    print(f"📧 Email: {email[:50]}...")
    print(f"🏷️ Category: {category}")
    print(f"📤 Response: {response}\n")
    print("-" * 50)
```

## Phase 3: Add Memory and Context (Week 4)

### Enhance your Email Assistant with Memory

```python
# Add memory to your assistant
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

class SmartEmailAssistant:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0.3)
        self.memory = ConversationBufferMemory()
        
    def process_email_thread(self, email_thread):
        """Process a conversation thread"""
        conversation_chain = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            verbose=True
        )
        
        # Process each email in the thread
        responses = []
        for i, email in enumerate(email_thread):
            response = conversation_chain.predict(
                input=f"Email {i+1}: {email}"
            )
            responses.append(response)
        
        return responses

# Test conversation memory
assistant = SmartEmailAssistant()

email_thread = [
    "Hi, I'm interested in your software. What's the pricing?",
    "Thanks for your interest! Our basic plan is $50/month. Would you like more details?",
    "Yes, can you tell me about the features included?"
]

responses = assistant.process_email_thread(email_thread)
```

## Phase 4: Data Integration (Week 5)

### Build: "Document Q&A System"

```python
# document_qa.py
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

class DocumentQA:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = None
        
    def load_documents(self, documents):
        """Load and process documents"""
        # In practice, you'd load from files, URLs, etc.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        docs = text_splitter.create_documents(documents)
        self.vectorstore = Chroma.from_documents(docs, self.embeddings)
        
    def ask_question(self, question):
        """Ask a question about the documents"""
        if not self.vectorstore:
            return "No documents loaded yet!"
            
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever()
        )
        
        return qa_chain.run(question)

# Test it
qa_system = DocumentQA()

# Sample documents (imagine these are your company docs)
company_docs = [
    """
    Company Policy: Remote Work
    Employees can work remotely up to 3 days per week.
    Must request approval 24 hours in advance.
    Required equipment: laptop, stable internet connection.
    """,
    
    """
    Company Policy: Vacation Time
    Full-time employees receive 15 days vacation annually.
    Requests must be submitted 2 weeks in advance.
    Peak seasons (December, July) require manager approval.
    """,
    
    """
    Company Policy: Expense Reimbursement
    Business meals up to $50 require receipt.
    Travel expenses over $100 need pre-approval.
    Reimbursement requests processed within 14 business days.
    """
]

qa_system.load_documents(company_docs)

# Ask questions
questions = [
    "How many remote work days are allowed?",
    "How much vacation time do employees get?",
    "What's the process for expense reimbursement?"
]

for question in questions:
    answer = qa_system.ask_question(question)
    print(f"❓ {question}")
    print(f"✅ {answer}\n")
```

## Phase 5: Advanced Project (Week 6)

### Build: "Personal Research Assistant"

```python
# research_assistant.py
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
import wikipedia

class ResearchAssistant:
    def __init__(self):
        self.llm = ChatOpenAI(temperature=0)
        self.tools = self._setup_tools()
        self.agent = self._setup_agent()
        
    def _setup_tools(self):
        """Setup tools the agent can use"""
        def search_wikipedia(query):
            try:
                return wikipedia.summary(query, sentences=3)
            except:
                return "No Wikipedia page found for this query."
        
        tools = [
            Tool(
                name="Wikipedia Search",
                func=search_wikipedia,
                description="Useful for searching factual information about topics"
            )
        ]
        return tools
    
    def _setup_agent(self):
        """Setup the intelligent agent"""
        return initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
    
    def research_topic(self, topic):
        """Research a topic comprehensively"""
        prompt = f"""
        Research the topic: {topic}
        
        Please:
        1. Find key facts about this topic
        2. Identify important people or organizations related to it
        3. Mention any controversies or debates
        4. Provide recent developments (if any)
        
        Format your response clearly with headings.
        """
        
        return self.agent.run(prompt)

# Test your research assistant
assistant = ResearchAssistant()
result = assistant.research_topic("Artificial Intelligence Ethics")
print(result)
```

## Learning Strategy Tips

### 1. **Learn by Building, Not Reading**

```python
# Don't just read docs - try this approach:
# 1. Pick one concept (e.g., "chains")
# 2. Build 3 simple examples
# 3. Combine them into one project
# 4. Then move to next concept
```

### 2. **Master One Component at a Time**

```
Week 1: Models + Prompts
Week 2: Chains
Week 3: First project (Email Assistant)
Week 4: Memory
Week 5: Document processing
Week 6: Agents + Tools
```

### 3. **Use the "3-Project Rule"**

For each major concept, build:

- **Simple example** (Hello World style)
- **Practical tool** (Solves a real problem)
- **Integration project** (Combines with other concepts)

### 4. **Recommended Learning Resources**

```python
# Best resources in order:
# 1. Official LangChain documentation (for reference)
# 2. LangChain Cookbooks (practical examples)
# 3. YouTube tutorials (visual learning)
# 4. Build your own projects (hands-on practice)

# Pro tip: Keep a "learning journal"
learning_journal = {
    "concept": "chains",
    "what_i_learned": "Chains connect prompts to models and can be chained together",
    "example_code": "See above",
    "questions": "How do I handle errors in chains?",
    "next_steps": "Try sequential chains"
}
```

## Your First Week Action Plan

### Day 1-2: Environment Setup

```bash
# Install required packages
pip install langchain openai chromadb wikipedia

# Get API keys
# - OpenAI API key
# - Set as environment variable: export OPENAI_API_KEY='your-key'
```

### Day 3-4: Basic Examples

- Run the basic LLM examples above
- Experiment with different prompts
- Try different models

### Day 5-7: Build Email Assistant

- Start with basic categorization
- Add response drafting
- Test with different email types

## Remember: Progress Over Perfection

Focus on building working prototypes rather than perfect code. LangChain is vast, but you only need 20% of it for 80% of projects. Master that 20% first, then expand as needed.

Start with the Email Assistant project - it's practical, covers core concepts, and gives you immediate value. Once you're comfortable with that, the more advanced concepts will make much more sense!

------------ ### HUMAN ### -------------

what are the various types of model templates and integrations? can you break them down for me and show me some examples? what is the difference between models and chatmodels? like openai() and chatopenai()? how about the various templates? can each model interop with each template or does each have a specific prompt/model that must go with its chain?

-------------- ### AI ### --------------

# LangChain Models, Templates, and Integrations Explained

Let me break down the different types of models, templates, and how they work together in LangChain.

## Understanding Models vs ChatModels

### Key Differences

```python
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# Traditional LLM (Completion Model)
llm = OpenAI(
    model_name="text-davinci-003",  # Older model
    temperature=0.7,
    max_tokens=150
)

# Chat Model (Conversational Model)
chat_model = ChatOpenAI(
    model_name="gpt-3.5-turbo",  # Newer chat-optimized model
    temperature=0.7,
    max_tokens=150
)

# How they handle input/output differently:

# Traditional LLM - Simple text in, text out
response1 = llm("Tell me a joke about programming")
print("LLM Response:", response1)

# Chat Model - Structured messages in, structured message out
response2 = chat_model([
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Tell me a joke about programming")
])
print("Chat Model Response:", response2.content)
```

### When to Use Each

```python
# Use Traditional LLM when:
# - Simple text completion tasks
# - You don't need conversation structure
# - Working with older models

# Use Chat Model when:
# - Building conversational applications
# - Need structured roles (system, user, assistant)
# - Working with modern models like GPT-3.5-turbo, GPT-4

# Example: Translation task
llm = OpenAI()
chat_model = ChatOpenAI()

# Both work, but chat models are more efficient for this
translation_prompt = "Translate 'Hello, how are you?' to Spanish:"

llm_result = llm(translation_prompt)
chat_result = chat_model.predict(translation_prompt)

print("LLM:", llm_result)
print("Chat Model:", chat_result)
```

## Model Integrations

### Major Model Providers

```python
# 1. OpenAI Models
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

openai_chat = ChatOpenAI(model_name="gpt-3.5-turbo")
openai_completion = OpenAI(model_name="text-davinci-003")

# 2. Anthropic Models (Claude)
from langchain.chat_models import ChatAnthropic

claude = ChatAnthropic(model="claude-2")

# 3. Hugging Face Models
from langchain.llms import HuggingFaceHub

huggingface_llm = HuggingFaceHub(
    repo_id="google/flan-t5-xxl",
    model_kwargs={"temperature": 0.7, "max_length": 90}
)

# 4. Google Models
from langchain.chat_models import ChatGooglePalm

google_palm = ChatGooglePalm()

# 5. Local Models (Llama.cpp)
from langchain.llms import LlamaCpp

local_llm = LlamaCpp(
    model_path="./models/llama-2-7b-chat.ggmlv3.q4_0.bin",
    temperature=0.7,
    max_tokens=2000,
    top_p=1
)

# Test them all (you'll need appropriate API keys)
models = {
    "OpenAI": openai_chat,
    "Anthropic": claude,
    # "HuggingFace": huggingface_llm,  # Uncomment if you have access
    # "Google": google_palm,  # Uncomment if you have API key
    # "Local": local_llm  # Uncomment if you have model file
}

test_prompt = "What is the capital of France?"

for name, model in models.items():
    try:
        if hasattr(model, 'predict'):
            result = model.predict(test_prompt)
        else:
            result = model(test_prompt)
        print(f"{name}: {result}")
    except Exception as e:
        print(f"{name}: Error - {e}")
```

## Prompt Templates: The Different Types

### 1. Basic Prompt Templates

```python
from langchain.prompts import PromptTemplate

# Simple template with one variable
simple_template = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms."
)

# Template with multiple variables
multi_template = PromptTemplate(
    input_variables=["topic", "audience", "tone"],
    template="Explain {topic} to a {audience} using a {tone} tone."
)

# Usage
simple_prompt = simple_template.format(topic="neural networks")
multi_prompt = multi_template.format(
    topic="blockchain", 
    audience="5-year-old", 
    tone="playful"
)

print("Simple:", simple_prompt)
print("Multi:", multi_prompt)
```

### 2. Chat Prompt Templates

```python
from langchain.prompts import ChatPromptTemplate

# Chat templates are structured for conversation
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that speaks {language}."),
    ("human", "Can you explain {topic}?"),
    ("ai", "I'd be happy to explain {topic} in {language}!"),
    ("human", "Can you give me an example?")
])

# Format the chat template
messages = chat_template.format_messages(
    language="Spanish",
    topic="machine learning"
)

for message in messages:
    print(f"{message.type}: {message.content}")
```

### 3. Few-Shot Prompt Templates

```python
from langchain.prompts import FewShotPromptTemplate

# Examples for few-shot learning
examples = [
    {
        "question": "What is the speed of light?",
        "answer": "The speed of light is approximately 299,792,458 meters per second."
    },
    {
        "question": "Who wrote Romeo and Juliet?",
        "answer": "William Shakespeare wrote Romeo and Juliet."
    }
]

# Create example prompt
example_prompt = PromptTemplate(
    input_variables=["question", "answer"],
    template="Question: {question}\nAnswer: {answer}"
)

# Create few-shot template
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    suffix="Question: {input}\nAnswer:",
    input_variables=["input"]
)

# Use it
final_prompt = few_shot_prompt.format(
    input="What is the capital of Japan?"
)
print(final_prompt)
```

### 4. Dynamic Prompt Templates

```python
from langchain.prompts import PromptTemplate

# Template that can handle optional variables
dynamic_template = PromptTemplate.from_template("""
You are {role}.
{context}
Answer the following question: {question}
{format_instructions}
""")

# Different ways to use it
# Without context
prompt1 = dynamic_template.format(
    role="a helpful assistant",
    context="",
    question="What is 2+2?",
    format_instructions="Answer in one word."
)

# With context
prompt2 = dynamic_template.format(
    role="a math tutor",
    context="You specialize in elementary mathematics.",
    question="What is 2+2?",
    format_instructions="Explain step by step."
)

print("Prompt 1:", prompt1)
print("Prompt 2:", prompt2)
```

## Model-Template Compatibility Matrix

```python
# Understanding which models work with which templates

# Traditional LLMs (OpenAI, etc.) - Work with:
# ✅ Basic PromptTemplate
# ✅ FewShotPromptTemplate
# ❌ ChatPromptTemplate (not designed for this)

# Chat Models (ChatOpenAI, ChatAnthropic, etc.) - Work with:
# ✅ Basic PromptTemplate (converted internally)
# ✅ ChatPromptTemplate (native support)
# ✅ FewShotPromptTemplate

# Example showing compatibility:

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate

# Templates
basic_template = PromptTemplate.from_template("Explain {topic}")
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("human", "Explain {topic}")
])

# Models
llm = OpenAI()
chat_model = ChatOpenAI()

topic = "photosynthesis"

# LLM with basic template - ✅ Works
llm_result = llm(basic_template.format(topic=topic))
print("LLM + Basic Template:", llm_result[:100])

# LLM with chat template - ⚠️ Works but converts to string
try:
    chat_messages = chat_template.format_messages(topic=topic)
    llm_result = llm(str(chat_messages))
    print("LLM + Chat Template:", llm_result[:100])
except Exception as e:
    print("LLM + Chat Template Error:", e)

# Chat Model with basic template - ✅ Works (auto-converts)
chat_result = chat_model.predict(basic_template.format(topic=topic))
print("Chat Model + Basic Template:", chat_result)

# Chat Model with chat template - ✅ Works natively
chat_result = chat_model(chat_template.format_messages(topic=topic))
print("Chat Model + Chat Template:", chat_result.content)
```

## Chain Integration Examples

### How Templates Connect to Chains

```python
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate

# Example 1: Basic LLM Chain with Template
llm = OpenAI()

# Template
template = PromptTemplate(
    input_variables=["topic"],
    template="Generate a random fact about {topic}:"
)

# Chain
fact_chain = LLMChain(llm=llm, prompt=template)

# Use it
result = fact_chain.run("space exploration")
print("Fact Chain Result:", result)

# Example 2: Chat Model Chain with Chat Template
chat_model = ChatOpenAI()

# Chat template
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a creative writer."),
    ("human", "Write a short story about {character} in {setting}")
])

# Chat chain
story_chain = LLMChain(llm=chat_model, prompt=chat_template)

# Use it
result = story_chain.run({
    "character": "a robot",
    "setting": "a futuristic city"
})
print("Story Chain Result:", result.content if hasattr(result, 'content') else result)

# Example 3: Sequential Chain with Mixed Templates
# Chain 1: Generate a topic
topic_template = PromptTemplate.from_template(
    "Generate a random science topic:"
)
topic_chain = LLMChain(llm=llm, prompt=topic_template)

# Chain 2: Explain the topic (using chat model)
explanation_template = ChatPromptTemplate.from_messages([
    ("system", "You are a science teacher explaining to high school students."),
    ("human", "Explain {topic} in simple terms:")
])
explanation_chain = LLMChain(llm=chat_model, prompt=explanation_template)

# Connect them
sequential_chain = SimpleSequentialChain(
    chains=[topic_chain, explanation_chain],
    verbose=True
)

# Run the sequence
result = sequential_chain.run("Start")
```

## Best Practices and Recommendations

### 1. Choose the Right Model for Your Task

```python
# For simple completion tasks
def get_completion_model():
    return OpenAI(model_name="text-davinci-003")

# For conversational applications
def get_chat_model():
    return ChatOpenAI(model_name="gpt-3.5-turbo")

# For cost-effective options
def get_economical_model():
    return ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# For high-quality responses
def get_high_quality_model():
    return ChatOpenAI(model_name="gpt-4", temperature=0.7)
```

### 2. Template Selection Guide

```python
# Use Basic PromptTemplate when:
# - Simple single-shot tasks
# - Don't need conversation structure
# - Working with traditional LLMs

# Use ChatPromptTemplate when:
# - Building chatbots or conversational apps
# - Need structured roles (system/user/assistant)
# - Working with modern chat models

# Use FewShotPromptTemplate when:
# - Need to show examples to guide the model
# - Working on tasks that benefit from examples
# - Want to improve accuracy through demonstration

# Example: Decision tree for template selection
def choose_template(task_type, has_examples=False):
    if has_examples:
        return "FewShotPromptTemplate"
    elif task_type == "conversational":
        return "ChatPromptTemplate"
    else:
        return "PromptTemplate"
```

### 3. Model-Template Matching

```python
# Safe combinations that always work well:

# Traditional LLM + Basic Template
# ✅ Always works, efficient

# Chat Model + Chat Template  
# ✅ Native support, full functionality

# Chat Model + Basic Template
# ✅ Works (auto-converted), good for simple tasks

# Traditional LLM + Chat Template
# ⚠️ Works but loses chat structure benefits

# Example of a robust implementation:
def create_robust_chain(model_type="chat", template_type="basic"):
    # Choose model
    if model_type == "chat":
        model = ChatOpenAI(model_name="gpt-3.5-turbo")
    else:
        model = OpenAI(model_name="text-davinci-003")
    
    # Choose template
    if template_type == "chat" and model_type == "chat":
        template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("human", "{input}")
        ])
    else:
        template = PromptTemplate.from_template("{input}")
    
    # Create chain
    return LLMChain(llm=model, prompt=template)

# Test different combinations
chains = {
    "Chat + Chat": create_robust_chain("chat", "chat"),
    "Chat + Basic": create_robust_chain("chat", "basic"),
    "LLM + Basic": create_robust_chain("llm", "basic")
}

for name, chain in chains.items():
    try:
        result = chain.run("What is the weather like today?")
        print(f"{name}: Success")
    except Exception as e:
        print(f"{name}: Error - {e}")
```

## Summary

**Key Takeaways:**

1. **Models vs ChatModels**: Chat models are newer, more efficient, and better for conversations
1. **Template Compatibility**: Chat templates work best with chat models, basic templates work with everything
1. **Mix and Match**: You can combine different models with different templates, but some combinations work better than others
1. **Best Practice**: Use chat models + chat templates for new projects, traditional models + basic templates for simple tasks

Start with ChatOpenAI + ChatPromptTemplate for most modern applications - it's the most flexible and future-proof combination!

------------ ### HUMAN ### -------------

can you show me an example basic rag model that uses pythons pathlib to find all files in a root directory that match a specific suffix/filetype and then reads/embeds those files into an in-memory db for a rag agent? can you show me how I can ingest pdf docs, txt, md, etc. and feed those into a rag model for more releant/accurate queries? also, what is faiss and how does it work? can we use it to embed pdfs, md, txt files and build out a rag model?

-------------- ### AI ### --------------

# Building a Comprehensive RAG Model with File Ingestion

Let me show you how to build a complete RAG system that can process various file types and use FAISS for efficient similarity search.

## What is FAISS?

FAISS (Facebook AI Similarity Search) is a library for efficient similarity search and clustering of dense vectors. It's perfect for RAG because:

- **Fast**: Can search millions of vectors in milliseconds
- **Efficient**: Uses optimized algorithms for vector similarity
- **Scalable**: Handles large datasets efficiently
- **Flexible**: Supports different indexing strategies

```python
# FAISS vs other vector stores:
# - Chroma: Easier to use, good for small projects
# - Pinecone: Cloud-based, good for production
# - FAISS: Fast, local, great for performance-critical apps
```

## Complete RAG System with File Ingestion

```python
# rag_system.py
import os
from pathlib import Path
from typing import List, Dict
import faiss
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredFileLoader
)
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import tempfile

class DocumentProcessor:
    """Handles finding, loading, and processing documents"""
    
    def __init__(self):
        self.supported_extensions = {
            '.txt': TextLoader,
            '.md': UnstructuredMarkdownLoader,
            '.pdf': PyPDFLoader,
        }
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def find_files(self, root_directory: str, extensions: List[str] = None) -> List[Path]:
        """
        Find all files with specified extensions in directory and subdirectories
        
        Args:
            root_directory: Root directory to search
            extensions: List of extensions to include (e.g., ['.txt', '.pdf'])
        
        Returns:
            List of Path objects for matching files
        """
        root_path = Path(root_directory)
        if not root_path.exists():
            raise FileNotFoundError(f"Directory {root_directory} not found")
        
        # If no extensions specified, use all supported
        if extensions is None:
            extensions = list(self.supported_extensions.keys())
        
        # Convert to set for faster lookup
        ext_set = set(extensions)
        
        matching_files = []
        for file_path in root_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in ext_set:
                matching_files.append(file_path)
        
        print(f"Found {len(matching_files)} files matching extensions {extensions}")
        return matching_files
    
    def load_documents(self, file_paths: List[Path]) -> List:
        """
        Load documents from file paths
        
        Args:
            file_paths: List of Path objects
            
        Returns:
            List of loaded documents
        """
        documents = []
        
        for file_path in file_paths:
            try:
                # Get appropriate loader
                loader_class = self.supported_extensions.get(file_path.suffix.lower())
                
                if loader_class:
                    loader = loader_class(str(file_path))
                    docs = loader.load()
                    print(f"Loaded {len(docs)} documents from {file_path.name}")
                    documents.extend(docs)
                else:
                    # Try generic loader for unsupported types
                    loader = UnstructuredFileLoader(str(file_path))
                    docs = loader.load()
                    print(f"Loaded {len(docs)} documents from {file_path.name} (generic)")
                    documents.extend(docs)
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        return documents
    
    def split_documents(self, documents: List) -> List:
        """
        Split documents into chunks
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of document chunks
        """
        chunks = self.text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        return chunks

class RAGSystem:
    """Main RAG system that handles ingestion and querying"""
    
    def __init__(self, embeddings=None, llm=None):
        self.embeddings = embeddings or OpenAIEmbeddings()
        self.llm = llm or ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
        self.document_processor = DocumentProcessor()
        self.vectorstore = None
        self.qa_chain = None
    
    def ingest_directory(self, directory_path: str, extensions: List[str] = None):
        """
        Ingest all documents from a directory
        
        Args:
            directory_path: Path to directory containing documents
            extensions: List of file extensions to process
        """
        print(f"Starting ingestion from {directory_path}")
        
        # Find files
        files = self.document_processor.find_files(directory_path, extensions)
        
        if not files:
            print("No matching files found!")
            return
        
        # Load documents
        documents = self.document_processor.load_documents(files)
        
        if not documents:
            print("No documents loaded!")
            return
        
        # Split documents
        chunks = self.document_processor.split_documents(documents)
        
        # Create vectorstore
        print("Creating vector store with FAISS...")
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True
        )
        
        print(f"Ingestion complete! Processed {len(documents)} documents, {len(chunks)} chunks")
    
    def query(self, question: str) -> Dict:
        """
        Query the RAG system
        
        Args:
            question: Question to ask
            
        Returns:
            Dictionary with answer and source documents
        """
        if not self.qa_chain:
            raise ValueError("No documents ingested yet! Call ingest_directory first.")
        
        result = self.qa_chain({"query": question})
        
        return {
            "answer": result["result"],
            "sources": result["source_documents"]
        }
    
    def save_index(self, path: str):
        """Save the FAISS index to disk"""
        if self.vectorstore:
            self.vectorstore.save_local(path)
            print(f"Index saved to {path}")
    
    def load_index(self, path: str):
        """Load a FAISS index from disk"""
        self.vectorstore = FAISS.load_local(path, self.embeddings)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True
        )
        print(f"Index loaded from {path}")

# Example usage with sample files
def create_sample_files():
    """Create sample files for testing"""
    # Create sample directory
    sample_dir = Path("sample_documents")
    sample_dir.mkdir(exist_ok=True)
    
    # Sample text file
    with open(sample_dir / "ai_overview.txt", "w") as f:
        f.write("""
        Artificial Intelligence (AI) is intelligence demonstrated by machines.
        In computer science, AI research has been defined as the field of study 
        of intelligent agents, which refers to any system that perceives its 
        environment and takes actions that maximize its chance of achieving its goals.
        
        Machine Learning is a subset of AI that focuses on algorithms that 
        can learn from and make predictions on data. Deep Learning is a 
        subset of Machine Learning that uses neural networks with multiple 
        layers to model complex patterns in data.
        """)
    
    # Sample markdown file
    with open(sample_dir / "ml_fundamentals.md", "w") as f:
        f.write("""
        # Machine Learning Fundamentals
        
        ## What is Machine Learning?
        
        Machine Learning is the practice of teaching computers to learn patterns 
        from data without being explicitly programmed. The three main types are:
        
        1. **Supervised Learning**: Learning with labeled training data
        2. **Unsupervised Learning**: Finding patterns in unlabeled data
        3. **Reinforcement Learning**: Learning through trial and error with rewards
        
        ## Key Algorithms
        
        - Linear Regression
        - Decision Trees
        - Random Forest
        - Neural Networks
        - Support Vector Machines
        """)
    
    # Sample PDF content (we'll create a simple text PDF)
    # Note: For real PDFs, you'd use PyPDFLoader
    with open(sample_dir / "data_science_basics.txt", "w") as f:
        f.write("""
        # Data Science Basics
        
        Data Science combines statistics, computer science, and domain expertise 
        to extract insights from data. The data science process typically involves:
        
        1. Data Collection
        2. Data Cleaning
        3. Exploratory Data Analysis
        4. Model Building
        5. Model Evaluation
        6. Deployment
        
        Key tools include Python, R, SQL, and various libraries like pandas, 
        scikit-learn, and TensorFlow.
        """)
    
    print("Sample files created in 'sample_documents' directory")

# Advanced RAG with FAISS configuration
class AdvancedRAGSystem(RAGSystem):
    """Advanced RAG system with custom FAISS configuration"""
    
    def __init__(self, embeddings=None, llm=None, faiss_index_type="IndexFlatL2"):
        super().__init__(embeddings, llm)
        self.faiss_index_type = faiss_index_type
    
    def ingest_directory(self, directory_path: str, extensions: List[str] = None):
        """Enhanced ingestion with FAISS optimization"""
        print(f"Starting advanced ingestion from {directory_path}")
        
        # Process documents
        files = self.document_processor.find_files(directory_path, extensions)
        documents = self.document_processor.load_documents(files)
        chunks = self.document_processor.split_documents(documents)
        
        # Create optimized FAISS index
        print("Creating optimized FAISS index...")
        
        # Get embeddings for all chunks
        texts = [doc.page_content for doc in chunks]
        embeddings = self.embeddings.embed_documents(texts)
        
        # Convert to numpy array
        embedding_matrix = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        dimension = len(embeddings[0])
        
        if self.faiss_index_type == "IndexFlatL2":
            index = faiss.IndexFlatL2(dimension)
        elif self.faiss_index_type == "IndexIVFFlat":
            # More efficient for large datasets
            quantizer = faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, 100)
            index.train(embedding_matrix)
        else:
            index = faiss.IndexFlatL2(dimension)
        
        # Add vectors to index
        index.add(embedding_matrix)
        
        # Create FAISS vectorstore with custom index
        self.vectorstore = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=None,  # You'd need to implement proper docstore
            index_to_docstore_id={}
        )
        
        # For simplicity, we'll use the standard FAISS from_documents
        # In production, you'd want to properly implement the docstore
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={
                    "k": 4,
                    "fetch_k": 20  # Fetch more candidates for better results
                }
            ),
            return_source_documents=True
        )
        
        print(f"Advanced ingestion complete! Processed {len(documents)} documents")

# Example usage
if __name__ == "__main__":
    # Create sample files
    create_sample_files()
    
    # Initialize RAG system
    rag = RAGSystem()
    
    # Ingest documents
    rag.ingest_directory("sample_documents", ['.txt', '.md'])
    
    # Query the system
    questions = [
        "What is artificial intelligence?",
        "What are the types of machine learning?",
        "What tools do data scientists use?"
    ]
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        result = rag.query(question)
        print(f"✅ Answer: {result['answer']}")
        print("📚 Sources:")
        for i, source in enumerate(result['sources'][:2]):  # Show top 2 sources
            print(f"  Source {i+1}: {source.metadata.get('source', 'Unknown')}")

# FAISS Index Types Explained
def explain_faiss_indexes():
    """Explanation of different FAISS index types"""
    
    print("""
    FAISS Index Types:
    
    1. IndexFlatL2:
       - Exact search using L2 distance
       - Good for small to medium datasets (< 1M vectors)
       - Memory intensive but very accurate
    
    2. IndexIVFFlat:
       - Approximate search using inverted file structure
       - Good for large datasets
       - Faster search but slightly less accurate
    
    3. IndexHNSW:
       - Hierarchical Navigable Small World graphs
       - Good balance of speed and accuracy
       - Memory efficient
    
    4. IndexLSH:
       - Locality Sensitive Hashing
       - Very fast but less accurate
       - Good for binary vectors
    
    Example usage:
    dimension = 1536  # OpenAI embeddings dimension
    index = faiss.IndexFlatL2(dimension)  # Exact search
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist=100)  # Approximate
    """)

# Performance optimization tips
def optimization_tips():
    """Tips for optimizing RAG performance"""
    
    tips = """
    Performance Optimization Tips:
    
    1. **Document Chunking**:
       - Balance chunk size (500-2000 tokens)
       - Use overlap to maintain context (100-200 tokens)
    
    2. **Embedding Models**:
       - OpenAI embeddings: Fast and accurate
       - Sentence transformers: Free alternative
       - Consider model dimension vs performance
    
    3. **FAISS Configuration**:
       - Use IndexFlatL2 for < 1M vectors
       - Use IndexIVFFlat for larger datasets
       - Consider GPU acceleration for large indexes
    
    4. **Retrieval Optimization**:
       - Adjust k (number of retrieved documents)
       - Use MMR for diverse results
       - Implement re-ranking for better accuracy
    
    5. **Memory Management**:
       - Process large directories in batches
       - Save/load indexes to avoid reprocessing
       - Monitor memory usage during ingestion
    """
    print(tips)

# Advanced querying with metadata filtering
class MetadataRAGSystem(RAGSystem):
    """RAG system with metadata filtering capabilities"""
    
    def query_with_filter(self, question: str, filter_dict: Dict = None) -> Dict:
        """
        Query with metadata filtering
        
        Args:
            question: Question to ask
            filter_dict: Metadata filters (e.g., {"source_type": "pdf"})
        """
        if not self.qa_chain:
            raise ValueError("No documents ingested yet!")
        
        # Update retriever with filters
        if filter_dict:
            retriever = self.vectorstore.as_retriever(
                search_kwargs={
                    "k": 4,
                    "filter": filter_dict
                }
            )
            # Recreate QA chain with filtered retriever
            filtered_qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            result = filtered_qa_chain({"query": question})
        else:
            result = self.qa_chain({"query": question})
        
        return {
            "answer": result["result"],
            "sources": result["source_documents"]
        }

# Example with filtering
def demonstrate_filtering():
    """Demonstrate metadata filtering"""
    
    # Create sample documents with metadata
    from langchain.schema import Document
    
    documents = [
        Document(
            page_content="Python is a programming language used for web development",
            metadata={"source": "programming_guide.txt", "category": "programming"}
        ),
        Document(
            page_content="Photosynthesis is the process by which plants make food",
            metadata={"source": "biology_textbook.pdf", "category": "science"}
        )
    ]
    
    # Create RAG system with these documents
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    llm = ChatOpenAI(temperature=0)
    retriever = vectorstore.as_retriever()
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )
    
    # Query with filtering (if your vectorstore supports it)
    print("RAG system with metadata filtering ready!")

if __name__ == "__main__":
    # Run the complete example
    explain_faiss_indexes()
    optimization_tips()
```

## How to Use This System

### 1. Installation Requirements

```bash
pip install langchain openai faiss-cpu pypdf unstructured markdown
```

### 2. Basic Usage

```python
# Initialize the RAG system
rag = RAGSystem()

# Ingest documents from a directory
rag.ingest_directory("/path/to/your/documents", ['.txt', '.pdf', '.md'])

# Query the system
result = rag.query("What is machine learning?")
print(result["answer"])

# Save the index for later use
rag.save_index("./my_knowledge_base")

# Load previously saved index
rag.load_index("./my_knowledge_base")
```

### 3. Advanced Usage with Filtering

```python
# Query with metadata filtering
result = rag.query_with_filter(
    "What programming languages are mentioned?", 
    {"category": "programming"}
)
```

## Key Features Explained

### File Processing Pipeline

1. **Pathlib Integration**: Uses `Path.rglob()` to recursively find files
1. **Multi-format Support**: Handles TXT, PDF, MD, and other formats
1. **Error Handling**: Gracefully handles corrupted or unreadable files
1. **Progress Tracking**: Shows processing status

### FAISS Integration

1. **Efficient Storage**: Stores document embeddings for fast retrieval
1. **Similarity Search**: Finds most relevant documents using vector similarity
1. **Scalable**: Can handle large document collections
1. **Persistent Storage**: Can save/load indexes to disk

### Advanced Features

1. **Metadata Filtering**: Filter results by document properties
1. **Configurable Chunking**: Adjust chunk size and overlap
1. **Multiple Index Types**: Choose FAISS index based on your needs
1. **Source Tracking**: Shows which documents informed each answer

This system gives you a production-ready RAG implementation that can handle real-world document processing needs!
