import os
import pprint
import subprocess as sp
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

api = os.getenv("OPENAI_API_KEY")
base = os.getenv("OPENAI_API_BASE")

foundation = "openai/gpt-5-mini"

allowed_models = os.getenv("ALLOWED_MODELS")
print(allowed_models)

# ALLOWED_MODELS = foundation
# os.putenv("ALLOWED_MODELS", foundation)
# pprint.pp(os.getenv("ALLOWED_MODELS"))

llm = ChatOpenAI(model=foundation, base_url=base, api_key=api)

messages = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful programming assistant wth a specialty in {topic}. output your responses in markdown format.",
        ),
        ("assistant", "I will explain {topic} in detailed yet understandable terms."),
        (
            "human",
            "What is {topic} and how does it work? Can you show me some simple examples?",
        ),
    ]
)

buf = lambda: print("\n" + "-" * 25 + "\n")

r = llm.invoke(chat := messages.format_messages(topic="langchain"))
chat.append(r)

chat.append(
    HumanMessage(
        "great, this lib looks absolutely indespensable for building any type of RAG/agentic models. problem is it feels a bit like drinking from a firehose at the moment, can you give me a beginners tutorial and show me the best place to start? I want to build a strong foundation and focus on the fundamentals so that I can have a strong base to build robust and powerful ai systems from in the future!"
    )
)
r2 = llm.invoke(chat)
chat.append(r2)

chat.append(
    HumanMessage(
        "can you give me a basic beginner tutorial and tell me what the best place to start is? in what order should I tackle the concepts? what should I build first to get a solid foundation of knowledge? what should I save for later down the line? basically, what is most important to learn in the beginning and what should I focus on?"
    )
)
r3 = llm.invoke(chat)
chat.append(r3)

chat.append(
    HumanMessage(
        "can you show me a basic example of a rag pipeline that can ingest txt, md, and pdf documents? keep it simple and use plenty of examples please! thank you!"
    )
)
r4 = llm.invoke(chat)
chat.append(r4)


while True:
    q = input("> ")
    if q == "q":
        break

    buf()
    chat.append(HumanMessage(q))
    chat.append(r := llm.invoke(chat))

    print(r)
    buf()

fp = Path("./response.md")

if fp.exists():
    fp.unlink()

with fp.open("a") as f:
    for i in chat:
        f.write(i.content)
        f.write("\n\n" + "_" * 25 + "\n")

# else:

print("finished reading all md files!")
sp.run(["mdformat", "response.md"])

print("formatted respone.md file!")
sp.run(["bat", "response.md"])
