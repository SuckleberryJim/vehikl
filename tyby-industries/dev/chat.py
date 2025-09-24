import os
import subprocess as sp
from pathlib import Path

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_deepseek import ChatDeepSeek
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

load_dotenv()


def model_selector(provider: str = "openai"):
    # models = ["openai/gpt-5-mini", "google/gemini-2.5-flash", "x-ai/grok-code-fast-1"]
    base = os.getenv("OPENAI_URL_BASE")
    key = os.getenv("OPENAI_API_KEY")
    global llm
    llm = None
    match provider:
        case "openai":
            llm = "openai/gpt-5-mini"
        case "gemini":
            llm = "google/gemini-2.5-flash"
        case "xai":
            llm = "x-ai/grok-code-fast-1"
        case "deepseek":
            llm = "deepseek/deepseek-chat"
        case "qwen":
            llm = "alibaba/qwen3-coder-plus"
        case "llama":
            llm = "meta-llama/llama-4-maverick:free"
        case _:
            llm = "openai/gpt-5-nano"

    print(f" ### {llm} ### ".center(70, "-"))
    return ChatOpenAI(
        model=llm,
        temperature=0,
        base_url=base,
        api_key=key,
    )  # max_tokens=200


model = model_selector(input("model to use? > (default openai gpt-5-nano) > "))

llm = llm.split("/")[-1]

fp = Path(f"./responses/{llm}.md")

if fp.exists():
    for i in range(1, 100):
        n = Path(f"./responses/{llm}-c{i}.md")
        if not n.exists():
            fp = n
        else:
            continue

template = ChatPromptTemplate(
    [
        (
            "system",
            "you are a python and ML/AI tutor with a specialty in {topic}. your primary objective is to teach {topic} to someone with no experience in {topic}. answer user queries in a detailed and thorough yet simple manner. output your responses in markdown. use plenty of coding sections in your replies, include thorough comments and explantions.",
        ),
        ("assistant", "I will provide simple yet detailed prompts on {topic}"),
        ("human", "explain {topic} to me in simple yet detailed terms."),
    ]
)
chat = template.format_messages(topic="langchain")


chat.append(model.invoke(chat))
# print(chat[-1].content)

chat.append(
    HumanMessage(
        "what is the best way to master langchain? it feels like drinking from a firehose trying to understand everything all at once, where should I begin and what should I build first to establish a strong foundation of skills?"
    )
)

chat.append(model.invoke(chat))

chat.append(
    HumanMessage(
        "what are the various types of model templates and integrations? can you break them down for me and show me some examples? what is the difference between models and chatmodels? like openai() and chatopenai()? how about the various templates? can each model interop with each template or does each have a specific prompt/model that must go with its chain?"
    )
)

chat.append(model.invoke(chat))

chat.append(
    HumanMessage(
        "can you show me an example basic rag model that uses pythons pathlib to find all files in a root directory that match a specific suffix/filetype and then reads/embeds those files into an in-memory db for a rag agent? can you show me how I can ingest pdf docs, txt, md, etc. and feed those into a rag model for more releant/accurate queries? also, what is faiss and how does it work? can we use it to embed pdfs, md, txt files and build out a rag model?"
    )
)

chat.append(model.invoke(chat))


persist = True

while persist:
    q = input("> ")
    if q == "q":
        break

    chat.append(HumanMessage(q))
    chat.append(model.invoke(chat))
    # print(chat[-1].content)

if fp.exists():
    fp.unlink()

with fp.open("a") as f:
    for i in chat:
        f.write("\n" * 2 + f" ### {i.type.upper()} ### ".center(40, "-") + "\n" * 2)
        f.write(i.content)

sp.run(["mdformat", str(fp)])
# use d and u for 1/2 page up/down moves
sp.run(
    ["bat", "-sl", "markdown", "--decorations", "never", "--color", "always", str(fp)]
)
