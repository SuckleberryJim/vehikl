import pprint

from langchain_community.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

query = input("search the web! > ") or "what is langchain?"

s = search.invoke(query)

pprint.pp(s)
