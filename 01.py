# from langchain_community.tools import DuckDuckGoSearchRun

# ddg_search = DuckDuckGoSearchRun()

# search_result = ddg_search.run('Who was Alan Turning?')
# print(search_result)


# from langchain_experimental.utilities import PythonREPL


# python_repl = PythonREPL()
# result = python_repl.run('print(5 * 5)')
# print(result)


from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wikpedia = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper()

)

wikpedia_results = wikpedia.run('Who was Alan Turing')
print(wikpedia_results)
