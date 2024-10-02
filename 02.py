import os
from langchain.prompts import PromptTemplate
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_openai import ChatOpenAI


OPEN_AI_API = os.environ['OPEN_AI_API']

model = ChatOpenAI(
    model='gpt-3.5-turbo',
    api_key=OPEN_AI_API
    )

wikpedia_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper()
)

agent_executer = create_python_agent(
    llm=model,
    tool=wikpedia_tool,
    verbose=True,
)

prompt_template = PromptTemplate(
    input_variables=['query'],
    template='''
    Search on the web for {query} and give a summary about it.
    '''
)
query = 'Alan Turing'
prompt = prompt_template.format(query=query)

response = agent_executer.invoke(prompt)
print(response.get('output'))
