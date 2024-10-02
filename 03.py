import os
from langchain.agents import Tool
from langchain.prompts import PromptTemplate
from langchain_experimental.utilities import PythonREPL
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain_openai import ChatOpenAI


OPEN_AI_API = os.environ['OPEN_AI_API']

model = ChatOpenAI(
    model='gpt-3.5-turbo',
    api_key=OPEN_AI_API
    )

python_repl = PythonREPL()
python_repl_tool = Tool(
    name='Python REPL',
    description='Python shell. Use this to execute python code. Execute only valid python code'
                'If you need to get a return you can use the function "print(...)"',
    func=python_repl.run
)

agent_executer = create_python_agent(
    llm=model,
    tool=python_repl_tool,
    verbose=True,
)

prompt_template = PromptTemplate(
    input_variables=['query'],
    template='''
    Solve this problem {query} explaning how you got this answer
    '''
)
query = r'How much is 20 por cent of 3000 '
prompt = prompt_template.format(query=query)

response = agent_executer.invoke(prompt)
print(response.get('output'))
