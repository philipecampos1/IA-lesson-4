import os
from langchain import hub
from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI

OPEN_AI_API = os.environ['OPEN_AI_API']

model = ChatOpenAI(
    model='gpt-3.5-turbo',
    api_key=OPEN_AI_API
    )

prompt = '''
Behave like a personal finance assistante. wich will answer questions
giving adivices about finance and investment
Questions : {q}
'''

prompt_template = PromptTemplate.from_template(prompt)

python_repl = PythonREPL()
python_repl_tool = Tool(
    name='Python REPL',
    description='Python shell. Use this to execute python code. Execute only valid python code'
                'If you need to get a return you can use the function "print(...)"'
                'Use for finance calculation necessary to answer the question ',
    func=python_repl.run
)

search = DuckDuckGoSearchRun()
duckduckgo_tool = Tool(
    name='Search DuckDuckGo',
    description='Usefull to find information about economy and options for investiment'
    'You always need to search on the internet for the bests tips using this tools.'
    'Do not answer directly. You need to inform that this answer has elements that has been search on the internet',
    func=search.run
)

react_instructions = hub.pull('hwchase17/react')

tools = [python_repl_tool, duckduckgo_tool]

agent = create_react_agent(
    llm=model,
    tools=tools,
    prompt=react_instructions,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)
question = '''
    I recive per month around £10000 and I spend around £8000 some months need to pay 200 to 2000 more.
    Witch investiment tips can you give me.
    '''

output = agent_executor.invoke(
    {'input': prompt_template.format(q=question)}
)

print(output.get('output'))
