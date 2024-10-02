import os
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI

OPEN_AI_API = os.environ['OPEN_AI_API']

model = ChatOpenAI(
    model='gpt-4',
    api_key=OPEN_AI_API
    )

db = SQLDatabase.from_uri('sqlite:///ipca.db')

toolkit = SQLDatabaseToolkit(
    db=db,
    llm=model
)
system_message = hub.pull('hwchase17/react')

agent = create_react_agent(
    llm=model,
    tools=toolkit.get_tools(),
    prompt=system_message,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=toolkit.get_tools(),
    verbose=True
)


prompt = '''
Use the necessary tools to answer questions related to IPCA throw the years.
Questions {q}
'''

prompt_template = PromptTemplate.from_template(prompt)

question = '''Based in the history of the data since 2004.
Do a preview of values of IPCA each future month until the end of 2024'''

output = agent_executor.invoke(
    {'input': prompt_template.format(q=question)}
)

print(output.get('output'))
