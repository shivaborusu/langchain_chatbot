
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
# from langchain_core.messages import SystemMessage
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import multiply, add, exponentiate
from langchain_community.tools import BraveSearch

import streamlit as st
import mlflow
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()

# Set the MLflow tracking URI
MLFLOW_TRACKING_SERVER = os.environ.get("MLFLOW_TRACKING_SERVER")
mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_SERVER)

experiment_name = "experiment_"+datetime.now().strftime("%m_%d_%Y_%H_%M")

if 'experiment_name' not in st.session_state:
    st.session_state.experiment_name = experiment_name

mlflow.set_experiment(st.session_state.experiment_name)
mlflow.set_active_model(name="langchain_chat_groq_llama_3")
mlflow.langchain.autolog()


messages = [
    ('system', 'you are a chatbot, keep answers short and limit to 4 sentences. \
     When you do not have the required capabilities or you do not have the  \
      context beyond your training cut off date use these tools, only one at a time \
     {available_tools}. \
     never call any tool which is not listed'),
    MessagesPlaceholder(variable_name='history', n_messages=6),
    ('human', '{question}'),
    MessagesPlaceholder(variable_name='agent_scratchpad')
]

brave_search = BraveSearch.from_api_key(api_key=os.environ.get('BRAVE_API_KEY'),
                                 search_kwargs={"count": 1})

available_tools = [multiply, add, exponentiate, brave_search]

chat_prompt = ChatPromptTemplate.from_messages(messages=messages)


llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.3, max_tokens=300)

# chain = chat_prompt | llm 
agent_compose = create_tool_calling_agent(llm=llm, tools=available_tools, prompt=chat_prompt)
chain = AgentExecutor(agent=agent_compose, tools=available_tools, verbose=True)

st.title("ChatJPT 🚀")

question = st.chat_input("Enter your question here")


if "message_history" not in st.session_state:
    st.session_state.message_history = []


for message in st.session_state.message_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


if question:
    st.chat_message("human").markdown(question)
    response = chain.invoke({'available_tools':available_tools,
                             'history':st.session_state.message_history, "question":question})
    st.session_state.message_history.append({'role':'human', 'content':question})
    st.chat_message("assistant").markdown(response['output'])
    st.session_state.message_history.append({'role':'assistant', 'content':response['output']})

    # print("MESSAGE HIST: ",st.session_state.message_history)
    # print("CHAT PROMPT: ",chat_prompt.invoke({"question":question, 'history':st.session_state.message_history}))