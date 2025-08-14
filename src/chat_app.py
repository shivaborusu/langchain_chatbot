from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
# from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
import streamlit as st
import mlflow
from datetime import datetime
from dotenv import load_dotenv
import os
load_dotenv()

# Set the MLflow tracking URI
MLFLOW_TRACKING_SERVER = os.environ.get("MLFLOW_TRACKING_SERVER")
mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_SERVER)

experiment_name = "experiment_"+datetime.now().strftime("%m_%d_%Y_%H_%M")


if 'experiment_name' not in st.session_state:
    st.session_state.experiment_name = experiment_name

mlflow.set_experiment(st.session_state.experiment_name)

mlflow.langchain.autolog()

messages = [
    ('system', 'you are a chatbot application, keep answers short and limit to 2 sentences'),
    MessagesPlaceholder(variable_name='history', n_messages=6),
    ('human', '{question}')
]

chat_prompt = ChatPromptTemplate.from_messages(messages=messages)


llm = ChatGroq(model="llama-3.3-70b-versatile",temperature=0.3, max_tokens=100)


chain = chat_prompt | llm 


st.title("ChatJPT 🚀")

question = st.chat_input("Enter your question here")


if "message_history" not in st.session_state:
    st.session_state.message_history = []


for message in st.session_state.message_history:
    with st.chat_message(message['role']):
        st.markdown(message['content'])


if question:
    st.chat_message("human").markdown(question)
    response = chain.invoke({"question":question, 'history':st.session_state.message_history})
    st.session_state.message_history.append({'role':'human', 'content':question})
    st.chat_message("assistant").markdown(response.content)
    st.session_state.message_history.append({'role':'assistant', 'content':response.content})

    # print("MESSAGE HIST: ",st.session_state.message_history)
    # print("CHAT PROMPT: ",chat_prompt.invoke({"question":question, 'history':st.session_state.message_history}))


# if st.button("Clear"):
#     st.session_state.message_history = []