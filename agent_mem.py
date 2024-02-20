import streamlit as st
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

st.title("Agente de LangChain con memoria")
st.write("Este agente tiene memoria. Cada vez que se le pregunta algo, recuerda lo que se le preguntó antes.")
st.write("Puede utilizar dos herramientas: Tavily y el Recuperador (Retriever) conectado a la transcripción del siguiente video de YouTube:")
st.video("https://youtu.be/hvAPnpSfSGo?si=WGO9qvzuBjRYKGaE")

# Tavily
from langchain_community.tools.tavily_search import TavilySearchResults
search = TavilySearchResults()

# Recuperador (Retriever)
from langchain_community.document_loaders import YoutubeLoader # cargador de transcripciones de Youtube
from langchain.text_splitter import RecursiveCharacterTextSplitter # divisor de textos
from langchain_openai import OpenAIEmbeddings # embeddings de OpenAI
from langchain_community.vectorstores import FAISS # almacenamiento de vectores

loader = YoutubeLoader.from_youtube_url(
    "https://youtu.be/hvAPnpSfSGo?si=WGO9qvzuBjRYKGaE", add_video_info=False # LangGraph: Multi-Agent Workflows
)

docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)
vector = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vector.as_retriever()

from langchain.tools.retriever import create_retriever_tool
retriever_tool = create_retriever_tool(
    retriever,
    "langgraph_search",
    "Search for information about LangGraph. For any questions about LangGraph, you must use this tool!",
)

tools = [search, retriever_tool]


# Modelo de lenguaje (LLM)
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

#Prompt
from langchain import hub
prompt = hub.pull("hwchase17/openai-functions-agent")

# Agente
from langchain.agents import create_openai_functions_agent
agent = create_openai_functions_agent(llm, tools, prompt)

# Agent Executor
from langchain.agents import AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("¿En qué te puedo ayudar?")

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    # Esto es necesario porque en la mayoría de los escenarios del mundo real, se necesita una identificación de sesión.
    # Realmente no se usa aquí porque estamos usando un ChatMessageHistory simple en memoria.
    lambda session_id: msgs,
    input_messages_key="input",
    history_messages_key="chat_history",
)

view_messages = st.expander("Ver el contenido de los mensajes en el estado de sesión de Streamlit")

# Renderizar mensajes actuales de StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# Si el usuario ingresa una nueva consulta, generar y mostrar una nueva respuesta
if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    # Nota: Langchain guarda automáticamente los nuevos mensajes en el historial durante la ejecución
    config = {"configurable": {"session_id": "any"}}
    response = agent_with_chat_history.invoke({"input": prompt}, config)
    st.chat_message("assistant").write(response["output"])

# Mostrar los mensajes hasta el final, para que los recién generados aparezcan inmediatamente
with view_messages:
    """
    Historial de mensajes inicializado con:
    ```python
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    ```

    Contenido de `st.session_state.langchain_messages`:
    """
    view_messages.json(st.session_state.langchain_messages)