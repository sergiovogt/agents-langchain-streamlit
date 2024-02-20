from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Tavily
from langchain_community.tools.tavily_search import TavilySearchResults
search = TavilySearchResults()

# Recuperador (Retriever)
from langchain_community.document_loaders import YoutubeLoader # cargador de transcripciones de Youtube
from langchain.text_splitter import RecursiveCharacterTextSplitter # divisor de textos
from langchain_openai import OpenAIEmbeddings # embeddings de OpenAI
from langchain_community.vectorstores import FAISS # almacenamiento de vectores

# loader = YoutubeLoader.from_youtube_url(
#     "https://youtu.be/hvAPnpSfSGo?si=WGO9qvzuBjRYKGaE", add_video_info=False # LangGraph: Multi-Agent Workflows
# )

loader = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=T2M9hSswlIs", add_video_info=False # LangGraph: Multi-Agent Workflows
)


docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)
vector = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vector.as_retriever()

from langchain.tools.retriever import create_retriever_tool
# retriever_tool = create_retriever_tool(
#     retriever,
#     "langgraph_search",
#     "Search for information about LangGraph. For any questions about LangGraph, you must use this tool!",
# )


retriever_tool = create_retriever_tool(
    retriever,
    "youtube_advicer",
    "Información sobre consejos de YouTube. Para cualquier pregunta sobre consejos sobre como iniciar un canal de YouTube en el 2024, debe usar esta herramienta!",
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


