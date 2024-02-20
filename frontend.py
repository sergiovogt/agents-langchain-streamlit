import streamlit as st
import backend
from langchain_community.callbacks import StreamlitCallbackHandler

st.title("Agente de LangChain sin memoria")
st.write("Este agente no tiene memoria. Cada vez que se le pregunta algo, no recuerda lo que se le preguntó antes.")
st.write("Puede utilizar dos herramientas: Tavily y el Recuperador (Retriever) conectado a la transcripción del siguiente video de YouTube:")
st.video("https://youtu.be/hvAPnpSfSGo?si=WGO9qvzuBjRYKGaE")

if consulta := st.chat_input():
    st.chat_message("user").write(consulta)
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        response = backend.agent_executor.invoke(
            {"input": consulta}, {"callbacks": [st_callback]}
        )
        st.write(response["output"])