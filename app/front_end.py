#%%
import streamlit as st
from app.gpt_chat import gpt_input,gpt_anwser

def front_end():

    st.set_page_config(page_title="Docker Chatbot", page_icon="📋",layout="wide")

    with st.sidebar:
        st.title('ChatPDF LangChain App')
        st.markdown(
            ''' 
            This app is an **LLM-powered chatbot** designed for professionals to answer questions about Docker and Containers. It uses:
            - [Streamlit](https://docs.streamlit.io/)
            - [LangChain](https://python.langchain.com/docs/get_started/introduction)
            - [OpenAI API](https://openai.com/blog/openai-api)
            '''
        )
        st.write('Created by [Moisés Arruda](https://www.linkedin.com/in/dataengineer-moisesarruda/)')

    st.title("Docker ChatBot : Pergunte qualquer coisa sobre Docker e Containers!")

    if "messages" not in st.session_state.keys():
        st.session_state.messages=[
            {"role":"assistant","content": "Olá! Se não sabe por onde começar, que tal perguntar para que serve o Docker e o que são Containers?"}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_prompt = st.chat_input()

    if user_prompt is not None:
        st.session_state.messages.append(
            {"role":"user","content":user_prompt})
        with st.chat_message("user"):
            st.write(user_prompt)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Loading..."):
                db = gpt_input()
                ai_response = gpt_anwser(query=user_prompt,db=db)
                ai_response = ai_response
                st.write(ai_response)
        new_ai_message = {"role":"assistant","content":ai_response}
        st.session_state.messages.append(new_ai_message)

#front_end()