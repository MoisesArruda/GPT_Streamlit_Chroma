
from dotenv import load_dotenv
from langchain.chains import LLMChain
from app.class_gpt import GPTConfig,PromptChroma,VectorStoreMemory
load_dotenv()

def gpt_input():

    llm_embeddings = GPTConfig().create_embeddings()
    pages = PromptChroma().create_pages(r'data/Containers_com_Docker.pdf')
    chunks = PromptChroma().create_chunks(pages)
    cache = VectorStoreMemory().save_cache()
    db = VectorStoreMemory().input_Chroma(chunks,llm_embeddings,cache)

    return db


def gpt_anwser(query=None,db=None):
     
    llm_chat = GPTConfig().create_chat()
    prompt = PromptChroma().create_prompt()
    db = db
    contexto,memory = VectorStoreMemory().input_memory(db,query)
    llm_chain = LLMChain(llm=llm_chat,prompt=prompt,memory=memory,verbose=True)
    response = llm_chain.run(query=query,contexto=contexto,memory=memory)

    return response


db = gpt_input()
print(gpt_anwser("O que é o Docker?",db))

