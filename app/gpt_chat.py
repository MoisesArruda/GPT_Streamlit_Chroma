
from dotenv import load_dotenv
from langchain.chains import LLMChain
from class_gpt import GPTConfig,PromptChroma,VectorStoreMemory

load_dotenv()

def gpt_input():

    llm_embeddings = GPTConfig().create_embeddings()
    #pages = PromptChroma().create_pages(r'data/Containers_com_Docker.pdf')
    chunks = PromptChroma().create_chunks(r'data/Containers_com_Docker.pdf')
    cache = VectorStoreMemory().save_cache()
    db = VectorStoreMemory().input_Chroma(chunks,llm_embeddings,cache)

    return db


def gpt_anwser(query=None,db=None):
     
    llm_chat = GPTConfig().create_chat()
    prompt = PromptChroma().create_prompt()
    db = db
    contexto,memory = VectorStoreMemory().input_memory(db,query)
    llm_chain = LLMChain(llm=llm_chat,prompt=prompt,memory=memory,verbose=True)
    response = llm_chain.run(query=query,context=contexto,memory=memory)

    return response


chamadas = 0

if chamadas == 0:
    db = gpt_input()
    chamadas += 1
else:
    pass
gpt_anwser("Como criar uma imagem?",db)

