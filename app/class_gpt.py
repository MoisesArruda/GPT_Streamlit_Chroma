#%%
import os
import glob

from typing import List
from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory,FileChatMessageHistory
from langchain.vectorstores import Chroma
from PyPDF2 import PdfReader
from langchain.cache import InMemoryCache

load_dotenv()

###### Configurar LLM_Chat e Embeddings ######

class GPTConfig:

    """
    Classe para configurar os objetos do AzureChatOpenAI e OpenAIEmbeddings para serem utilizados.
    
    Attributes:
        openai_api_base (str): A URL base da API do OpenAI.
        openai_api_version (str): A versão da API do OpenAI.
        openai_api_key (str): A chave da API do OpenAI.
        openai_api_type (str): O tipo da API do OpenAI.
        deployment_name (str): O nome do deployment da API do OpenAI.
        temperature (float): O valor da temperatura para determinar o comportamento da resposta, por padrão é 0.0.
        chunk_size (int): O tamanho do chunk para o OpenAIEmbeddings.
    """
    def __init__(self):
        self.openai_api_base = os.getenv("OPENAI_API_BASE")
        self.openai_api_version = os.getenv("OPENAI_API_VERSION")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_api_type = os.getenv("OPENAI_API_TYPE")
        self.deployment_name = os.getenv("DEPLOYMENT_NAME")
        self.deployment = os.getenv("EMBEDDING_DEPLOYMENT_NAME")
        self.max_tokens=7000

    def create_chat(self,t=0.1):
        """
        Configura os objetos do AzureChatOpenAI para serem utilizados.

        Returns:
            AzureChatOpenAI: O objeto de AzureChatOpenAi configurado.
        """
        return AzureChatOpenAI(
            openai_api_base=self.openai_api_base,
                openai_api_version=self.openai_api_version,
                openai_api_key=self.openai_api_key,
                openai_api_type=self.openai_api_type,
                deployment_name=self.deployment_name,
                temperature=t,
                max_tokens=self.max_tokens
        )
    
    def create_embeddings(self,chunk_size=1):
        """
        Configura os objetos do OpenAIEmbeddings para serem utilizados.

        Returns:
            OpenAIEmbeddings: O objeto de OpenAIEmbeddings configurado.
        """
        return OpenAIEmbeddings(
            deployment=self.deployment,
            chunk_size=chunk_size
            )


###### Detectar arquivos ######

class FileData:
    """
    Classe para configurar o caminho da pasta de dados.
    """
    def __init__(self,folder_path=None):
        self.folder_path= folder_path or r'data/'

    def get_folder(self):
        return self.folder_path
    
    def pdf_files(self):
        """
        Verifica se a pasta existe e retorna os nomes dos arquivos PDF na pasta.

        Args:
            pasta(str): Caminho da pasta.

        Return:
            List[str]: Lista com os nomes dos arquivos PDF na pasta.

        Raises:
            FileNotFoundError: Se a pasta não existir.
            ValueError: Se não houver arquivos PDF na pasta.
        """
        try:
            if not os.path.exists(self.folder_path):
                raise FileNotFoundError(f'A pasta {self.folder_path} não existe')

            files_folder = glob.glob(os.path.join(self.folder_path, '*.pdf'))
            if not files_folder:
                raise ValueError('Erro: Não há arquivos .pdf na pasta')
            else:
                # Para cada arquivo na pasta, retornar apenas o nome sem o caminho completo
                nomes_arquivos = [
                    os.path.basename(arquivo) for arquivo in files_folder
                ]

            return nomes_arquivos
        except (FileNotFoundError, ValueError) as e:
            raise


###### Prompt e leitura arquivo ######
    
class PromptChroma:
    """
    Classe para configurar o prompt e em sequência realizar o armazenamento
    do documento que será utilizado para realizar a busca.
    """
    def __init__(self):
        self.prompt=None
        self.loader=None
        self.pages=None
        self.chunks=None
        

    def create_prompt(self) -> PromptTemplate:
        """Cria o prompt a partir do template."""  
        template = """Você é um agente de IA que deve auxiliar as dúvidas das pessoas com suas dúvidas. Responda as pessoas sobre assuntos relacionados a containers e docker.\
              Responda as perguntas sempre em português. Caso não consiga responder, responda com "Eu não sei".
        chat_history = {chat_history}
        context = {context}
        Human: {query}
        Answer:"""

        prompt = PromptTemplate(template=template,input_variables=['chat_history','query','context'])
        self.prompt=prompt
        return self.prompt

    def create_chunks(self,caminho_arquivo):
        """Carrega e divide o documento. Cria chunks a partir do loader."""
        self.loader=PdfReader(caminho_arquivo)
        text = ""
        for page in self.loader.pages:
            text += page.extract_text()
        
        text_splitter = RecursiveCharacterTextSplitter(
            
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        self.chunks = text_splitter.split_text(text)

        return self.chunks

    

#    def create_chunks(self,text):
        """Cria chunks a partir do loader."""
 #       self.text_splitter = CharacterTextSplitter(
            
  #          chunk_size=1000,
   #         chunk_overlap=200,
    #        length_function=len
     #   )

      #  self.chunks = self.text_splitter.split_text(text)

       # return self.chunks
    

class VectorStoreMemory():
    """
    Classe para configurar o vector store e em sequência realizar o armazenamento
    do documento que será utilizado para realizar a busca.
    """
    def __init__(self):
        self.db = None
        self.context = None
        self.memory = None
        self.cache = None

    def save_cache(self):
        
        self.llm_cache = InMemoryCache()

        return self.llm_cache

    def input_Chroma(self,chunks,embeddings,cache):
        """Cria o vector store a partir do loader."""
        self.db = Chroma.from_texts(texts=chunks,embedding=embeddings,cache=cache)

        return self.db

    def input_memory(self,db,query):
        """Cria o vector store a partir do loader."""

        self.context = db.similarity_search(query,k=1)

        self.memory = ConversationBufferMemory(
            chat_memory = FileChatMessageHistory(file_path="historic_json/messages.json"),
            memory_key="chat_history",
            input_key="query",
            contexto=self.context,
            return_messages=True
        )

        return self.context,self.memory
    
