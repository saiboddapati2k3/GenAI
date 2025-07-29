from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser , PydanticOutputParser
from pydantic import BaseModel , Field
from typing import Literal

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 

from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser

from dotenv import load_dotenv
load_dotenv()

loader = PyPDFLoader('problem statement.pdf')
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300, 
    chunk_overlap = 20,
    separators= ['\n\n' , '\n' , ' ' ]
)

splitted_text = splitter.split_documents(docs)


print(type(docs[0]))
 