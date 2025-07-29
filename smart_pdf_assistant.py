from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
load_dotenv()


model = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash'
)

loader = PyPDFLoader('problem statement.pdf')
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300, 
    chunk_overlap = 20,
    separators= ['\n\n' , '\n' , ' ' ]
)

splitted_text = splitter.split_documents(docs)
embedding = GoogleGenerativeAIEmbeddings( model =  'models/embedding-001')

vectorstore = Chroma.from_documents(splitted_text , embedding  , persist_directory= './store')
vectorstore.persist()

parser = StrOutputParser()

user_query = input('Enter your query: ')
retrieved_docs = vectorstore.similarity_search(user_query, k=3)

prompt = PromptTemplate(
    template='''
        You are an helpful AI assistant.Please asnwer the following question based ont he given context..here is the context ->
        {context} and here is the query -> {user_query}
    ''',
    input_variables= ['context' , 'user_query']
)

chain = prompt | model | parser

response = chain.invoke({'context' : retrieved_docs , 'user_query' : user_query})
print(response)





