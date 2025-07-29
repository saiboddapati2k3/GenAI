from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('problem statement.pdf')

docs = loader.load()

for i , doc in enumerate(docs):
    print('This is the content in the page ' , i +1 )
    print(doc.page_content,"\n \n")
