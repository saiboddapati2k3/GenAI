from langchain_google_genai import  ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature = 1.5,
    max_output_tokens = 2048
)

prompt1 = PromptTemplate(
    template= 'Write a detailed report on the following {topic}. ' ,
    input_variables= ['topic']
)

prompt2 = PromptTemplate(
    template = 'summarize the following {text} in about 5 lines' ,
    input_variables = ['text']
)

formatted_promopt = prompt1.format( topic = 'cricket')

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic' : 'cricket'})
print(result)

