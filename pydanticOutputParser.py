from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel ,Field



from dotenv import load_dotenv
load_dotenv()


class Person(BaseModel):

    name : 'str' = Field( description= 'Give me the name of the Person')
    age : 'int' = Field( description= 'Give me the age of the Preson')
    city: 'str' = Field( description= 'Give me the city of the Person')
    stats: list['str'] = Field(description= 'Give all the important stas of his career in a list ')

parser = PydanticOutputParser(pydantic_object= Person)


prompt = PromptTemplate(
    template= 'Give me some informtion about {person} {format_information}' ,
    input_variables= ['person'] ,
    partial_variables= {'format_information' : parser.get_format_instructions()}
)

model = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash'
)

chain = prompt | model | parser

response = chain.invoke({'person' : 'Rishab Pant'})

print(response.stats)

