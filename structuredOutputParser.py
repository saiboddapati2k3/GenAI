from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser , ResponseSchema


from dotenv import load_dotenv
load_dotenv()


model = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash'
)

schema = [
    ResponseSchema( name = 'fact_1' , description= 'Give me the fact 1 about the topic'),
    ResponseSchema( name = 'fact_2' , description= 'Give me the fact 2 about the topic'),
    ResponseSchema( name = 'fact_3' , description= 'Give me the fact 3 about the topic')
]


parser = StructuredOutputParser.from_response_schemas(schema)

prompt = PromptTemplate(
    template = 'Give me 3 intresting facts about the sport {topic} Please return only a valid JSON object with keys: , {format_instruction}',
    input_variables = ['topic'],
    partial_variables= {'format_instruction' : parser.get_format_instructions()}

)


chain = prompt | model | parser

response = chain.invoke({'topic': 'cricket'})

print(response)


chain.get_graph().print_ascii()
