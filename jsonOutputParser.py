from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

load_dotenv()


model = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash'

)


parser = JsonOutputParser()

prompt1 = PromptTemplate(
    template = 'Give me the name age and city of a sports person from indian test cricket team wicket keepers \n {format_instruction}',
    input_variables = [],
    partial_variables= { 'format_instruction' : parser.get_format_instructions() }

)



# prompt = prompt1.format()

# result = model.invoke(prompt)
# f_result = parser.parse(result.content)

# print(f_result)

chain = prompt1 | model | parser

response = chain.invoke({})

print(response)


