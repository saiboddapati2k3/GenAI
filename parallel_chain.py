from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain.schema.runnable import RunnableParallel
from dotenv import load_dotenv
load_dotenv()

model1 = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash'
)

model2 = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash'
)

prompt1 = PromptTemplate(
    template = '''
        Generate a short and simple notes from the following {text}
    ''',
    input_variables= ['text']

)

prompt2 = PromptTemplate(
    template = '''
        Generate some 5 questions from the following {text}
    ''',
    input_variables= ['text']

)

prompt3 = PromptTemplate(
    template = ''' 
        Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}
    ''' ,
    input_variables = ['notes' , 'quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {
        'notes': prompt1 | model1| parser ,
        'quiz' : prompt2 | model2 | parser

    }
)

merge_chain = prompt3 | model1 | parser


final_chain = parallel_chain | merge_chain

response = final_chain.invoke({'text' : 'Rishabh Pant is an Indian cricketer known for his explosive batting and sharp wicket-keeping skills. He made his Test debut in 2018 and quickly gained fame for match-winning performances, especially during the 2021 Australia tour. Despite a major car accident in 2022, he made a strong comeback in IPL 2024. Pant is admired for his fearless attitude and resilience, making him one of the most exciting players in modern cricket.'})

print(response)
