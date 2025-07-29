from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser , PydanticOutputParser

from langchain_core.runnables import RunnableParallel , RunnableSequence , RunnablePassthrough , RunnableLambda , RunnableMap
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash' 
)

prompt_summary = PromptTemplate(
    template = '''
        Summarize the following {topic} in about 50 words
        ''',
    input_variables= ['topic']
)

prompt_quiz = PromptTemplate(
    template = '''
            Generate a quiz on this {sport}..Include 5 questions .
        ''',
    input_variables= ['sport']

)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'summary' : RunnableSequence(prompt_summary , model , parser) ,
    'quiz' : RunnableSequence( prompt_quiz , model ,parser)
}
)

response = parallel_chain.invoke({'topic' : 'Cricket as a Sport' , 'sport': 'Cricket as a Sport'})


print(response)

