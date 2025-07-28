# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_huggingface import ChatHuggingFace
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser , PydanticOutputParser
# from pydantic import BaseModel , Field
# from typing import Literal

# from langchain_core.runnables import RunnableBranch


# from dotenv import load_dotenv
# load_dotenv()

# model = ChatGoogleGenerativeAI(
#     model = 'gemini-2.5-flash'
# )

# class Sentiment(BaseModel):

#     sentiment : Literal['positive' ,'negative'] = Field(description= 'Classify the sentiment of the given review into positive or negative')

# parser = PydanticOutputParser( pydantic_object= Sentiment)

# prompt = PromptTemplate(
#     template = '''
#         Classify the sentiment of the given {feedback} into Positive or Negative  ,{format_instruction}
# ''',
#     input_variables= ['feedback'] ,
#     partial_variables = {'format_instruction' : parser.get_format_instructions()}
# )


# feedback = '''
#     The iphone 15 pro has got mixed reviews among the users. 
#     Some users find it very basic with no modifications from 14 models 
#     while the other find the phone worth the money.

# '''
# classifier_chain = prompt | model | parser

# positive_prompt = PromptTemplate(
#     template='''
#     Give a thanking msg to the user for providing a positive feedback which is {feedback}
# ''',

# input_variables= ['feedback']
# )

# positive_chain = positive_prompt | model | StrOutputParser()

# negative_prompt = PromptTemplate(
#     template='''
#     Appologize to the user and ask for any further queries and suggestions to improve the product .The feedback given by the user is {feedback}
# ''',
# input_variables= ['feedback']
# )

# negative_chain = negative_prompt | model | StrOutputParser()

# default_chain = PromptTemplate(
#     template="Sorry, we could not understand the sentiment of the following: {feedback}",
#     input_variables=["feedback"]
# ) | model | StrOutputParser()

# branch_chain = RunnableBranch(
#     ( lambda x: x.sentiment == 'positive' , positive_chain),
#     ( lambda x: x.sentiment == 'negative' , negative_chain),
#     (default_chain)
# )


# full_chain = classifier_chain | branch_chain


# response = full_chain.invoke({"feedback" : feedback})

# print(response)





from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser , PydanticOutputParser
from pydantic import BaseModel , Field
from typing import Literal

from langchain_core.runnables import RunnableBranch


from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(
    model = 'gemini-2.5-flash'
)


class Sentiment(BaseModel):

    sentiment : Literal['positive' ,'negative'] = Field(description= 'Classify the sentiment of the given review into positive or negative')

parser = PydanticOutputParser( pydantic_object= Sentiment)


prompt = PromptTemplate(
    template= '''
    Give me the sentiment of the following {feedback} {format_instruction}

''' ,
input_variables= ['feedback'] ,
partial_variables= {'format_instruction': parser.get_format_instructions()}
)


feedback = '''
    The iphone 15 pro has got mixed reviews among the users. 
    Some users find it very basic with no modifications from 14 models 
    while the other find the phone worth the money.

'''


sentiment_chain = prompt | model | parser


positive_prompt = PromptTemplate(
    template= '''
    Write an appropriate response to this {feedback}  feedback in about 2 lines .
''', input_variables = ['sentiment']
)

positive_chain = positive_prompt | model | StrOutputParser()


negative_prompt = PromptTemplate(
    template= '''
    Write an appropriate response to this {feedback} feedback in about 2 lines .
''', input_variables = ['sentiment']
)

negative_chain = positive_prompt | model | StrOutputParser()


default_chain = PromptTemplate(
    template="Sorry, we could not understand the sentiment of the following: {feedback}",
    input_variables=["feedback"]
) | model | StrOutputParser()


branch_chain = RunnableBranch(
    ( lambda x: x.sentiment == 'positive' , positive_chain),
    ( lambda x: x.sentiment == 'negative' , negative_chain),
    (default_chain)
)



full_chain = sentiment_chain | branch_chain

response = full_chain.invoke({'feedback': feedback})

print(response)
