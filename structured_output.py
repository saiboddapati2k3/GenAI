from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict , Annotated , Optional , Literal

from pydantic import description , Field


load_dotenv()

class Review(TypedDict):

    summary: str = Field(description= 'Generate a 2 line summary of the review')
    sentiment: Literal['pos' , 'neg'] = Field( description = 'Classify the review in to positive or negative')
    author: Optional['str'] = Field(default = None , description = 'Write down the author of the Book if mentioned in the review' )
    key_plots: list['str'] = Field( description= 'List down all the key plots in the review')


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    max_output_tokens=2048
)



structured_model = model.with_structured_output(Review)

review_book = '''"Rich Dad Poor Dad" is a game-changing personal finance book that challenges conventional
 beliefs about money. Through the contrasting lessons of his two "dads"—one rich and one poor—Robert Kiyosaki
   breaks down complex financial principles into simple, relatable stories. It’s not a step-by-step guide, but 
   it shifts your mindset about assets, liabilities, and financial freedom. A must-read for anyone starting 
   their journey toward financial literacy.'''

response = structured_model.invoke( review_book)

print(response['author'])