from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()



model = ChatOpenAI(
    model="gpt-4o",  
    temperature=0.7
)
response = model.invoke("What is the result of 3rd test match cricket between IND vs ENG test series 2025")
print(response.content)
