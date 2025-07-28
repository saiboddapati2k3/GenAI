from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint

from dotenv import load_dotenv

load_dotenv()


llm = HuggingFaceEndpoint(repo_id= "mistralai/Mistral-7B-Instruct-v0.2" , task="text-generation")

model = ChatHuggingFace(
    llm = llm
)
response = model.invoke("What is the result of 3rd test match cricket between IND vs ENG test series 2025")
print(response.content)
