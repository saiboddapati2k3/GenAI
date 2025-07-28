from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline

from dotenv import load_dotenv

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model = "",
    task = "",

    model_kwargs= {
    }
)


model = ChatHuggingFace(llm = llm)

response = model.invoke("This is some random question")
print(response.content)