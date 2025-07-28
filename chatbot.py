from langchain_google_genai import ChatGoogleGenerativeAI


from langchain_core.prompts import ChatPromptTemplate ,MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from dotenv import load_dotenv
load_dotenv()

# from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
# messages = [
#     SystemMessage(content="You are a helpful assistant. Answer my queries patiently."),
#     HumanMessage(content="What is the capital of India?")
# ]
# response = model.invoke(messages)
# messages.append(AIMessage(content=response))

prompt = ChatPromptTemplate(
    [
        ('system' , 'You are a helpful AI Assistant'),
        MessagesPlaceholder(variable_name= 'chat_history' ),
        ('human' , '{user_input}')
    ]
)

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.7,
    max_output_tokens=2048
)


chat_history = []


while True:

    que = input("You: ")
    if que.lower() in ['exit' , 'quit'] :
        break

    messages = prompt.format_messages(
        chat_history = chat_history ,
        user_input = que
    )

    input_text = '\n'.join(msg.content for msg in messages)
    response = model.invoke(input_text)

    print("AI: " , response.content)

    chat_history.append(HumanMessage(content=que))
    chat_history.append(AIMessage(content=response.content))

print(chat_history)

