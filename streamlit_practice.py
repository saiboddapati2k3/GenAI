from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
import streamlit as st
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id = "mistralai/Mistral-7B-Instruct-v0.2",
    task = 'text-generation'
)

model = ChatHuggingFace( llm = llm )

st.title('Welcome to the ChatBot')
st.write('Plaase select the options below to get a summary of the sresearch paper')

paper = st.selectbox('Select the Research Paper' , [
    "Attention Is All You Need",
    "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    "GPT: Improving Language Understanding by Generative Pre-Training",
    "Word2Vec: Efficient Estimation of Word Representations in Vector Space",
    "Deep Residual Learning for Image Recognition"
])

style = st.selectbox('select the summarization style' , ['Mathematical', 'Beginner Friendly' , 'Code Oriented' , 'Technical'])
length = st.selectbox('Enter the explaination length' , ['Long: detailed explanation' , 'Medium: 3-5 paragraphs ' , 'Short: 1-2 Paragraphs'])


but = st.button('Click here to get the Summary')

prompt_templete = PromptTemplate(template = '''
        Please summarize the research paper titled "{paper_input}" with the following specifications:
        Explanation Style: {style_input}
        Explanation Length: {length_input}

        Mathematical Details:
        Include relevant mathematical equations if present in the paper.
        Explain the mathematical concepts using simple, intuitive code snippets where applicable.
        Analogies:
        Use relatable analogies to simplify complex ideas.
        If certain information is not available in the paper, respond with: "Insufficient information available." Ensure the summary is clear, accurate, and aligned with the provided style and length.
''' , input_variables= ['paper_input' ,'style_input' , 'length_input'])

prompt = prompt_templete.invoke({
    'paper_input':paper,
    'style_input': style,
    'length_input': length
}
)

if(but):
    st.write('Hello')
    result = model.invoke(prompt)
    st.write(result.content)

