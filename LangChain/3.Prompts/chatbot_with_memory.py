import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate, load_prompt


load_dotenv()



llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # repo_id = "facebook/blenderbot-400M-distill",
    # repo_id = "meta-llama/Llama-3.3-70B-Instruct",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

chat_history = [
    SystemMessage(content='You are a helpful AI assistant')
]

while True:
    user_input = input("YOU :" )
    chat_history.append(HumanMessage(content=user_input))
    if user_input == "exit":
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print("AI :", result.content)

print(chat_history)    