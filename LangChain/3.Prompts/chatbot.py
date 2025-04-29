import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
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

while True:
    user_input = input("YOU :" )
    if user_input == "exit":
        break
    result = model.invoke(user_input)
    print("AI :", result.content)