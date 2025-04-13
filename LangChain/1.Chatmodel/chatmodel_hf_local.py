from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

print("⏳ Loading model...")

llm = HuggingFacePipeline.from_model_id(
    model_id='google/flan-t5-small',
    task='text2text-generation',  # important for flan-t5
    pipeline_kwargs=dict(
        temperature=0.5,
        do_sample=True, 
        max_new_tokens=100
    )
)
# model = ChatHuggingFace(llm=llm)
result = llm.invoke("Answer the following question concisely:\nWhat is the capital of Bangladesh?")
print(result)



