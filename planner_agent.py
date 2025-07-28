from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFacePipeline
from langgraph.graph import StateGraph, END
from langchain.schema.runnable import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from transformers import pipeline
from typing import TypedDict
import torch

# ✅ Use flan-t5-small in text2text mode
pipe = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    device=0 if torch.cuda.is_available() else -1,
    max_new_tokens=300,
)

llm = HuggingFacePipeline(pipeline=pipe)

# ✅ Prompt Template for flan-t5-small
prompt = PromptTemplate.from_template(
    "You are a travel assistant.\nGiven this destination information:\n\n{answer}\n\nCreate a 3-day travel itinerary."
)

# ✅ Updated chaining (no LLMChain, no warning)
llm_chain = prompt | llm | StrOutputParser()

# ✅ LangGraph state definition
class GraphState(TypedDict):
    question: str
    vectorstore: object
    answer: str
    plan: str

# 🔍 Search destinations
def search_destination_info(state: GraphState) -> GraphState:
    query = state["question"]
    vectorstore = state["vectorstore"]
    docs = vectorstore.similarity_search(query, k=5)
    answer = "\n".join(doc.page_content for doc in docs)
    return {**state, "answer": answer}

# 🧠 Generate itinerary
def generate_plan(state: GraphState) -> GraphState:
    plan = llm_chain.invoke({"answer": state["answer"]})
    return {**state, "plan": plan}

# 🔧 Build the LangGraph
def build_graph(vectorstore):
    builder = StateGraph(GraphState)

    builder.add_node("search_info", RunnableLambda(search_destination_info))
    builder.add_node("generate_plan", RunnableLambda(generate_plan))

    builder.set_entry_point("search_info")
    builder.add_edge("search_info", "generate_plan")
    builder.add_edge("generate_plan", END)

    return builder.compile()
