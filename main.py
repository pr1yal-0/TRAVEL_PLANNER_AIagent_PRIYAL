import os
import torch
from destination_utils import create_vector_store, load_vector_store
from planner_agent import build_graph

VECTORSTORE_PATH = "vectorstore.pkl"

def main():
    print("📟 Using device:", "cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(VECTORSTORE_PATH):
        print("📁 Vectorstore missing. Creating it from destinations.csv...")
        create_vector_store()

    vectorstore = load_vector_store()

    question = input("🧳 Describe your ideal vacation (e.g. '5-day temple trip to Jaipur under $1000'):\n> ").strip()
    if not question:
        print("❌ Error: You must enter a query.")
        return

    graph = build_graph(vectorstore)
    output = graph.invoke({
        "question": question,
        "vectorstore": vectorstore
    })

    print("\n📋 Your custom itinerary:")
    print(output.get("plan", "⚠️ Could not generate a plan."))

if __name__ == "__main__":
    main()
