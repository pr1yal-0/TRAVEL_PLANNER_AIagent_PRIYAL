# app.py

import streamlit as st
from destination_utils import create_vector_store, load_vector_store
from planner_agent import build_graph
import os

st.set_page_config(page_title="Travel Planner AI", page_icon="🌍")

st.title("🌍 Travel Planner AI")
st.write("Describe your dream trip and get a full itinerary with weather + attractions.")

# Step 1: Load or build vectorstore
if not os.path.exists("vectorstore.pkl"):
    st.warning("🔄 Creating vectorstore from destinations.csv...")
    create_vector_store()

vectorstore = load_vector_store()
graph = build_graph(vectorstore)

# Step 2: User Input
query = st.text_input("🧳 Describe your ideal vacation:")

if query:
    with st.spinner("✈️ Planning your trip..."):
        try:
            # ✅ FIXED: Proper format for LangGraph
            result = graph.invoke({
                "question": query,
                "vectorstore": vectorstore
            })

            # ✅ FIXED: match key to planner_agent.py output
            st.subheader("🔮 Itinerary")
            st.text(result["plan"])

        except Exception as e:
            st.error(f"Something went wrong: {e}")
