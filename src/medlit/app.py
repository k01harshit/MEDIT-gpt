import streamlit as st
from src.medlit.agent.graph import build_graph

# Configure the web page
st.set_page_config(
    page_title="MedLit Copilot", 
    page_icon="🧬", 
    layout="centered"
)

st.title("🧬 MedLit Copilot")
st.markdown("**Evidence-based Pharmacovigilance RAG Assistant**")
st.markdown("Answers are strictly grounded in indexed PubMed literature.")
st.divider()

# Initialize the LangGraph agent only once per session
if "agent" not in st.session_state:
    with st.spinner("Initializing models and loading databases... This takes a moment."):
        st.session_state.agent = build_graph()
        
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am MedLit Copilot. Ask me a pharmacovigilance question, and I will search the local PubMed index for evidence."}
    ]

# Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Accept user input
if user_query := st.chat_input("E.g., Does metformin increase lactic acidosis risk?"):
    
    # Display the user's question
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    # Generate the assistant's answer
    with st.chat_message("assistant"):
        with st.spinner("Retrieving chunks, reranking, and generating..."):
            # Invoke the LangGraph workflow
            final_state = st.session_state.agent.invoke({"query": user_query})
            answer = final_state["answer"]
            
            # Display the answer
            st.markdown(answer)
            
    # Save the assistant's answer to history
    st.session_state.messages.append({"role": "assistant", "content": answer})