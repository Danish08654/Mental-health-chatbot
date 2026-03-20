import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ---------------------------
# Load model (cached)
# ---------------------------
model_path = "model/mental_model"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.7,
        top_k=50
    )

chatbot = load_model()

# ---------------------------
# UI
# ---------------------------
st.title("🧠 Mental Health Support Chatbot")
st.write("Talk about stress, anxiety, or emotions 💬")

# ---------------------------
# Chat history (IMPORTANT)
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"**🧑 You:** {msg['content']}")
    else:
        st.markdown(f"**🤖 Bot:** {msg['content']}")

# ---------------------------
# Input box
# ---------------------------
user_input = st.text_input("How are you feeling?")

# ---------------------------
# Send button
# ---------------------------
if st.button("Send"):
    if user_input.strip() != "":
        
        # Save user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.spinner("Thinking... 🤔"):
            
            # Better prompt format
            prompt = f"User: {user_input}\nAssistant:"

            result = chatbot(prompt)
            response = result[0]["generated_text"]

            # Clean output
            clean_response = response.replace(prompt, "").strip()

        # Save bot response
        st.session_state.messages.append({
            "role": "bot",
            "content": clean_response
        })

        # Rerun to update UI
        st.rerun()