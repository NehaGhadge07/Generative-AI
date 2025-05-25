# import streamlit as st
# from openai import OpenAI


# st.title("🐋 Chat Deepseek 🐋")
# st.subheader("🪆 DeepSeek Chatbot Prototype 🪆", divider="blue")


# #   ========== Sidebar UI ==========
# st.sidebar.markdown("## Parameters")
# st.sidebar.divider()
# temp = st.sidebar.slider("Temperature", 0.0,1.0, value=0.5)


# # ===== API Client ======
# client = OpenAI(
#     base_url = "https://openrouter.ai/api/v1",
#     api_key=st.secrets.DEEPSEEK_API_KEY
# )


# # ====== Chat History =======
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []


# def render_chat_history_messages():
#     print(st.session_state.chat_history)

#     if len(st.session_state.chat_history) > 0:
#         for message in st.session_state.chat_history:
#             with st.chat_message(message["role"]):
#                 st.write(message["content"])

# render_chat_history_messages()



# if prompt := st.chat_input():
#     try:
#         # append the prompt to the chat history
#         st.session_state.chat_history.append(
#             {"role":"user","content": prompt}
#         )

#         #dissplay the user's prompt/message
#         with st.chat_message("user"):
#             st.markdown(prompt)


#         # display the llm message
#         with st.chat_message("assistant"):
#             # placeholder for the llm response
#             placeholder = st.empty()

#         chat_completion = client.chat.completions.create(
#             model="deepseek/deepseek-r1:free",
#             messages=[
#                 {"role" : "user", "content":"You are a professional chef. Generate a step-by-step recipe based on the ingredients provided by the user. Include cooking time, required utensils, and preparation tips."},
#             ] + st.session_state.chat_history,
#             stream=True,
#             temperature=temp
#         )

#         full_response =""

#         for chunk in chat_completion:
#             full_response += chunk.choices[0].delta.content or ""
#             placeholder.write(full_response)
#         st.session_state.chat_history(
#             {"role":"assistant", "content":full_response}
#         )



#     except Exception as e:
#         print("ERROR :", e)

# import streamlit as st
# from openai import OpenAI

# st.title("👨‍🍳 ChatCookBot 👨‍🍳")
# st.subheader("🍲 Your Personal Cooking Assistant 🍲", divider="orange")

# # Sidebar - Temperature slider
# st.sidebar.markdown("## Parameters")
# st.sidebar.divider()
# temp = st.sidebar.slider("Temperature", 0.0, 1.0, value=0.5)

# # API Client for DeepSeek
# client = OpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key=st.secrets.DEEPSEEK_API_KEY
# )

# # Chat history initialization
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # Render chat history
# def render_chat_history_messages():
#     if st.session_state.chat_history:
#         for message in st.session_state.chat_history:
#             with st.chat_message(message["role"]):
#                 st.write(message["content"])

# render_chat_history_messages()

# # Cook-only system prompt
# system_prompt = {
#     "role": "system",
#     "content": (
#         "You are a professional chef and cooking assistant. "
#         "Only respond to queries related to cooking, recipes, food preparation, or kitchen tips. "
#         "If the user asks anything unrelated to cooking, politely inform them that you only assist with cooking."
#     )
# }


# # Handle user input
# if prompt := st.chat_input("Ask a cooking question..."):
#     try:
#         # Add user's message to chat history
#         st.session_state.chat_history.append({"role": "user", "content": prompt})

#         # Display user's message
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         # Assistant response placeholder
#         with st.chat_message("assistant"):
#             placeholder = st.empty()

#         # Get response from DeepSeek
#         chat_completion = client.chat.completions.create(
#             model="deepseek/deepseek-r1:free",
#             messages=[system_prompt] + st.session_state.chat_history,
#             stream=True,
#             temperature=temp
#         )

#         full_response = ""
#         for chunk in chat_completion:
#             full_response += chunk.choices[0].delta.content or ""
#             placeholder.write(full_response)

#         # Append assistant response to chat history
#         st.session_state.chat_history.append(
#             {"role": "assistant", "content": full_response}
#         )

#     except Exception as e:
#         st.error(f"Error: {e}")


# import streamlit as st
# import pandas as pd
# from openai import OpenAI
# from sentence_transformers import SentenceTransformer
# import faiss
# import numpy as np

# # ========== Load Dataset ==========
# @st.cache_data
# def load_and_embed_data(file_path="13k-recipes.csv"):
#     df = pd.read_csv(file_path)
#     texts = df["Cleaned_Ingredients"].tolist()

#     # Create embeddings
#     model = SentenceTransformer("all-MiniLM-L6-v2")
#     embeddings = model.encode(texts)

#     # Store in FAISS vector database
#     dim = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dim)
#     index.add(np.array(embeddings))

#     return df, index, embeddings, model

# df, index, all_embeddings, embed_model = load_and_embed_data()

# # ========== Streamlit UI ==========
# st.title("👨‍🍳 ChatCookBot with RAG")
# st.subheader("📚 Now Enhanced with Your Recipe Dataset!", divider="orange")
# st.sidebar.markdown("## Parameters")
# temp = st.sidebar.slider("Temperature", 0.0, 1.0, value=0.5)

# # ========== API Client ==========
# client = OpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key=st.secrets.DEEPSEEK_API_KEY
# )

# # ========== Session State ==========
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # ========== Retrieve Relevant Recipe ==========
# def retrieve_relevant_chunks(query, k=3):
#     query_embedding = embed_model.encode([query])
#     distances, indices = index.search(np.array(query_embedding), k)
#     results = df.iloc[indices[0]]["Cleaned_Ingredients"].tolist()
#     return "\n\n".join(results)

# # ========== System Prompt ==========
# system_prompt = {
#     "role": "system",
#     "content": (
#         "You are a professional chef and cooking assistant. "
#         "Use the provided recipe database to help users. "
#         "Only respond to cooking-related queries."
#     )
# }

# # ========== Display Chat ==========
# def render_chat_history():
#     for message in st.session_state.chat_history:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

# render_chat_history()

# # ========== Handle User Input ==========
# if prompt := st.chat_input("Ask about a recipe..."):
#     try:
#         st.session_state.chat_history.append({"role": "user", "content": prompt})

#         with st.chat_message("user"):
#             st.markdown(prompt)

#         with st.chat_message("assistant"):
#             placeholder = st.empty()

#         # RAG: Retrieve top relevant recipes
#         retrieved_context = retrieve_relevant_chunks(prompt)

#         full_prompt = (
#             f"User question: {prompt}\n\n"
#             f"Here are some relevant recipes from our dataset:\n{retrieved_context}"
#         )

#         chat_completion = client.chat.completions.create(
#             model="deepseek/deepseek-r1:free",
#             messages=[system_prompt] + [{"role": "user", "content": full_prompt}],
#             stream=True,
#             temperature=temp
#         )

#         full_response = ""
#         for chunk in chat_completion:
#             full_response += chunk.choices[0].delta.content or ""
#             placeholder.markdown(full_response)

#         st.session_state.chat_history.append({"role": "assistant", "content": full_response})

#     except Exception as e:
#         st.error(f"Error: {e}")

# app.py
import streamlit as st
import pickle
import faiss
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

# === Load FAISS & Embeddings ===
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
FAISS_INDEX_FILE = "recipe_faiss.index"
DOCS_FILE = "docs.pkl"
TOP_K = 3 # top documents/text extracted

@st.cache_resource
def load_vector_store():
    index = faiss.read_index(FAISS_INDEX_FILE)
    with open(DOCS_FILE, "rb") as f:
        docs = pickle.load(f)
    model = SentenceTransformer(EMBEDDING_MODEL)
    return index, docs, model

index, docs, embed_model = load_vector_store()

# === Streamlit UI ===
st.title("👨‍🍳 ChatCookBot with RAG")
st.subheader("Enhanced with Your Recipe Dataset", divider="orange")
temp = st.sidebar.slider("Temperature", 0.0, 1.0, value=0.5)

# === OpenAI Client ===
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=st.secrets.DEEPSEEK_API_KEY
)

# === Chat History ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def render_chat():
    for m in st.session_state.chat_history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

render_chat()

# === RAG: Retrieve Relevant Recipes ===
def retrieve_context(query):
    query_embed = embed_model.encode([query])
    distances, indices = index.search(np.array(query_embed), TOP_K)

    # Check if it's unrelated (distance too high)
    if distances[0][0] > 1.0:  # You can tweak this threshold
        return None

    results = [docs[i] for i in indices[0]]
    return "\n\n".join(results)
    


# === System Prompt ===
system_prompt = {
    "role": "Master Chef",
    "content": (
        "You are a professional chef and recipe assistant. "
        "You respond **only** to queries about cooking, recipes, ingredients, and kitchen-related advice. "
        "If a user asks anything unrelated (like cars, technology, movies), politely decline and explain your purpose. "
        "However, for polite greetings like 'hi', 'hello', 'how are you', respond warmly before inviting them to ask a cooking-related question."
    )
}

# Keyword list to check if the user question is cooking-related
COOKING_KEYWORDS = [
    "cook", "recipe", "ingredients", "bake", "boil", "roast", "fry", "grill", 
    "vegetables", "chicken", "meat", "dish", "meal", "food", "nutrition", "kitchen",
    "dessert", "bread", "cake", "snack", "dinner", "lunch", "breakfast"
]

def is_cooking_related(query):
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in COOKING_KEYWORDS)


# === Handle User Input ===
if prompt := st.chat_input("Ask a cooking question..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        
    # === Greeting Handler ===
    greetings = ["hi", "hello", "hey", "how are you", "good morning", "good evening"]
    if prompt.lower().strip() in greetings:
        response = "Hello! 👋 I'm your cooking assistant. Feel free to ask me about any recipe, ingredient, or kitchen tip! 🍳"
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        placeholder.markdown(response)
        st.stop()

    # Check if the question is cooking-related
    if not is_cooking_related(prompt):
        response = "Sorry, I can only assist with cooking, recipes, and food-related topics. 🍽️"
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
    else:
        # Retrieve relevant context
        context = retrieve_context(prompt)
        if context is None:
            response = "I couldn’t find anything related to cooking for that question. I specialize only in food and recipes."
            with st.chat_message("assistant"):
                st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        else:
            # Proceed with RAG + LLM response
            final_prompt = f"User asked: {prompt}\n\nRelevant recipes:\n{context}"
            with st.chat_message("assistant"):
                placeholder = st.empty()
            
            chat_completion = client.chat.completions.create(
                model="deepseek/deepseek-r1:free",
                messages=[system_prompt, {"role": "user", "content": final_prompt}],
                stream=True,
                temperature=temp
            )

            full_response = ""
            for chunk in chat_completion:
                full_response += chunk.choices[0].delta.content or ""
                placeholder.markdown(full_response)

            st.session_state.chat_history.append({"role": "assistant", "content": full_response})
