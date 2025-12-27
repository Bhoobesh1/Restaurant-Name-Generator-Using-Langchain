import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

# ---------------- UI ----------------
st.set_page_config(page_title="Restaurant Name Generator", layout="centered")
st.title("🍽️ Restaurant Name Generator")

cuisine = st.sidebar.text_input("Enter the Cuisine",placeholder="Enter the Cuisine")

generate = st.sidebar.button(
    "Generate",
    key="generate_btn"
)


# ---------------- LLM ----------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)

# ---------------- Prompts ----------------
prompt_template_name = PromptTemplate(
    input_variables=["cuisine"],
    template="I want to open a restaurant for {cuisine} food. Tell me a only one fancy restaurant name. Do NOT add explanations, headings, or extra text "
)

prompt_template_items = PromptTemplate(
    input_variables=["restaurant_name"],
    template = """
Generate exactly 10 menu items for the restaurant "{restaurant_name}".

Rules:
- Number each item clearly using this format only:
1. Item name
2. Item name
3. Item name
- Put each item on a new line
- Do NOT add explanations, headings, or extra text
- Do NOT use bullet points or emojis

Return only the numbered list.
"""
)

# ---------------- Sequential Chain ----------------
def sequential_chain(inputs):
    cuisine = inputs["cuisine"]

    restaurant_name = llm.invoke(
        prompt_template_name.format(cuisine=cuisine)
    ).content.strip()

    menu_items = llm.invoke(
        prompt_template_items.format(restaurant_name=restaurant_name)
    ).content.strip()

    return {
        "restaurant_name": restaurant_name,
        "menu_items": menu_items
    }

chain = RunnableLambda(sequential_chain)

# ---------------- Run App ----------------
if generate:
    with st.spinner("Generating restaurant details..."):
        response = chain.invoke({"cuisine": cuisine})

    st.success("Done!")

    st.header(response["restaurant_name"])
    st.subheader("📜 Menu Items")
    menu_list = response["menu_items"].split("\n")
    for item in menu_list:
        st.write(item)
