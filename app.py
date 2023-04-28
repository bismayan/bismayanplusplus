import os
from openai_key import api_key
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain


def set_openai_key(ret_value: bool = False):
    """Reads the Open AI API Key and sets the env variable OPENAI_API_KEY with the value.
    Also optionally return the value if the ret_value arg is true. 

    Args:
        ret_value (bool, optional): Whether to return the key. Defaults to False.

    Returns:
        str: Value of API Key if ret_value is True else None
    """
    os.environ["OPENAI_API_KEY"] = api_key
    if ret_value:
        return api_key
    else:
        return


def initialize_templates():
    """Initializes title and script templates

    Returns:
        langchain.prompts.PromptTemplate, langchain.prompts.PromptTemplate: Both the script and title templates
    """
    title_template = PromptTemplate(input_variables=['topic'],
                                    template="Write me a Youtube video title about {topic}")
    script_template = PromptTemplate(input_variables=['title'],
                                     template="Write me a Youtube Video script based on the title TITLE: {title}")
    return script_template, title_template


def initialize_chain(title_template, script_template):
    """Returns the final Langchain Sequential Chain

    Args:
        title_template (PromptTemplate): Prompt to pass to Title Chain
        script_template (PromptTemplate): Prompt to pass to Script Chain

    Returns:
        SimpleSequentialChain: Returns the FInal LLM Chain with the Two LLMs chained
    """
    llm = OpenAI(temperature=0.9)
    title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)
    script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True)
    final_chain = SimpleSequentialChain(
        chains=[title_chain, script_chain], verbose=True)
    return final_chain


def main():
    script_template, title_template = initialize_templates()
    final_chain = initialize_chain(title_template, script_template)
    st.title("🤖 🤖 Welcome to Bismayan++!")
    pr = st.text_input(
        "Unlike my creator, I am somewhat intelligent. What would you like to get the script for?", value="")
    if pr:
        response = final_chain.run(pr)
        st.write(response)


if __name__ == "__main__":
    set_openai_key()
    main()
