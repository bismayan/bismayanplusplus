import os

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory


def set_openai_key(ret_value: bool = False):
    """Reads the Open AI API Key and sets the env variable OPENAI_API_KEY with the value.
    Also optionally return the value if the ret_value arg is true. 

    Args:
        ret_value (bool, optional): Whether to return the key. Defaults to False.

    Returns:
        str: Value of API Key if ret_value is True else None
    """
    if os.environ.get("OPENAI_API_KEY") is None:
        try:
            from openai_key import api_key
            os.environ["OPENAI_API_KEY"] = api_key
        except:
            raise KeyError("OPENAI_API_KEY is not defined")
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
                                     template="Write me a short 300 word Youtube Video script based on the title TITLE: {title}")
    return script_template, title_template


def initialize_chain(title_template, script_template, memory):
    """Returns the final Langchain Sequential Chain

    Args:
        title_template (PromptTemplate): Prompt to pass to Title Chain
        script_template (PromptTemplate): Prompt to pass to Script Chain

    Returns:
        SimpleSequentialChain: Returns the FInal LLM Chain with the Two LLMs chained
    """
    llm = OpenAI(temperature=0.9)
    title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key="title", memory=memory)
    script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key="script", memory=memory)
    final_chain = SequentialChain(chains=[title_chain, script_chain], input_variables=['topic'], 
                                  output_variables=['title', 'script'], verbose=True)
    return final_chain


def main():
    script_template, title_template = initialize_templates()
    memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
    final_chain = initialize_chain(title_template, script_template, memory)
    
    st.title("🤖 🤖 Welcome to Bismayan++!")
    pr = st.text_input(
        "Unlike my creator, I am somewhat intelligent. What would you like to get the script for?", value="")
    if pr:
        with st.spinner(text="Working on it..."):
            response = final_chain({"topic":pr})
            st.write(response['title'])
            st.write(response['script'])

            with st.expander("Message History"):
                st.info(memory.buffer)


if __name__ == "__main__":
    set_openai_key()
    main()
