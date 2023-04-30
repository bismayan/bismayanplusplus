import os

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

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
    script_template = PromptTemplate(input_variables=['title', 'wikipedia_research'],
                                     template="Write me a short 300 word Youtube Video script based on the title TITLE: {title}, \
                                        while also using this wikipedia research: {wikipedia_research}")
    return script_template, title_template


def initialize_chain(title_template, script_template, t_memory, s_memory):
    """Returns the final Langchain Sequential Chain

    Args:
        title_template (PromptTemplate): Prompt to pass to Title Chain
        script_template (PromptTemplate): Prompt to pass to Script Chain

    Returns:
        SimpleSequentialChain: Returns the FInal LLM Chain with the Two LLMs chained
    """
    llm = OpenAI(temperature=0.9)
    title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key="title", memory=t_memory)
    script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key="script", memory=s_memory)
    return title_chain, script_chain


def main():
    script_template, title_template = initialize_templates()
    title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
    script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')
    title_chain, script_chain = initialize_chain(title_template, script_template, title_memory, script_memory)
    wiki= WikipediaAPIWrapper()


    st.title("🤖 🤖 Welcome to Bismayan++!")
    pr = st.text_input(
        "Unlike my creator, I am somewhat intelligent. What would you like to get the script for?", value="")
    if pr:
        with st.spinner(text="Working on it..."):
            title = title_chain.run(pr)
            wiki_research= wiki.run(pr)
            script= script_chain.run(title=title, wikipedia_research= wiki_research)
            st.write(title)
            st.write(script)

            with st.expander("Title History"):
                st.info(title_memory.buffer)
            with st.expander("Script History"):
                st.info(script_memory.buffer)
            with st.expander("Wikipedia Research"):
                st.info(wiki_research)



if __name__ == "__main__":
    set_openai_key()
    main()
