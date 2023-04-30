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
    """Initializes title and tut templates

    Returns:
        langchain.prompts.PromptTemplate, langchain.prompts.PromptTemplate: Both the tut and title templates
    """
    title_template = PromptTemplate(input_variables=['topic'],
                                    template="Write me a article title about {topic}")
    tut_template = PromptTemplate(input_variables=['title', 'wikipedia_research'],
                                     template="Write me a 500 word article based on the title TITLE: {title}, \
                                        while also using this wikipedia research: {wikipedia_research}")
    tut_template2 = PromptTemplate(input_variables=['title', 'wikipedia_research', 'tut1'],
                                     template="Complete the 500 word article based on the title TITLE: {title}, \
                                        while also using this wikipedia research: {wikipedia_research} and the previous output {tut1}")
    return title_template, tut_template, tut_template2


def initialize_chain(temp,title_template, tut_template, tut_template2, title_memory, tut_memory, tut_memory2):
    """Returns the final Langchain Sequential Chain

    Args:
        title_template (PromptTemplate): Prompt to pass to Title Chain
        tut_template (PromptTemplate): Prompt to pass to tut Chain

    Returns:
        SimpleSequentialChain: Returns the FInal LLM Chain with the Two LLMs chained
    """
    llm = OpenAI(temperature=temp)
    title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key="title", memory=title_memory)
    tut_chain = LLMChain(llm=llm, prompt=tut_template, verbose=True, output_key="tut1", memory=tut_memory)
    tut_chain_2 = LLMChain(llm=llm, prompt=tut_template2, verbose=True, output_key="tut2", memory=tut_memory2)
    return title_chain, tut_chain, tut_chain_2


def main():
    
    st.title("🤖 🤖 Welcome to Bismayan++! The (sometimes) Amazing Teacher Bot")
    
    title_template, tut_template, tut_template2 = initialize_templates()
    title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
    tut_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')
    tut_memory2 = ConversationBufferMemory(input_key='tut1', memory_key='chat_history')
    temp= st.slider("Creativity", min_value=0.1, max_value=0.95, step=0.05, value=0.8)
    title_chain, tut_chain, tut_chain2 = initialize_chain(temp,title_template, tut_template, tut_template2, 
                                                 title_memory, tut_memory, tut_memory2)
    wiki= WikipediaAPIWrapper()

    
    pr = st.text_input(
         "What would you like to Learn Today?", value="")
    if pr:
        with st.spinner(text="Working on it..."):
            title = title_chain.run(pr)
            wiki_research= wiki.run(pr)
            tut1= tut_chain.run(title=title, wikipedia_research= wiki_research)
            tut2= tut_chain2.run(title=title, wikipedia_research= wiki_research, tut1= tut1)
            st.write(title)
            st.write( " ". join([tut1,tut2]))

            with st.expander("Title History"):
                st.info(title_memory.buffer)
            with st.expander("Tutorial History 1"):
                st.info(tut_memory.buffer)
            with st.expander("Tutorial History 2"):
                st.info(tut_memory2.buffer)    
            with st.expander("Wikipedia Research"):
                st.info(wiki_research)



if __name__ == "__main__":
    set_openai_key()
    main()
