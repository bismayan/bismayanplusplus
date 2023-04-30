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
    """Initializes title and article templates

    Returns:
        langchain.prompts.PromptTemplate, langchain.prompts.PromptTemplate: Both the article and title templates
    """
    title_template = PromptTemplate(input_variables=['topic'],
                                    template="Write me a article title about {topic}")
    article_template = PromptTemplate(input_variables=['title', 'wikipedia_research'],
                                     template="Start a long tutorial based on the title: {title}, \
                                        while also using this wikipedia research: {wikipedia_research}.")
    article_template2 = PromptTemplate(input_variables=['title', 'wikipedia_research', 'article1'],
                                     template="Complete the tutorial based on the title: {title}, \
                                        while also using this wikipedia research: {wikipedia_research} and the previous output: {article1}")
    return title_template, article_template, article_template2


def initialize_chain(temp,title_template, article_template, article_template2, title_memory, article_memory, article_memory2):
    """Returns the final Langchain Sequential Chain

    Args:
        title_template (PromptTemplate): Prompt to pass to Title Chain
        article_template (PromptTemplate): Prompt to pass to article Chain

    Returns:
        SimpleSequentialChain: Returns the FInal LLM Chain with the Two LLMs chained
    """
    llm = OpenAI(temperature=temp)
    title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key="title", memory=title_memory)
    article_chain = LLMChain(llm=llm, prompt=article_template, verbose=True, output_key="article1", memory=article_memory)
    article_chain_2 = LLMChain(llm=llm, prompt=article_template2, verbose=True, output_key="article2", memory=article_memory2)
    return title_chain, article_chain, article_chain_2


def main():
    
    st.set_page_config(
        page_title="Bismayan++",
        page_icon="🤖🤖",
        layout="wide",
        initial_sidebar_state="expanded")
    st.markdown("<h1 style='text-align: center; color: white;'>🤖🤖 Welcome to Bismayan++ </h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: white;'>The (sometimes) Amazing Tutorial Generating Bot </h3>", unsafe_allow_html=True)
    title_template, article_template, article_template2 = initialize_templates()
    title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
    article_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')
    article_memory2 = ConversationBufferMemory(input_key='article1', memory_key='chat_history')
    col1, _, col3 = st.columns([2,1,1])
    with col3:
        temp= st.slider("Creativity", min_value=0.1, max_value=0.95, step=0.05, value=0.9)
    with col1:
        pr = st.text_input(
            "What would you like to generate a Tutorial about Today?", value="")
    title_chain, article_chain, article_chain2 = initialize_chain(temp,title_template, article_template, article_template2, 
                                                 title_memory, article_memory, article_memory2)
    wiki= WikipediaAPIWrapper()
       
    if pr:
        with st.spinner(text="Working on it..."):
            with st.container():    
                title = title_chain.run(pr)
                wiki_research= wiki.run(pr)
                article1= article_chain.run(title=title, wikipedia_research= wiki_research)
                article2= article_chain2.run(title=title, wikipedia_research= wiki_research, article1= article1)
                st.subheader(title)
                st.write( " ". join([article1,article2]))

            with st.expander("Title History"):
                st.info(title_memory.buffer)
            with st.expander("Article History 1"):
                st.info(article_memory.buffer)
            with st.expander("Article History 2"):
                st.info(article_memory2.buffer)    
            with st.expander("Wikipedia Research"):
                st.info(wiki_research)



if __name__ == "__main__":
    set_openai_key()
    main()
