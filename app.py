#!/usr/bin/env python3

import torch
import os
import argparse
import time
import streamlit as st

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All

load_dotenv()

embeddings_model = os.environ.get("EMBEDDINGS_MODEL")
db_root = os.environ.get('DB_ROOT')
model_root = os.environ.get('MODEL_ROOT')
n_ctx = os.environ.get('N_CTX')
chunks = int(os.environ.get('CHUNKS',4))

from vectordb import CLIENT_SETTINGS


st.set_page_config(page_title='Java & Finance Assistant', layout='wide')
@st.cache_resource
def parse_arguments():
    print("Parsing Arguments")
    parser = argparse.ArgumentParser(description='Ask questions to your documents, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


@st.cache_resource
def load_model():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # device = "gpu" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    print(f"Using device: {device}")
    #embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name, model_kwargs={"device": device})
    embeddings = HuggingFaceInstructEmbeddings(model_name=embeddings_model, model_kwargs={"device": device})
    db = Chroma(persist_directory=db_root, embedding_function=embeddings, client_settings=CLIENT_SETTINGS)

    retriever = db.as_retriever(search_kwargs={"k": chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # Prepare the LLM
    llm = GPT4All(model=model_root, n_ctx=n_ctx, callbacks=callbacks, verbose=False)

    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents= not args.hide_source)
    return qa
    

args = parse_arguments()
qa = load_model()

# Define function to get user input
def get_text():
    """
    Get the user input text.
    Returns:
        (str): The text entered by the user
    """
    input_text = st.text_input("You: ", st.session_state["input"], key="input",
                            placeholder="Your Document assistant ! Ask me anything ...", 
                            label_visibility='hidden')
    return input_text

def time_convert(sec):
  mins = sec // 60
  sec = sec % 60
  hours = mins // 60
  mins = mins % 60
  print("Time Lapsed = {0}:{1}:{2}".format(int(hours),int(mins),sec))

if "generated" not in st.session_state:
    st.session_state["generated"] = []
if "sources" not in st.session_state:
    st.session_state["sources"] = []
if "past" not in st.session_state:
    st.session_state["past"] = []
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "stored_session" not in st.session_state:
    st.session_state["stored_session"] = []

# Define function to start a new chat
def new_query():
    """
    Clears session state and starts a new chat.
    """
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i]) 
        save.append("Bot:" + st.session_state["sources"][i])        
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["sources"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""


prompt = get_text()
# Get the answer from the chain
if prompt:
    start_time = time.time()
    res = qa(prompt)
    answer, docs = res['result'], [] if args.hide_source else res['source_documents']
    end_time = time.time()
    time_lapsed = end_time - start_time
    time_convert(time_lapsed)
    st.session_state.past.append(prompt)  
    st.session_state.generated.append(answer) 
    st.session_state.sources.append(docs) 
    print(docs)

# Allow to download as well
download_str = []
# Display the conversation history using an expander, and allow the user to download it
with st.expander("Results", expanded=True):
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        st.info(st.session_state["past"][i])
        st.success(st.session_state["generated"][i],)
        download_str.append(st.session_state["past"][i])
        download_str.append(st.session_state["generated"][i])
    
    # Can throw error - requires fix
    download_str = '\n'.join(download_str)
    if download_str:
        st.download_button('Download',download_str)

# Display stored conversation sessions in the sidebar
for i, sublist in enumerate(st.session_state.stored_session):
        with st.sidebar.expander(label= f"Conversation-Session:{i}"):
            st.write(sublist)

# Allow the user to clear all stored conversation sessions
if st.session_state.stored_session:   
    if st.sidebar.checkbox("Clear-all"):
        del st.session_state.stored_session

st.markdown("____")
st.markdown("")