# openvino-rag-server.py

import os
import time
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
#from PyPDF2 import PdfReader 
from fastapi.responses import JSONResponse



import logging
from langchain import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.vectorstores import Chroma
#from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from transformers import pipeline
from transformers import AutoTokenizer, pipeline
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import FAISS
from langchain.vectorstores import Chroma
from pydantic import BaseModel
import os
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceBgeEmbeddings
#from langchain.chains.summarize import load_summarize_chain
#from langchain.chains.combine_documents.stuff import StuffDocumentsChain
#from langchain.chains.llm import LLMChain
#from langchain.prompts import PromptTemplate
#from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceBgeEmbeddings

#from googletrans import Translator

from optimum.intel.openvino import OVModelForCausalLM


logging.basicConfig(level=logging.DEBUG) 
# from transformers.utils import logging

# logging.set_verbosity_info()
# logger = logging.get_logger("transformers")
# logger.info('INFO')
# logger.warning("WARN")
load_dotenv(verbose=True)
cache_dir         = os.environ['CACHE_DIR']
model_vendor      = os.environ['MODEL_VENDOR']
model_name        = os.environ['MODEL_NAME']
model_precision   = os.environ['MODEL_PRECISION']
inference_device  = os.environ['INFERENCE_DEVICE']
document_dir      = os.environ['DOCUMENT_DIR']
#vectorstore_dir   = os.environ['VECTOR_DB_DIR']
#vectorstore_dir ="stores/pet_cosine"
vector_db_postfix = os.environ['VECTOR_DB_POSTFIX']
num_max_tokens    = int(os.environ['NUM_MAX_TOKENS'])
embeddings_model  = os.environ['MODEL_EMBEDDINGS']
rag_chain_type    = os.environ['RAG_CHAIN_TYPE']
ov_config         = {"PERFORMANCE_HINT":"LATENCY", "NUM_STREAMS":"1", "CACHE_DIR":cache_dir}

### WORKAROUND for "trust_remote_code=True is required error" in HuggingFaceEmbeddings()
#from transformers import AutoModel
#model = AutoModel.from_pretrained(embeddings_model, trust_remote_code=True) 

# model_name_embed = "jinaai/jina-embeddings-v2-base-en"
# model_kwargs = {"device": "cpu"}
# encode_kwargs = {"normalize_embeddings": False}
# embeddings = HuggingFaceBgeEmbeddings(
#     model_name=model_name_embed, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
# )
# prompt_template = """Use the following pieces of information to answer the user's question.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.

# Context: {context}
# Question: {question}

# Only return the helpful answer below and nothing else.
# Helpful answer:
# """

# prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

# chain_type_kwargs = {"prompt": prompt}
# embeddings = HuggingFaceEmbeddings(
#     model_name = embeddings_model,
#     model_kwargs = {'device':'cpu'},
#     encode_kwargs = {'normalize_embeddings':True}
# )

#vectorstore_dir = f'{vectorstore_dir}_{vector_db_postfix}'
# vectorstore = Chroma(persist_directory=vectorstore_dir, embedding_function=embeddings)
# retriever = vectorstore.as_retriever()
#    search_type='similarity_score_threshold', 
#    search_kwargs={
#        'score_threshold' : 0.8, 
#        'k' : 4
#    }
#)
# print(f'** Vector store : {vectorstore_dir}')

# retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# query = "In the year 1941, Ram Piari died at Mayo Hospi tal at Lahore"
# semantic_search = retriever.get_relevant_documents(query)
# print(semantic_search)
# logging.debug(semantic_search)

# model_id = f'{model_vendor}/{model_name}'
# tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
# ov_model_path = f'./{model_name}/{model_precision}'
# model = OVModelForCausalLM.from_pretrained(model_id=ov_model_path, device=inference_device, ov_config=ov_config, cache_dir=cache_dir)
# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=num_max_tokens)
# llm = HuggingFacePipeline(pipeline=pipe)
# qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type=rag_chain_type, retriever=retriever, chain_type_kwargs=chain_type_kwargs,verbose=True)



class CaseId(BaseModel):
    caseId: str


class Query(BaseModel):
    caseId: str
    query: str


def run_generation(text_user_en, qa_chain):
    ans = qa_chain.run(text_user_en)
    logging.debug(f"Answer: {ans}") 
    return ans

app = FastAPI()

origins = [  
    "http://localhost:59507",  # Angular app  
    "http://localhost:8000",   # FastAPI server  
    "http://localhost",          
    "http://localhost:8080",   
     "http://localhost:4200",
]  
  
app.add_middleware(  
    CORSMiddleware,  
    #allow_origins=origins,  
    allow_credentials=True,  
    allow_methods=["*"],  
    allow_headers=["*"],
    allow_origins=['*']  
)
model_id = f'{model_vendor}/{model_name}' 
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
ov_model_path = f'./{model_name}/{model_precision}'
model = OVModelForCausalLM.from_pretrained(model_id=ov_model_path, device=inference_device, ov_config=ov_config, cache_dir=cache_dir)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=num_max_tokens)
llm = HuggingFacePipeline(pipeline=pipe)


@app.post("/apiCaseId")
async def case_processing(caseId: str):
    # data = await request.json()
    # caseId = data.get("caseId")
    #caseId = caseId.caseId

    filename = caseId + ".pdf"
    folder_name = r"C:\Users\Local_Admin\Desktop\MTL_Legal\openvino-llm-chatbot-rag\data"
    filepath = os.path.join(folder_name, filename)
    loader = PyPDFLoader(filepath)
    data = loader.load()

    # text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=20)
    # docs = text_splitter.split_documents(data)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=20)
    texts = text_splitter.split_documents(data)

    model_name_embed = "jinaai/jina-embeddings-v2-base-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": False}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name_embed, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

    # embeddings = HuggingFaceEmbeddings()
    vector_store_directory = f"stores/{caseId}"

    if not os.path.exists(vector_store_directory):
        vector_store = Chroma.from_documents(
            texts,
            embeddings,
            collection_metadata={"hnsw:space": "cosine"},
            persist_directory=f"stores/{caseId}",
        )

    # # retriever part separately
    # load_vector_store = Chroma(
    #     persist_directory=f"stores/{caseId}", embedding_function=embeddings
    # )
    # retriever = load_vector_store.as_retriever(search_kwargs={"k": 1})

    # query = "urge of Sri Ganapathy lyer"
    # semantic_search = retriever.get_relevant_documents(query)
    # print(semantic_search)

    # db = Chroma.from_documents(docs, embeddings)
    # ques = "give information about second appeal 4th April, 1928"
    # docs = db.similarity_search(ques)
    # print(docs[0])

    return JSONResponse(content={"result": "Vector Datastore Created Successfully"})

@app.post('/apiQuery')
async def root(caseId:str, query:str):
    # data = await request.json()
    # caseId = data.get("caseId")
    # query = data.get("query")
    #caseId = query.caseId
    #query = query.query
    if query:

        vectorstore_dir =f"stores/{caseId}"
        logging.debug(vectorstore_dir)
        model_name_embed = "jinaai/jina-embeddings-v2-base-en"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": False}
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name_embed, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
        )
        prompt_template = """Use the following pieces of information to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}
        Question: {question}

        Only return the helpful answer below and nothing else.
        Helpful answer:
        """

        prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])

        chain_type_kwargs = {"prompt": prompt}
        vectorstore = Chroma(persist_directory=vectorstore_dir, embedding_function=embeddings)
        retriever = vectorstore.as_retriever()

        semantic_search = retriever.get_relevant_documents(query)
        #print(semantic_search)
        logging.debug("Search: ",semantic_search)

        # model_id = f'{model_vendor}/{model_name}'
        # tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        # ov_model_path = f'./{model_name}/{model_precision}'
        # model = OVModelForCausalLM.from_pretrained(model_id=ov_model_path, device=inference_device, ov_config=ov_config, cache_dir=cache_dir)
        # pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=num_max_tokens)
        # llm = HuggingFacePipeline(pipeline=pipe)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type=rag_chain_type, retriever=retriever, chain_type_kwargs=chain_type_kwargs,verbose=True)

        stime = time.time()
        ans = run_generation(query, qa_chain)
        etime = time.time()
        wc = len(ans.split())            # simple word count
        process_time = etime - stime
        logging.debug(process_time)
        logging.debug(f'Input tokens: {query}')  
        logging.debug(f'QA chain tokens: {qa_chain}')

        # Log output tokens
        logging.debug(f'Output tokens: {ans}')
        words_per_sec = wc / process_time
        logging.debug(f'Word count: {wc}, Processing Time: {process_time:6.1f} sec, {words_per_sec:6.2} words/sec')
        return {'response': ans}
    return {'response': ''}

  
 


# # API reference
# # http://127.0.0.1:8000/docs

# # How to run (You need to have uvicorn and streamlit -> pip install uvicorn streamlit)
# # uvicorn openvino-rag-server:app
