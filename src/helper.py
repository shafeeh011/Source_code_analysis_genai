from git import Repo
from langchain.text_splitter import Language
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
import os

#  clone the repository
def clone_repo(repo_url):
    os.mkdir('test_repo', exist_ok=True)
    repo_path = 'repo/'
    Repo.clone_from(repo_url, to_path=repo_path)
    
# Loading the repository as documentsfrom langchain.embeddings.openai import OpenAIEmbeddings
def load_repo_as_documents(repo_path):
    loader = GenericLoader.from_filesystem(repo_path,
                                           glob="**/*",
                                           suffixes=[".py"],
                                           parser=LanguageParser(language=Language.PYTHON, parser_threshold=500))
    documents = loader.load()
    return documents

#Creating chunks from the documents
def create_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON,chunk_size=2000, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

# Download embeddings
def download_embeddings():
    embeddings = OpenAIEmbeddings(disallowed_special=())
    return embeddings