from src.helper import clone_repo, load_repo_as_documents, create_text_chunks, download_embeddings
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
import os

load_dotenv()
openai_api_key = os.environ.get('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = openai_api_key

#url = 'https://github.com/shafeeh011/medical_chatbot'


# Clone the repository
#clone_repo(url)

# Load the repository as documents
documents = load_repo_as_documents('repo/')

# Create chunks from the documents
text_chunks = create_text_chunks(documents)

# Download embeddings
embeddings = download_embeddings()


# Initialize a Chroma vectorstore
vectorstore = Chroma.from_documents(text_chunks, embeddings, persist_directory='./db')
vectorstore.persist()