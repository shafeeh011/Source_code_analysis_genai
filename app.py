from flask import Flask, render_template, jsonify, request
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from src.helper import load_repo_as_documents, create_text_chunks, download_embeddings
import os
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()

# Initialize OpenAI
openai_api_key = os.environ.get('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = openai_api_key

# Initialize embeddings
embeddings = download_embeddings()
persist_directory = 'db'

# Load the persisted vectorstore database from disk
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# Initialize chat model
llm = ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo')

# Initialize memory and QA chain
memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 3}),
    memory=memory
)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chatbot', methods=["GET", "POST"])
def get_answer():
    user_input = request.form['question']
    load_repo_as_documents(user_input)
    os.system('python store_index.py')
    return jsonify({"response": f"Repository loaded: {user_input}"})

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form['msg']
    
    if msg.lower() == 'clear':
        os.system('rm -rf db')
        return "Chat history cleared"
    
    result = qa({"question": msg})
    return str(result['answer'])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)