# load core modules
import os
from dotenv                      import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores      import FAISS
from langchain.chat_models       import ChatOpenAI
from langchain.text_splitter     import RecursiveCharacterTextSplitter
from langchain.chains            import ConversationalRetrievalChain
from langchain.memory            import ConversationBufferMemory

# load agents and tools modules
import pandas as pd
from langchain.tools.python.tool import PythonAstREPLTool
from langchain.agents            import initialize_agent, Tool
from langchain.agents            import AgentType
from langchain                   import LLMMathChain
from langchain.utilities         import SerpAPIWrapper, GoogleSearchAPIWrapper

# Directory where the HTML files are stored
FILES_DIR = "./docs"

# Load environment variables
load_dotenv()

# read all the TXT files and concatenate their content into a single string
def get_txt_contents():
    text = ""
    for filename in os.listdir(FILES_DIR):
        if filename.endswith(".txt"):
            with open(os.path.join(FILES_DIR, filename), 'r', encoding='utf-8') as f:
                text += f.read()
    return text

# Retrieve the concatenated content from all the TXT files
try:
    text = get_txt_contents()
except FileNotFoundError:
    print(f"Error: HTMLs files not found: {FILES_DIR}")
except Exception as e:
    print(f"Error occurred while reading the HTML files: {e}")

#Split text into tokens
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size      = 1000,
        chunk_overlap   = 150,
        length_function = len
    )

# Process the TXT content and create the list of document chunks
documents = text_splitter.split_text(text=text)

# Vectorize the documents and create a vectorstore using FAISS
embeddings  = OpenAIEmbeddings(model = "text-embedding-ada-002")
vectorstore = FAISS.from_texts(documents, embedding=embeddings)

# Initialize the Langchain chatbot using the OpenAI model
openai_llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")

# Set up a buffer for conversation history
conversation_memory = ConversationBufferMemory(
    memory_key      = "chat_history", 
    return_messages = True, 
    output_key      = "answer"
    )

local_txt_files = ConversationalRetrievalChain.from_llm(
        llm                     = openai_llm, 
        retriever               = vectorstore.as_retriever(search_type = "similarity"), 
        memory                  = conversation_memory,
        return_source_documents = False, 
        verbose                 = True,
        output_key              = "answer",
        chain_type              = "stuff",
        max_tokens_limit        = None
    )

# load employee_data.csv as dataframe
df = pd.read_csv("./docs/employee_data_v3.csv")  

# set access of python_repl tool to the dataframe
python = PythonAstREPLTool(locals={"df": df})  

# load the tool for Google Searches
search = SerpAPIWrapper()

# create calculator tool
calculator = LLMMathChain.from_llm(llm=openai_llm)

# create variables for f strings embedded in the prompts
user = "John Doe"  # set user
df_columns = df.columns.to_list()  # print column names of df

# prepare the vectordb retriever, the python_repl and langchain calculator as tools for the agent
tools = [
    Tool(
        name="Local TXT files",
        func=local_txt_files.run,
        description="""
        Useful for when you need to answer questions about employee timekeeping policies.

        <user>: What is the policy on unused vacation leave?
        <assistant>: I need to check the timekeeping policies to answer this question.
        <assistant>: Action: Timekeeping Policies
        <assistant>: Action Input: Vacation Leave Policy - Unused Leave
        ...
        """,
    ),
    Tool(
        name="Employee Data",
        func=python.run,
        description=f"""
        Useful for when you need to answer questions about employee data stored in pandas dataframe 'df'. 
        Run python pandas operations on 'df' to help you get the right answer.
        'df' has the following columns: {df_columns}
        
        <user>: How many Sick Leave do I have left?
        <assistant>: df[df['name'] == '{user}']['sick_leave']
        <assistant>: You have n sick leaves left.              
        """,
    ),
    Tool(
        name="Calculator",
        func=calculator.run,
        description=f"""
        Useful when you need to do math operations or arithmetic.
        """,
    ),
    Tool(
        name="Langchain Serpapi",
        func=search.run,
        description=f"""
        A search engine. Useful for when you need to answer questions about current events. Input should be a search query.

        <user>: What is the capital of France?
        <assistant>: Action: Langchain Serpapi
        <assistant>: Action Input: What is the capital of France?
        <assistant>: Paris
        """,
    ),
]

# change the value of the prefix argument in the initialize_agent function. This will overwrite the default prompt template of the zero shot agent type
agent_kwargs = {"prefix": f"You are friendly HR assistant. You are tasked to assist the current user: {user} on questions related to HR. You have access to the following tools:"}

# initialize the LLM agent
agent = initialize_agent(
    tools        = tools,
    llm          = openai_llm,
    agent        = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose      = True,
    agent_kwargs = agent_kwargs,
)

def get_response(user_input):
    try:
        response= agent.run(user_input)
    except Exception as e:
        response = str(e)
        if response.startswith("Could not parse LLM output: `"):
            response = response.removeprefix("Could not parse LLM output: `").removesuffix("`")
    return response

while True:
    user_input = input("### Please enter your question (or type 'exit' to quit): ")
    
    if user_input.lower() == 'exit':
        break

    response = get_response(user_input)

    print("******************************************************************************")
    print("### Response: " + response)
    print("******************************************************************************")
