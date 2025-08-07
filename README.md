# Agentic-AI-using-Basic-Langgraph

This project is an intelligent document processing pipeline that uses [LangGraph](https://github.com/langchain-ai/langgraph) and LLM to classify, analyze, and enhance textual documents automatically.

It can support three document types:
-  **Reports** – summarized concisely
-  **Forms** – classified and fields extracted
-  **Drafts** – enhanced for grammar, clarity, and structure

---

##  Demo

![Demo](Agentic%20AI%20using%20Langgraph/demo/langgraph_agent_gif1.gif)

> _A walkthrough of how the workflow classifies and processes a document interactively._

---

## Features

-  Document type classification: `report`, `form`, or `draft`
-  AI-enhanced content processing using LLM
-  Human-in-the-loop feedback simulation
-  Word document export (`.docx`)
-  Dedicated Document Mailing agent (`smtp`)
-  Built using LangGraph for modular workflow design

---

##  Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/jaydeep-b21/Agentic-AI-using-Basic-Langgraph.git
cd Agentic-AI-using-Basic-Langgraph
````

### 2. Set Up a Virtual Environment

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

Make a `requirements.txt`:

```bash
pip install -r requirements.txt
```

or install manually

---

##  API Key Setup

This project uses Cohere's Command-R+ model. You’ll need a LLM API key.

1. Get a key
2. Open `.py` file and replace:

```python
cohere_api_key = "your-api-key"
```

>  For better security, use `.env` or environment variables.

---

##  Usage

Run the script via terminal

When prompted, enter the path to a `.txt` file. The program will:

1. Classify the document type
2. Process accordingly
3. Display a preview
4. Ask for feedback
5. Save a `.docx` version in the `output/` folder
6. Agent attaches the document & mails to a person

---

## 📂 Project Structure

```
.
├── demo/                   # Contains demo.gif
├── output/                 # Generated .docx files
├── .py file   # Main Python script
├── README.md               # This file
└── requirements.txt        # (Optional) Dependency list
```

---

##  Contributing

Feel free to open issues or submit PRs for improvements!

---

##  Acknowledgments

* [LangGraph](https://github.com/langchain-ai/langgraph)
* [Cohere](https://cohere.com/)
* [LangChain](https://www.langchain.com/)

  
I'll break down the provided code line by line, explaining each section's purpose and functionality. The code implements an agentic workflow using LangGraph to process documents, classify their type, process them accordingly, incorporate human feedback, save the output, and email it. It leverages a language model (Cohere) and various tools for document handling and email sending.

### Imports and Setup
```python
from langgraph.graph import Graph, END
```
- **Purpose**: Imports the `Graph` class and `END` constant from `langgraph.graph` to build a workflow graph where nodes represent processing steps and edges define transitions. `END` marks the termination point of the workflow.

```python
from typing import TypedDict, Annotated, Union
from langchain_core.agents import AgentFinish
from langchain_core.messages import HumanMessage
from langchain_cohere import ChatCohere
from langchain_core.prompts import ChatPromptTemplate
import os
from typing import Literal
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader
import docx
import mimetypes
import smtplib
from email.message import EmailMessage
```
- **Purpose**:
  - `TypedDict`, `Annotated`, `Union`, `Literal`: Type hints from Python's `typing` module to define structured data types and constrain values.
  - `AgentFinish`, `HumanMessage`: From `langchain_core`, used for agent workflows, though `AgentFinish` is unused here.
  - `ChatCohere`: LangChain's integration with Cohere's language model for text processing.
  - `ChatPromptTemplate`: For creating structured prompts for the language model.
  - `os`: For file system operations (e.g., creating directories, accessing environment variables).
  - `Document`, `TextLoader`: From LangChain for loading and representing documents.
  - `docx`: Python-docx library for creating Word documents.
  - `mimetypes`, `smtplib`, `EmailMessage`: For handling file types and sending emails via SMTP.

```python
from dotenv import load_dotenv
load_dotenv()
```
- **Purpose**: Loads environment variables from a `.env` file using `python-dotenv`. This is used to securely access API keys and email credentials.

```python
cohere_api_key = os.environ["COHERE_API_KEY"]
```
- **Purpose**: Retrieves the Cohere API key from environment variables for authenticating with the Cohere language model.

```python
llm = ChatCohere(
    cohere_api_key=cohere_api_key,
    model="command-r-plus",  
    temperature=0.3
)
```
- **Purpose**: Initializes a `ChatCohere` instance with the Cohere API key, using the `command-r-plus` model and a temperature of 0.3 (controls response randomness; lower values make outputs more deterministic).

### State Definition
```python
class AgentState(TypedDict):
    input_document: Document
    document_type: Union[Literal["report"], Literal["form"], Literal["draft"], None]
    processed_content: str
    human_feedback: Union[str, None]
    human_satisfied: Union[bool, None]
    output_file: Union[str, None]
```
- **Purpose**: Defines `AgentState` as a `TypedDict` to structure the workflow's state. It tracks:
  - `input_document`: The input document (LangChain `Document` object).
  - `document_type`: The classified type of document (`report`, `form`, `draft`, or `None`).
  - `processed_content`: The processed text output.
  - `human_feedback`: User feedback if provided.
  - `human_satisfied`: Boolean indicating user satisfaction.
  - `output_file`: Path to the saved output file.

### Node Functions
Each node function processes the state and returns an updated state. These represent steps in the workflow.

#### `identify_document_type`
```python
def identify_document_type(state: AgentState) -> AgentState:
    """Analyze document to determine its type."""
    content = state["input_document"].page_content[:1000]  # Look at first 1000 chars
```
- **Purpose**: Defines a function to classify the document type by analyzing the first 1000 characters of the input document's content.

```python
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Analyze the document content and classify it as one of:
        - "report" (structured content with sections/headings)
        - "form" (templates, applications, standardized documents)
        - "draft" (poorly written, needs enhancement)
        
        Return JUST the type."""),
        ("user", "{content}")
    ])
```
- **Purpose**: Creates a prompt template instructing the language model to classify the document as `report`, `form`, or `draft` based on its content, returning only the type.

```python
    chain = prompt | llm
    doc_type = chain.invoke({"content": content}).content.lower()
```
- **Purpose**: Chains the prompt with the Cohere language model (`llm`) and invokes it with the document content to get the document type, converted to lowercase.

```python
    if "report" in doc_type:
        state["document_type"] = "report"
    elif "form" in doc_type or "template" in doc_type:
        state["document_type"] = "form"
    else:
        state["document_type"] = "draft"
    
    return state
```
- **Purpose**: Sets the `document_type` in the state based on the model's output. If the output contains "report," sets it to `report`; if it contains "form" or "template," sets it to `form`; otherwise, defaults to `draft`. Returns the updated state.

#### `summarize_report`
```python
def summarize_report(state: AgentState) -> AgentState:
    """Summarize report documents."""
    content = state["input_document"].page_content
```
- **Purpose**: Defines a function to summarize documents classified as `report`.

```python
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert report summarizer. Create a concise summary 
        that captures the key points, findings, and recommendations. 
        Keep it under 500 words."""),
        ("user", "{content}")
    ])
```
- **Purpose**: Creates a prompt instructing the model to summarize the report concisely (under 500 words), focusing on key points, findings, and recommendations.

```python
    chain = prompt | llm
    summary = chain.invoke({"content": content}).content
    state["processed_content"] = summary
    return state
```
- **Purpose**: Invokes the prompt with the full document content to generate a summary, stores it in `processed_content`, and returns the updated state.

#### `classify_form`
```python
def classify_form(state: AgentState) -> AgentState:
    """Classify and tag form documents."""
    content = state["input_document"].page_content
```
- **Purpose**: Defines a function to process documents classified as `form`.

```python
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a document classifier. Analyze this form/template and:
        1. Identify its purpose
        2. Extract all field names
        3. Tag with relevant categories
        
        Return this as a structured analysis."""),
        ("user", "{content}")
    ])
```
- **Purpose**: Creates a prompt to analyze a form, identifying its purpose, extracting field names, and tagging it with categories.

```python
    chain = prompt | llm
    analysis = chain.invoke({"content": content}).content
    state["processed_content"] = analysis
    return state
```
- **Purpose**: Invokes the prompt to generate a structured analysis, stores it in `processed_content`, and returns the updated state.

#### `enhance_draft`
```python
def enhance_draft(state: AgentState) -> AgentState:
    """Enhance poorly written drafts."""
    content = state["input_document"].page_content
```
- **Purpose**: Defines a function to enhance documents classified as `draft`.

```python
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an editor. Improve this draft by:
        1. Correcting grammar/spelling
        2. Improving clarity and flow
        3. Adding structure if needed
        4. Keeping the original meaning
        
        Return the enhanced version."""),
        ("user", "{content}")
    ])
```
- **Purpose**: Creates a prompt to improve the draft by correcting errors, enhancing clarity, adding structure, and preserving meaning.

```python
    chain = prompt | llm
    enhanced = chain.invoke({"content": content}).content
    state["processed_content"] = enhanced
    return state
```
- **Purpose**: Invokes the prompt to generate an enhanced version, stores it in `processed_content`, and returns the updated state.

#### `get_human_feedback`
```python
def get_human_feedback(state: AgentState) -> AgentState:
    """Simulate getting human feedback (in real app, would be UI input)."""
    print("\nProcessed Content Preview:")
    print(state["processed_content"][:800] + "...")
```
- **Purpose**: Defines a function to simulate collecting human feedback by printing the first 800 characters of the processed content.

```python
    feedback = input("\nAre you satisfied with this result? (yes/no): ").strip().lower()
    state["human_satisfied"] = feedback.startswith("y")
```
- **Purpose**: Prompts the user to indicate satisfaction (`yes`/`no`). Sets `human_satisfied` to `True` if the input starts with "y", otherwise `False`.

```python
    if not state["human_satisfied"]:
        state["human_feedback"] = input("What should be improved? ")
    
    return state
```
- **Purpose**: If the user is not satisfied, prompts for improvement feedback and stores it in `human_feedback`. Returns the updated state.

#### `email_agent`
```python
def email_agent(state: AgentState) -> AgentState:
    """Agent to email the processed document automatically with natural subject/body using Cohere LLM."""
    sender_email = os.environ["GMAIL_ADDRESS"]
    app_password = os.environ["GMAIL_APP_PASSWORD"]
    receiver_email = "jaydeep.biswas21@outlook.com"
```
- **Purpose**: Defines a function to email the processed document. Retrieves sender email and app password from environment variables and hardcodes the recipient email.

```python
    llm = ChatCohere(
        cohere_api_key=cohere_api_key,
        model="command-r-plus",
        temperature=0.4
    )
```
- **Purpose**: Initializes a new `ChatCohere` instance with a slightly higher temperature (0.4) for generating natural-sounding email text.

```python
    prompt = ChatPromptTemplate.from_messages([
        ("system", 
        "You're composing a short, friendly email from one person to another. "
        "You're simply sharing a document—not completing any task or providing detailed analysis.\n\n"
        "Based on the content, generate:\n"
        "1. A natural, human-sounding subject line (not AI-ish)\n"
        "2. A warm, professional body text. You can include a one-line summary of the document content if helpful.\n\n"
        "Always follow these rules:\n"
        "- Begin the email with 'Hi,' (not with the recipient's name)\n"
        "- End the email with 'Best,\nDocument Delivery'\n"
        "- You are just a messenger, not the one who created or worked on the content.\n\n"
        "Respond in the format:\n"
        "Subject: <subject line>\n\nBody:\n<body content>"),
        
        ("user", "Content Preview:\n{content}")
    ])
```
- **Purpose**: Creates a prompt to generate a natural email subject and body, adhering to specific formatting rules (e.g., starting with "Hi," and ending with "Best,\nDocument Delivery").

```python
    chain = prompt | llm
    response = chain.invoke({
        "type": state["document_type"],
        "content": state["processed_content"][:500]  # Limit to prevent overload
    }).content
```
- **Purpose**: Invokes the prompt with the `document_type` and first 500 characters of `processed_content` to generate the email text.

```python
    if "Subject:" in response and "Body:" in response:
        subject = response.split("Subject:")[1].split("Body:")[0].strip()
        body = response.split("Body:")[1].strip()
    else:
        subject = f"Here's the finalized {state['document_type']} document"
        body = (
            f"Hey,\n\nJust finished working on the {state['document_type']} you asked for. "
            f"It's attached — take a look when you get a chance.\n\n"
            f"Let me know what you think.\n\nCheers,\nJaydeep"
        )
```
- **Purpose**: Parses the subject and body from the model's response. If parsing fails, uses fallback subject and body text with the document type and a generic message.

```python
    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.set_content(body)
```
- **Purpose**: Creates an `EmailMessage` object, setting the sender, recipient, subject, and body.

```python
    file_path = state.get("output_file")
    if file_path and os.path.exists(file_path):
        with open(file_path, "rb") as f:
            file_data = f.read()
            mime_type, _ = mimetypes.guess_type(file_path)
            maintype, subtype = mime_type.split('/')
            msg.add_attachment(file_data, maintype=maintype, subtype=subtype, filename=os.path.basename(file_path))
    else:
        print("❌ Output file missing, skipping email attachment.")
        return state
```
- **Purpose**: Attaches the processed document (from `output_file`) to the email. Reads the file in binary mode, determines its MIME type, and adds it as an attachment. If the file is missing, prints an error and skips attachment.

```python
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login(sender_email, app_password)
            smtp.send_message(msg)
        print("Email sent successfully by agent.")
    except Exception as e:
        print(f"❌ Email agent failed: {e}")
    
    return state
```
- **Purpose**: Sends the email via Gmail's SMTP server using port 587 with TLS encryption. Logs in with the sender's email and app password, sends the message, and prints success or error messages. Returns the state.

#### `save_to_word`
```python
def save_to_word(state: AgentState) -> AgentState:
    """Save processed content to Word file."""
    os.makedirs("output", exist_ok=True)
```
- **Purpose**: Defines a function to save the processed content to a Word file. Creates an `output` directory if it doesn't exist.

```python
    doc = docx.Document()
    doc.add_paragraph(state["processed_content"])
```
- **Purpose**: Creates a new Word document and adds the `processed_content` as a paragraph.

```python
    original_path = state['input_document'].metadata.get('source', 'document')
    original_name = os.path.basename(original_path)
    original_name = os.path.splitext(original_name)[0]
    original_name = "".join(c for c in original_name if c.isalnum() or c in (' ', '_', '-'))
    original_name = original_name[:50]  # Limit length
```
- **Purpose**: Extracts the original filename from the document's metadata, removes the extension, sanitizes it (keeping alphanumeric characters, spaces, underscores, and hyphens), and limits it to 50 characters.

```python
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output/processed_{original_name}_{timestamp}.docx"
```
- **Purpose**: Generates a unique filename with a timestamp to avoid overwrites, in the format `output/processed_<original_name>_<timestamp>.docx`.

```python
    doc.save(filename)
    state["output_file"] = filename
    print(f"Saved to {filename}")
    return state
```
- **Purpose**: Saves the Word document to the generated filename, stores the path in `output_file`, prints the save location, and returns the updated state.

#### `restart_process`
```python
def restart_process(state: AgentState) -> AgentState:
    """Reset state to restart the process with feedback."""
    print("\nRestarting process with feedback...")
    state["processed_content"] = ""
    state["human_feedback"] = None
    state["human_satisfied"] = None
    return state
```
- **Purpose**: Resets the state to restart processing if the user is not satisfied. Clears `processed_content`, `human_feedback`, and `human_satisfied`, then returns the updated state.

### Conditional Routing
```python
def route_by_doc_type(state: AgentState) -> str:
    """Route to appropriate processor based on document type."""
    return state["document_type"]
```
- **Purpose**: Defines a function that returns the `document_type` to route the workflow to the appropriate processing node (`summarize_report`, `classify_form`, or `enhance_draft`).

```python
def route_by_feedback(state: AgentState) -> str:
    """Route based on human feedback."""
    if state["human_satisfied"]:
        return "save"
    return "restart"
```
- **Purpose**: Defines a function that routes to `save_to_word` if the user is satisfied (`human_satisfied` is `True`), or to `restart_process` if not.

### Workflow Construction
```python
workflow = Graph()
```
- **Purpose**: Initializes a new `Graph` instance to define the workflow.

```python
workflow.add_node("identify_document_type", identify_document_type)
workflow.add_node("summarize_report", summarize_report)
workflow.add_node("classify_form", classify_form)
workflow.add_node("enhance_draft", enhance_draft)
workflow.add_node("get_human_feedback", get_human_feedback)
workflow.add_node("save_to_word", save_to_word)
workflow.add_node("email_agent", email_agent)
workflow.add_node("restart_process", restart_process)
```
- **Purpose**: Registers each node function with a unique identifier in the workflow graph.

```python
workflow.add_conditional_edges(
    "identify_document_type",
    route_by_doc_type,
    {
        "report": "summarize_report",
        "form": "classify_form",
        "draft": "enhance_draft",
    }
)
```
- **Purpose**: Adds conditional edges from `identify_document_type` to route to `summarize_report`, `classify_form`, or `enhance_draft` based on the `document_type` returned by `route_by_doc_type`.

```python
workflow.add_edge("summarize_report", "get_human_feedback")
workflow.add_edge("classify_form", "get_human_feedback")
workflow.add_edge("enhance_draft", "get_human_feedback")
workflow.add_edge("save_to_word", "email_agent")
```
- **Purpose**: Adds direct edges to connect processing nodes to `get_human_feedback` and `save_to_word` to `email_agent`.

```python
workflow.add_conditional_edges(
    "get_human_feedback",
    route_by_feedback,
    {
        "save": "save_to_word",
        "restart": "restart_process",
    }
)
```
- **Purpose**: Adds conditional edges from `get_human_feedback` to route to `save_to_word` or `restart_process` based on `route_by_feedback`.

```python
workflow.add_edge("email_agent", END)
workflow.add_edge("restart_process", "identify_document_type")
```
- **Purpose**: Adds final edges: `email_agent` to `END` to terminate the workflow, and `restart_process` to `identify_document_type` to loop back if restarting.

```python
workflow.set_entry_point("identify_document_type")
```
- **Purpose**: Sets `identify_document_type` as the starting node of the workflow.

```python
app = workflow.compile()
```
- **Purpose**: Compiles the workflow into an executable application.

### Main Function
```python
def process_document(file_path: str):
    """Run the workflow on a document."""
    loader = TextLoader(file_path)
    doc = loader.load()[0]
```
- **Purpose**: Defines a function to process a document from a file path. Loads the document using `TextLoader`, which returns a list of `Document` objects, and takes the first one.

```python
    state = {
        "input_document": doc,
        "document_type": None,
        "processed_content": "",
        "human_feedback": None,
        "human_satisfied": None,
        "output_file": None
    }
```
- **Purpose**: Initializes the `AgentState` with the loaded document and default values for other fields.

```python
    for step in app.stream(state):
        for key, value in step.items():
            print(f"✅Completed step: {key}")
```
- **Purpose**: Runs the compiled workflow with the initial state, iterating through each step and printing the completed node names.

### Entry Point
```python
if __name__ == "__main__":
    file_path = input("Enter document path to process: ").strip().strip('\'"')
    process_document(file_path)
```
- **Purpose**: When the script is run directly, prompts the user for a document path, removes leading/trailing whitespace and quotes, and calls `process_document` with the path.

### Summary
The code defines an agentic workflow using LangGraph to process documents in the following steps:
1. **Identify Document Type**: Classifies the document as a `report`, `form`, or `draft` using Cohere's language model.
2. **Process Document**: Summarizes reports, classifies forms, or enhances drafts based on the type.
3. **Get Human Feedback**: Displays the processed content and collects user satisfaction and feedback.
4. **Routing**: If satisfied, saves the output to a Word file and emails it; if not, restarts the process.
5. **Save and Email**: Saves the processed content to a Word file and sends it via email with a natural subject and body generated by Cohere.

Each function is modular, handling a specific task, and the workflow is orchestrated using a graph with conditional routing based on document type and user feedback. The code integrates external services (Cohere for NLP, Gmail for emailing) and handles file operations (loading text, saving Word documents).chrome
