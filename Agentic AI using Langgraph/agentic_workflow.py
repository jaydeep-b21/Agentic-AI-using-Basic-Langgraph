from langgraph.graph import Graph, END
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

from dotenv import load_dotenv
load_dotenv()

cohere_api_key = os.environ["COHERE_API_KEY"]

llm = ChatCohere(
    cohere_api_key=cohere_api_key,
    model="command-r-plus",  
    temperature=0.3
)

# Define state
class AgentState(TypedDict):
    input_document: Document
    document_type: Union[Literal["report"], Literal["form"], Literal["draft"], None]
    processed_content: str
    human_feedback: Union[str, None]
    human_satisfied: Union[bool, None]
    output_file: Union[str, None]

# Define tools/nodes for each processing step
def identify_document_type(state: AgentState) -> AgentState:
    """Analyze document to determine its type."""
    content = state["input_document"].page_content[:1000]  # Look at first 1000 chars
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Analyze the document content and classify it as one of:
        - "report" (structured content with sections/headings)
        - "form" (templates, applications, standardized documents)
        - "draft" (poorly written, needs enhancement)
        
        Return JUST the type."""),
        ("user", "{content}")
    ])
    
    chain = prompt | llm
    doc_type = chain.invoke({"content": content}).content.lower()
    
    if "report" in doc_type:
        state["document_type"] = "report"
    elif "form" in doc_type or "template" in doc_type:
        state["document_type"] = "form"
    else:
        state["document_type"] = "draft"
    
    return state

def summarize_report(state: AgentState) -> AgentState:
    """Summarize report documents."""
    content = state["input_document"].page_content
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert report summarizer. Create a concise summary 
        that captures the key points, findings, and recommendations. 
        Keep it under 500 words."""),
        ("user", "{content}")
    ])
    
    chain = prompt | llm
    summary = chain.invoke({"content": content}).content
    state["processed_content"] = summary
    return state

def classify_form(state: AgentState) -> AgentState:
    """Classify and tag form documents."""
    content = state["input_document"].page_content
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a document classifier. Analyze this form/template and:
        1. Identify its purpose
        2. Extract all field names
        3. Tag with relevant categories
        
        Return this as a structured analysis."""),
        ("user", "{content}")
    ])
    
    chain = prompt | llm
    analysis = chain.invoke({"content": content}).content
    state["processed_content"] = analysis
    return state

def enhance_draft(state: AgentState) -> AgentState:
    """Enhance poorly written drafts."""
    content = state["input_document"].page_content
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an editor. Improve this draft by:
        1. Correcting grammar/spelling
        2. Improving clarity and flow
        3. Adding structure if needed
        4. Keeping the original meaning
        
        Return the enhanced version."""),
        ("user", "{content}")
    ])
    
    chain = prompt | llm
    enhanced = chain.invoke({"content": content}).content
    state["processed_content"] = enhanced
    return state

def get_human_feedback(state: AgentState) -> AgentState:
    """Simulate getting human feedback (in real app, would be UI input)."""
    # In a real app, this would come from user interface
    print("\nProcessed Content Preview:")
    print(state["processed_content"][:800] + "...")
    
    feedback = input("\nAre you satisfied with this result? (yes/no): ").strip().lower()
    state["human_satisfied"] = feedback.startswith("y")
    
    if not state["human_satisfied"]:
        state["human_feedback"] = input("What should be improved? ")
    
    return state

def email_agent(state: AgentState) -> AgentState:
    """Agent to email the processed document automatically with natural subject/body using Cohere LLM."""

    sender_email = os.environ["GMAIL_ADDRESS"]
    app_password = os.environ["GMAIL_APP_PASSWORD"]
    receiver_email = "jaydeep.biswas21@outlook.com"

    # === Use Cohere to generate a human-like subject and body ===
    
    llm = ChatCohere(
        cohere_api_key=cohere_api_key,
        model="command-r-plus",
        temperature=0.4
    )

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

    chain = prompt | llm
    response = chain.invoke({
        "type": state["document_type"],
        "content": state["processed_content"][:500]  # Limit to prevent overload
    }).content

    # Parse subject and body from response
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

    # === Build the email ===
    msg = EmailMessage()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject
    msg.set_content(body)

    # === Attach the processed document ===
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

    # === Send the email ===
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


def save_to_word(state: AgentState) -> AgentState:
    """Save processed content to Word file."""
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    doc = docx.Document()
    doc.add_paragraph(state["processed_content"])
    
    # Get and clean original filename
    original_path = state['input_document'].metadata.get('source', 'document')
    original_name = os.path.basename(original_path)
    original_name = os.path.splitext(original_name)[0]
    original_name = "".join(c for c in original_name if c.isalnum() or c in (' ', '_', '-'))
    original_name = original_name[:50]  # Limit length
    
    # Add timestamp to avoid overwrites
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"output/processed_{original_name}_{timestamp}.docx"
    
    doc.save(filename)
    state["output_file"] = filename
    print(f"Saved to {filename}")
    return state

def restart_process(state: AgentState) -> AgentState:
    """Reset state to restart the process with feedback."""
    print("\nRestarting process with feedback...")
    state["processed_content"] = ""
    state["human_feedback"] = None
    state["human_satisfied"] = None
    return state

# Define conditional edges
def route_by_doc_type(state: AgentState) -> str:
    """Route to appropriate processor based on document type."""
    return state["document_type"]

def route_by_feedback(state: AgentState) -> str:
    """Route based on human feedback."""
    if state["human_satisfied"]:
        return "save"
    return "restart"
# Build the workflow
workflow = Graph()

# Add nodes (unchanged)
workflow.add_node("identify_document_type", identify_document_type)
workflow.add_node("summarize_report", summarize_report)
workflow.add_node("classify_form", classify_form)
workflow.add_node("enhance_draft", enhance_draft)
workflow.add_node("get_human_feedback", get_human_feedback)
workflow.add_node("save_to_word", save_to_word)
workflow.add_node("email_agent", email_agent)
workflow.add_node("restart_process", restart_process)

# Add conditional routing FIRST
workflow.add_conditional_edges(
    "identify_document_type",
    route_by_doc_type,
    {
        "report": "summarize_report",
        "form": "classify_form",
        "draft": "enhance_draft",
    }
)

# Then add regular edges
workflow.add_edge("summarize_report", "get_human_feedback")
workflow.add_edge("classify_form", "get_human_feedback")
workflow.add_edge("enhance_draft", "get_human_feedback")
workflow.add_edge("save_to_word", "email_agent")

# Add feedback conditional routing
workflow.add_conditional_edges(
    "get_human_feedback",
    route_by_feedback,
    {
        "save": "save_to_word",
        "restart": "restart_process",
    }
)

# Final edges
workflow.add_edge("email_agent", END)
workflow.add_edge("restart_process", "identify_document_type")

# Set entry point
workflow.set_entry_point("identify_document_type")

# Compile the workflow
app = workflow.compile()

# usage
def process_document(file_path: str):
    """Run the workflow on a document."""
    loader = TextLoader(file_path)
    doc = loader.load()[0]
    
    state = {
        "input_document": doc,
        "document_type": None,
        "processed_content": "",
        "human_feedback": None,
        "human_satisfied": None,
        "output_file": None
    }
    
    for step in app.stream(state):
        for key, value in step.items():
            print(f"✅Completed step: {key}")

if __name__ == "__main__":
    file_path = input("Enter document path to process: ").strip().strip('\'"')
    process_document(file_path)
