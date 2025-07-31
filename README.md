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
