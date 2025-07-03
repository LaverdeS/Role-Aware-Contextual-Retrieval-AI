<h1 align="center">RACRA: Role-Aware Contextual Retrieval Agent</h1>

<p align="center">
  <a href="https://www.python.org/downloads/release/python-3120/" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white" alt="Python 3.12" style="margin-right:10px;" />
  </a>
  <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/" target="_blank" rel="noopener noreferrer">
    <img src="https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg" alt="CC BY-NC-ND 4.0 License" />
  </a>
</p>

RACRA is a multi-agent conversational system equipped role-aware RAG and several other tools. This role-sensitive agent is designed to streamline access to critical information for **Architects**, **Engineers**, and **Project Managers**. By integrating multiple data sources and leveraging contextual reasoning, RACRA enhances productivity, decision-making, and knowledge sharing across complex project environments.

---

## 🚀 Key Features

- **🔍 Autonomous Source Selection**  
  Dynamically determines the most relevant data source (SQL, vector DB, knowledge base) based on the user’s query.


- **🔗 Multi-Source Integration**  
  Combines information from multiple sources to provide comprehensive, unified answers.


- **📚 Citation Generation**  
  Automatically includes references to the original data sources for transparency and traceability.


- **🧠 Contextual Awareness**  
  Maintains conversational context to deliver coherent, relevant follow-up responses.


- **🛠️ Tool Utilization**  
  Employs specialized tools like web search, calculators, or custom plugins when needed.

---

## 👥 Role-Based Intelligence

RACRA understands and tailors its responses based on the user’s role:

### 👷 Engineer
- Access to technical specifications, system details, and engineering-focused data  
- Prioritized sources: technical documents, system blueprints, and spec sheets

### 🏛️ Architect
- Receives both design insights and project management data  
- Prioritized sources: design pattern repositories, schedules, budget allocations, and status reports

### 📋 Project Manager
- Access to supplier information, RFIs, and shared organizational knowledge  
- Prioritized sources: Suppliers Database, RFI Database, and Common Knowledge Base (shared architect/engineer knowledge)

---

## 📦 Requirements

Install the required packages using pip:

```bash
pip install -r requirements.txt
```

## 🛠️ Usage

- Insert API keys in the `.env` file (`OPENAI_API_KEY`, `TAVILY_API_KEY`, `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`)

- You can execute the tools in the `src/tools` directory to test them individually using the `--test_switcher` (A, B, or C) flag:

```bash
python src/tools.py --test_switcher A 
```

- To run the RACRA agent in demo mode (conversation with memory and tool calls), use the following command:

```bash
python src/agents.py --execution_mode demo 
```

- To run the RACRA agent in interactive mode (ui), use the following command:

```bash
python src/agents.py --execution_mode ui 
```

The gradio ui will open in your browser, allowing you to interact with the agent and see user and assistant messages.
The terminal output will display a more complete log with conversation turns, user queries, tool-calls and responses.

---

## 📎 License

This work is licensed under a
[Creative Commons Attribution-NonCommercial-NoDerivs 4.0 International License][cc-by-nc-nd].

See [`LICENSE`](./LICENSE) for details.

[![CC BY-NC-ND 4.0][cc-by-nc-nd-image]][cc-by-nc-nd]

[cc-by-nc-nd]: http://creativecommons.org/licenses/by-nc-nd/4.0/
[cc-by-nc-nd-image]: https://licensebuttons.net/l/by-nc-nd/4.0/88x31.png
