# CampaignX – AI Multi-Agent Marketing Automation

**FrostHack 2026 | IIT Mandi | Team Project**

## Project Structure

```
Agentic_Ai/
├── agents/                  # AI agent modules
│   ├── __init__.py
│   ├── planner.py           # Campaign strategy agent
│   ├── creator.py           # Email content generation agent
│   ├── executor.py          # Campaign execution agent
│   └── optimizer.py         # Performance analysis & optimization agent
│
├── assets/                  # Static frontend assets
│   └── style.css            # All dashboard CSS (loaded by app.py)
│
├── config/                  # Configuration helpers
│   └── __init__.py
│
├── data/                    # Data files
│   ├── superbfsi_api_spec.yaml   # OpenAPI spec for CampaignX APIs
│   └── mock_cohort.json          # Fallback cohort for offline testing
│
├── tests/                   # Test scripts
│   └── test_ollama.py
│
├── utils/                   # Shared utilities
│   ├── __init__.py
│   └── ollama_client.py     # Ollama LLM wrapper (used by all agents)
│
├── .env                     # Environment variables (not committed)
├── .env.example             # Template for .env
├── .gitignore
├── app.py                   # Streamlit dashboard entry point
├── requirements.txt
└── README.md
```

## Setup

1. Copy `.env.example` to `.env` and fill in your keys:
   ```
   CAMPAIGNX_API_KEY=your_key_here
   OLLAMA_MODEL=llama3.1:8b
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the dashboard:
   ```bash
   python -m streamlit run app.py
   ```

## Tech Stack
- **LLM:** Ollama (local) with `llama3.1:8b`
- **Agentic Framework:** LangChain
- **UI:** Streamlit + custom CSS
- **APIs:** SuperBFSI CampaignX API (OpenAPI spec-based dynamic discovery)
