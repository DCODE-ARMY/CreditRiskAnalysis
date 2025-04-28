<!-- ====================================================== -->
<!--  Market Cloud – Credit-Risk Dashboard  •  README.md     -->
<!--  (copy–paste directly into GitHub – all renders fine)   -->
<!-- ====================================================== -->

<h1 align="center">📊 Market Cloud – Credit-Risk Dashboard</h1>
<p align="center">
  <em>AI-driven, multi-agent platform that turns noisy market data, news &amp; public sentiment into a crystal-clear credit-risk score.</em>
</p>

<p align="center">
  <img alt="Streamlit UI" src="https://img.shields.io/badge/UI-Streamlit-E83E8C?logo=streamlit&logoColor=white">
  <img alt="Python"   src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white">
  <img alt="CrewAI"   src="https://img.shields.io/badge/https%3A%2F%2Funpkg.com%2Fsimple-icons%40v14%2Ficons%2Fcrewai.svg?logo=crewai">
  <img alt="License"  src="https://img.shields.io/github/license/your-org/market-cloud">
</p>

---

## 🗺️ About the Project
Traditional credit-risk analysis is **slow, siloed and manual**.  
**Market Cloud** re-imagines the workflow with:

* **Multi-Agent Intelligence** – each agent is an expert (research, sentiment, risk) that collaborates through [CrewAI](https://github.com/joaomdmoura/crewai).
* **Real-Time Data Fusion** – live market metrics (CDS spreads, bond yields), breaking news, regulatory filings **and** crowd sentiment are collected & scored on the fly.
* **Explainable Outputs** – every number and insight is backed by a Pydantic schema, interactive Plotly visual and one-click PDF.
* **Plug-and-Play** – drop in your own tools (Bloomberg, Refinitiv, etc.) or swap the LLM; the architecture is 100 % modular.

> From query to board-ready report in **〈 2 min ⌛️**.

---

## 🌟 Feature Highlights
| Layer | ⭐  Key Points |
| ----- | -------------- |
| **UI** | Streamlit dashboard with live agent logs, tabbed analytics, PDF export. |
| **Agents** | <br>• `credit_researcher_agent` – mines quantitative & qualitative data.<br>• `sentiment_agent` – converts signals into –1 … +1 sentiment scores.<br>• `credit_risk_agent` – weights each component (50 / 30 / 20 %) and outputs a 1-10 risk score + action items.<br> |
| **Tools** | `FinanceTool` (Financial Modeling Prep API) & `SerperDevTool` (news/web). Easily extend with Gmail/Calendar/GMeet, etc. |
| **LLM** | OpenAI **GPT-4o** provides reasoning muscle; interchangeable with Anthropic Claude, Llama 3, etc. |
| **Outputs** | • 📈 Plotly radar & gauge charts • 🗃️ Typed JSON via Pydantic • 📄 8-page PDF (ReportLab). |

---

## 🏗️ Architecture

<p align="center">
  <img src="docs/architecture.svg" alt="Architecture Diagram" width="100%">
</p>

*(Need the editable Draw.io file? Check `docs/architecture.drawio`.)*

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/your-org/market-cloud.git
cd market-cloud

# 2. Create & activate virtual env
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Export API keys
export GPT_API="sk-..."             # OpenAI (or set OPENAI_API_KEY)
export Serper_API="serper-key"
export FMP_API="fmp-key"

# 5. Launch the app
streamlit run app.py
