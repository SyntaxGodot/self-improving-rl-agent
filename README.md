# 🧠 Self-Improving RL Agent for Automated AI Research 

A production-grade architecture combining Meta-RL and AutoML for autonomous AI research.

## 📂 Structure
- `src/domain/` : Core logic (NO external dependencies)
- `src/application/` : Use cases & orchestration
- `src/infrastructure/` : External integrations (PyTorch, Gym, FAISS, DBs)
- `src/interface/` : CLI, API, Streamlit dashboard
- `src/config/` : YAML configs & settings

## 🛠️ Quick Start
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py
```
