# 🚀 InsightAd: Intelligent CTR Ad Ranking & Explanation Engine

**InsightAd** is a full-stack machine learning system for personalized ad retrieval, ranking, and interpretation.  
It combines classical dual-tower architectures for user–ad matching with a CTR prediction model and integrates **LLM-based natural language explanations** of why ads were recommended.

🎯 Built for personalized advertising pipelines, recommendation research, and interview-ready ML portfolio projects.

---

## 📌 Features

- 🔍 **Two-Tower BERT Retrieval** for efficient user–ad matching
- 📈 **CTR Prediction Model** (MLP-based ranking)
- 🧠 **LLM Explanation Engine** using OpenAI GPT
- 🎛️ **Streamlit UI** for real-time recommendations
- ⚙️ **FastAPI backend** (optional) for scalable APIs
- ✅ **Modular pipeline**: `retrieval/`, `ranking/`, `inference/`, `streamlit_app/`
- 🧪 Ready for evaluation, ablation, and live testing

---

## 🧠 System Architecture

User Features Ad Features
│ │
▼ ▼
User Tower Ad Tower (BERT)
│ │
└────▶ Dot Product ───▶ Retrieval Scores (Top-K)
│
▼
CTR Ranker (MLP, multi-objective loss)
│
▼
Final CTR Score + Explanations (LLM)

---

## 🗂️ Project Structure

InsightAd/
├── retrieval/ # Two-tower architecture
├── ranking/ # CTR MLP model
├── inference/ # FastAPI endpoints
├── preprocessing/ # Data loading, merging, features
├── streamlit_app/ # Streamlit UI
├── data/ # Model files (.pt), embeddings, etc.
├── requirements.txt
├── app.py # Optional FastAPI entry
└── README.md


---

## 🚀 Quickstart

### 🔧 1. Install Dependencies

```bash
git clone https://github.com/<your-username>/InsightAd.git
cd InsightAd
pip install -r requirements.txt
```
### ⚙️ 2. Train or Load Models

```bash
# Optional: train your dual-tower or CTR models
# Or place .pt model files in /data/
```

### ✅ 3. Launch Streamlit UI

```bash
streamlit run streamlit_app/app.py
```

### 🧠 4. Enable GPT(Openapi Key)

```bash
export OPENAI_API_KEY=sk-xxxxx
```

---

📘 Example Use Case

User ID: 29837
Top-5 Recommended Ads: [85, 22, 7, 120, 44]

🧠 Explanation (GPT):
"This ad was shown because the user recently interacted with sports content
and this product has a high CTR for users in the California region."

---

🛠️ Built With
- PyTorch, scikit-learn

- Streamlit, FastAPI

- OpenAI GPT-3.5 / GPT-4 (via openai)

- pandas, NumPy, tqdm

---

"InsightAd brings transparency to click-through predictions by combining powerful ranking models with language-based explanations."

---
