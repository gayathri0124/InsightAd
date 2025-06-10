# 📈 InsightAd: CTR Prediction and LLM-Based Ad Explanation

**InsightAd** is a machine learning pipeline that predicts **Click-Through Rates (CTR)** for online ads using user and ad features, and provides **natural language explanations** using a Large Language Model (LLM) such as OpenAI GPT or HuggingFace models.

---

## 🚀 Features

- ✅ CTR prediction using a PyTorch-based model (MLP or Dual-Tower)
- ✅ Structured preprocessing of real ad-click event data
- ✅ Inference from a saved model (`ctr_model.pt`) — no retraining needed
- ✅ Human-readable explanations via OpenAI or HuggingFace LLMs
- ✅ Modular and scalable design for future improvements (retrieval, ranking, etc.)

---

## 📊 Data Format

Use any CTR dataset (e.g., Outbrain Click Prediction from Kaggle) with:

- `clicks.csv`: `display_id`, `ad_id`, `clicked`
- `events.csv`: `display_id`, `platform`, `geo_location`, `uuid`
- `ads.csv`: `ad_id`, `campaign_id`, `advertiser_id`

Place them in the `data/` folder.

---

## 🛠️ Installation

```bash
git clone https://github.com/<your-username>/insightad.git
cd insightad
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

✅ Step 1: Train CTR Model
```bash
python train_ctr.py
```
🤖 Step 2: Predict and Explain via LLM
```bash
python inference_llm.py
```

Sample Output
🟣 Explanation for user_23, ad_14, predicted CTR = 0.1872

🧠 LLM Explanation:
This ad was likely shown because the user was browsing on a mobile device in North America, which aligns with the campaign's target demographic...

