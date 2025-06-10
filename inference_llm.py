import torch
import pandas as pd
from preprocessing.prepare_ctr_data import load_and_merge_data, preprocess_data
from ranking.ctr_model import CTRModel
from llm_utils import explain_recommendation

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device set to use {device}")

df = load_and_merge_data()
X, y = preprocess_data(df)  

sample_X = X.sample(n=5, random_state=42)
X_tensor = torch.tensor(sample_X.values, dtype=torch.float32).to(device)


input_dim = X.shape[1]
model = CTRModel(input_dim)
model.load_state_dict(torch.load("ctr_model.pt", map_location=device))
model.to(device)
model.eval()


with torch.no_grad():
    preds = model(X_tensor).cpu().numpy().flatten()

for i in range(5):
    user_id = f"user_{i}"
    ad_id = f"ad_{i}"

    sample_features = sample_X.iloc[i].to_dict()
    ctr_score = round(float(preds[i]), 4)

    context = {
        "user": sample_features,  
        "ad": sample_features,
        "score": ctr_score
    }

    print(f"\n🔍 Explanation for user {user_id}, ad {ad_id}, predicted CTR={ctr_score}")
    try:
        explanation = explain_recommendation(user_id, ad_id, context)
        print("🧠 LLM Explanation:", explanation)
    except Exception as e:
        print("⚠️ Error generating explanation:", e)
