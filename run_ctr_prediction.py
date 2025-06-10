from llm_utils import explain_recommendation
import torch
import pandas as pd

user_id = "user_001"
ad_id = "ad_123"

context = {
    "user": {"age": 25, "interests": ["fitness", "tech"]},
    "ad": {"category": "fitness", "headline": "AI Smartwatch for Runners"},
    "score": 0.89  
}

print(f"CTR Score for User {user_id} on Ad {ad_id}: {context['score']}")

explanation = explain_recommendation(user_id, ad_id, context)
print("\n🧠 LLM Explanation:")
print(explanation)
