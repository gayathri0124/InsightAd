from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="tiiuae/falcon-rw-1b",  # Small model that fits locally
    trust_remote_code=True,
    device="cpu"
)

def explain_recommendation(user_id, ad_id, context):
    prompt = f"""
User features:
{context['user']}

Ad {ad_id} features:
{context['ad']}

CTR prediction: {context['score']}

Why was this ad shown?
"""
    response = generator(prompt, max_new_tokens=100)[0]["generated_text"]
    return response
