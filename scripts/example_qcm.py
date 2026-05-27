"""
Exemple 2 — Question à choix multiple (QCM)
=============================================
Appelle l'endpoint POST /generate de la FastAPI locale (ou déployée).
Le prompt est formaté comme un QCM médical avec choix de réponse,
identique au format des datasets d'entraînement (MedMCQA, MedQA, etc.).

Lancer la FastAPI d'abord :
    $env:INFERENCE_BACKEND = "vllm"
    $env:VLLM_BASE_URL     = "http://<IP>:8000"
    $env:MODEL_ID          = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
    uv run --project deployment --no-dev uvicorn app.main:app --port 8000 --reload
"""

import httpx
import time

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
API_URL = "http://localhost:8000"  # URL de ta FastAPI (pas du vLLM)

# Format QCM — adapte question et choix selon ton dataset
QUESTION = "Quel médicament est le traitement de première ligne pour l'hypertension artérielle essentielle non compliquée ?"
CHOICES = {
    "A": "Amoxicilline",
    "B": "Amlodipine",
    "C": "Méthotrexate",
    "D": "Furosémide",
}
CORRECT_ANSWER = "B"  # Pour évaluation, None si inconnu

# Format du prompt QCM
PROMPT = (
    f"Question : {QUESTION}\n\n"
    + "\n".join(f"{key}. {val}" for key, val in CHOICES.items())
    + "\n\nRéponds uniquement par la lettre de la bonne réponse (A, B, C ou D), "
    "suivie d'une courte justification."
)

# ==========================================
# 🏥 APPEL API
# ==========================================
print(f"📡 Connexion à la FastAPI : {API_URL}")

# Vérification santé
health = httpx.get(f"{API_URL}/health", timeout=10)
health.raise_for_status()
info = health.json()
print(f"✅ Backend: {info['backend']} | Modèle: {info['model_id']}")

# Génération
print(f"\n📝 QCM :\n{PROMPT}\n")
start = time.perf_counter()

response = httpx.post(
    f"{API_URL}/generate",
    json={
        "prompt": PROMPT,
        "max_new_tokens": 256,
        "temperature": 0.1,  # basse pour les QCM : on veut une réponse déterministe
    },
    timeout=120,
)
response.raise_for_status()

elapsed = round((time.perf_counter() - start) * 1000)
result = response.json()

# ==========================================
# 📋 AFFICHAGE + ÉVALUATION OPTIONNELLE
# ==========================================
model_answer = result["text"]

print("🩺 --- RÉPONSE DU MODÈLE ---")
print(model_answer)

if CORRECT_ANSWER:
    # Détection simple : la réponse commence-t-elle par la bonne lettre ?
    correct = model_answer.strip().upper().startswith(CORRECT_ANSWER)
    verdict = "✅ CORRECT" if correct else "❌ INCORRECT"
    print(f"\n{verdict} (attendu : {CORRECT_ANSWER})")

print(f"\n⏱️  Temps de réponse : {elapsed} ms")
print(f"🔧 Backend : {result['backend']} | Modèle : {result['model_id']}")
