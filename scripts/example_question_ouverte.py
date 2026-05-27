"""
Exemple 1 — Question médicale ouverte
======================================
Appelle l'endpoint POST /generate de la FastAPI locale (ou déployée).
La FastAPI relaye la requête vers le vLLM distant et logue entrée/sortie.

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

PROMPT = "Quels sont les symptômes les plus courants d'une crise d'asthme ?"

# ==========================================
# 🏥 APPEL API
# ================================
# ==========
print(f"📡 Connexion à la FastAPI : {API_URL}")

# Vérification santé
health = httpx.get(f"{API_URL}/health", timeout=10)
health.raise_for_status()
info = health.json()
print(f"✅ Backend: {info['backend']} | Modèle: {info['model_id']}")

# Génération
print(f"\n💬 Prompt : {PROMPT}\n")
start = time.perf_counter()

response = httpx.post(
    f"{API_URL}/generate",
    json={
        "prompt": PROMPT,
        "max_new_tokens": 512,
        "temperature": 0.3,
    },
    timeout=120,
)
response.raise_for_status()

elapsed = round((time.perf_counter() - start) * 1000)
result = response.json()

# ==========================================
# 📋 AFFICHAGE
# ==========================================
print("🩺 --- RÉPONSE DU MODÈLE ---")
print(result["text"])
print(f"\n⏱️  Temps de réponse : {elapsed} ms")
print(f"🔧 Backend : {result['backend']} | Modèle : {result['model_id']}")
