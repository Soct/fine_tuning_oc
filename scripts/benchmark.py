"""
Benchmark vLLM via la FastAPI — métriques de performance
=========================================================
Mesure la latence et le débit (tokens/s) sur N requêtes consécutives,
ce qui justifie l'utilisation de vLLM pour le serving (batching continu,
optimisations KV-cache, PagedAttention).

Lancer la FastAPI d'abord :
    $env:INFERENCE_BACKEND = "vllm"
    $env:VLLM_BASE_URL     = "http://<IP>:8000"
    $env:MODEL_ID          = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"
    uv run --project deployment --no-dev uvicorn app.main:app --port 8000
"""

import statistics
import time

import httpx

# ==========================================
# ⚙️  CONFIGURATION
# ==========================================
API_URL = "http://localhost:8000"
N_REQUESTS = 5
PROMPTS = [
    "Quels sont les symptômes les plus courants d'une crise d'asthme ?",
    "Expliquez le mécanisme d'action des inhibiteurs de l'ECA.",
    "Quels sont les signes cliniques d'un infarctus du myocarde ?",
    "Comment diagnostique-t-on un diabète de type 2 ?",
    "Quelles sont les contre-indications des AINS ?",
]

# ==========================================
# 🏥  VÉRIFICATION SANTÉ
# ==========================================
print(f"📡 Connexion à la FastAPI : {API_URL}")
health = httpx.get(f"{API_URL}/health", timeout=10)
health.raise_for_status()
info = health.json()
print(f"✅ Backend : {info['backend']} | Modèle : {info['model_id']}\n")

# ==========================================
# 🔥  BENCHMARK
# ==========================================
print(f"⏳ Lancement de {N_REQUESTS} requêtes...\n")

results = []
for i in range(N_REQUESTS):
    prompt = PROMPTS[i % len(PROMPTS)]
    t0 = time.perf_counter()
    response = httpx.post(
        f"{API_URL}/generate",
        json={"prompt": prompt, "max_new_tokens": 256, "temperature": 0.3},
        timeout=180,
    )
    response.raise_for_status()
    latency_ms = (time.perf_counter() - t0) * 1000

    data = response.json()
    usage = data.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    tps = usage.get("tokens_per_second")

    results.append(
        {
            "latency_ms": latency_ms,
            "completion_tokens": completion_tokens,
            "tokens_per_second": tps,
        }
    )
    tps_str = f"{tps:.1f} tok/s" if tps else "N/A"
    tokens_str = str(completion_tokens) if completion_tokens else "N/A"
    print(f"  [{i + 1}/{N_REQUESTS}] {round(latency_ms)} ms | {tokens_str} tokens | {tps_str}")

# ==========================================
# 📊  RÉSUMÉ
# ==========================================
latencies = [r["latency_ms"] for r in results]
tps_values = [r["tokens_per_second"] for r in results if r["tokens_per_second"]]
token_counts = [r["completion_tokens"] for r in results if r["completion_tokens"]]

print(f"\n{'=' * 50}")
print(f"📊  RÉSULTATS — {N_REQUESTS} requêtes séquentielles")
print(f"{'=' * 50}")
print(f"  Latence moyenne    : {round(statistics.mean(latencies))} ms")
print(f"  Latence médiane    : {round(statistics.median(latencies))} ms")
if len(latencies) >= 2:
    sorted_lat = sorted(latencies)
    p95_idx = max(0, round(0.95 * len(sorted_lat)) - 1)
    print(f"  Latence p95        : {round(sorted_lat[p95_idx])} ms")
print(f"  Latence min / max  : {round(min(latencies))} / {round(max(latencies))} ms")
if tps_values:
    print(f"  Débit moyen        : {statistics.mean(tps_values):.1f} tokens/s")
    print(f"  Débit max          : {max(tps_values):.1f} tokens/s")
if token_counts:
    print(f"  Tokens générés moy : {round(statistics.mean(token_counts))}")
print(f"{'=' * 50}")
print(
    "\n💡 vLLM utilise PagedAttention + continuous batching :"
    "\n   → pas de padding, mémoire KV-cache paginée, débit optimal"
    "\n   → latence bien inférieure à un serving naïf HuggingFace"
)
