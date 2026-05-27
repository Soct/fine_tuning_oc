# Justification vLLM

Le POC local expose une API FastAPI avec un backend `echo` volontairement leger. Ce choix permet de tester l'API, les logs JSON, Docker et la CI sans GPU CUDA ni installation lourde.

Le depot supporte maintenant aussi un backend `vllm` cote FastAPI : l'API devient alors une facade qui appelle un serveur vLLM OpenAI-compatible via HTTP.

Pour un deploiement d'inference LLM, vLLM est le choix cible parce qu'il apporte :

- un serveur OpenAI-compatible mature pour exposer le modele via HTTP ;
- un meilleur debit que `transformers.generate` grace a son moteur d'inference optimise ;
- la gestion efficace du KV cache avec PagedAttention ;
- du batching continu utile quand plusieurs requetes arrivent en meme temps ;
- une integration adaptee aux GPU de production, avec configuration du nombre de tokens, de la memoire GPU et du parallelisme.

Dans ce projet, le modele fine-tune est un adapter LoRA Qwen3. La trajectoire production recommandee est :

1. fusionner ou charger l'adapter LoRA selon le support exact de la version vLLM retenue ;
2. servir le modele avec `vllm serve` sur une machine GPU ;
3. pointer `/generate` vers le serveur vLLM ou remplacer le backend `echo` par un client HTTP OpenAI-compatible ;
4. mesurer latence, debit, VRAM et taux d'erreur sur un jeu de prompts medicalement representatif.

Le fallback actuel ne pretend pas valider la qualite du modele. Il valide seulement le packaging applicatif : contrat API, observabilite minimale, tests et portabilite Docker.

## Architecture recommandee

Deux topologies sont raisonnables :

```text
Client -> FastAPI -> vLLM -> GPU -> modele
```

ou, si aucune logique API metier n'est necessaire :

```text
Client -> vLLM -> GPU -> modele
```

Dans ce depot, on retient la premiere option pour pouvoir conserver :

- les routes `/health` et `/generate` ;
- les logs JSON par requete ;
- une couche applicative distincte du moteur d'inference ;
- un packaging testable meme sans GPU.

## Lancer localement FastAPI + vLLM

1. Sur la machine GPU, lancer vLLM :

```bash
vllm serve unsloth/Qwen3-1.7B-unsloth-bnb-4bit \
	--host 0.0.0.0 \
	--port 8001
```

2. Dans le projet FastAPI, pointer le backend sur vLLM :

```bash
INFERENCE_BACKEND=vllm \
VLLM_BASE_URL=http://127.0.0.1:8001 \
MODEL_ID=unsloth/Qwen3-1.7B-unsloth-bnb-4bit \
uv run --project deployment --no-dev uvicorn app.main:app --host 0.0.0.0 --port 8000
```

3. Tester :

```bash
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/generate \
	-H 'Content-Type: application/json' \
	-d '{"prompt":"Quels sont les signes d alerte d une douleur thoracique ?","max_new_tokens":64,"temperature":0.2}'
```

## Deployer sur Google Cloud

Le chemin simple est Google Compute Engine.

### Option recommandee

1. VM GPU Ubuntu pour vLLM.
2. VM legere ou meme VM pour FastAPI.
3. Communication sur reseau prive ou sur une IP autorisee.

### Sequence cible

1. Creer une VM GPU avec drivers NVIDIA et Python/Docker.
2. Recuperer le modele de base et l'adapter LoRA si necessaire.
3. Lancer vLLM sur le port `8001`.
4. Lancer FastAPI avec :

```bash
INFERENCE_BACKEND=vllm
VLLM_BASE_URL=http://<adresse-vllm>:8001
MODEL_ID=unsloth/Qwen3-1.7B-unsloth-bnb-4bit
```

5. Exposer FastAPI via une IP publique ou un reverse proxy.
6. Mesurer latence, debit, VRAM et erreurs.

### Variables d'environnement utiles

- `INFERENCE_BACKEND=vllm`
- `VLLM_BASE_URL=http://127.0.0.1:8001`
- `MODEL_ID=unsloth/Qwen3-1.7B-unsloth-bnb-4bit`
- `VLLM_API_KEY=<token optionnel si proxy protege>`
- `VLLM_TIMEOUT_SECONDS=120`

### Limite importante

Le support exact d'un adapter LoRA avec vLLM depend de la version retenue et du mode de chargement. Si besoin, il faut fusionner l'adapter avec le modele de base avant le service, ou adapter la commande `vllm serve` au mecanisme LoRA supporte par la version choisie.
