# Justification vLLM

Le POC local expose une API FastAPI avec un backend `echo` volontairement leger. Ce choix permet de tester l'API, les logs JSON, Docker et la CI sans GPU CUDA ni installation lourde.

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
