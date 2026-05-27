from openai import OpenAI
import time
import re

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
IP_VM = ""  
SHOW_THINK = True

client = OpenAI(
    api_key="PROJET_ETUDIANT", 
    base_url=f"http://{IP_VM}:8000/v1"
)

print(f"📡 Connexion au serveur {IP_VM}...")
start_time = time.time()

response = client.chat.completions.create(
    model="unsloth/Qwen3-1.7B-unsloth-bnb-4bit",
    messages=[
        {"role": "system", "content": "Tu es un assistant médical IA utile, précis et concis."},
        {"role": "user", "content": "Quels sont les symptômes les plus courants d'une crise d'asthme ?"}
    ],
    temperature=0.3,
    max_tokens=2048
)

message = response.choices[0].message

# La vraie réponse médicale (déjà nettoyée par vLLM !)
reponse_finale = message.content 

# La réflexion (vLLM l'isole dans un attribut "reasoning")
# On utilise getattr car c'est un attribut spécial ajouté par vLLM
brouillon = getattr(message, "reasoning", "Aucune réflexion disponible.")

end_time = time.time()

# --- AFFICHAGE ---
print("\n🩺 --- RÉPONSE DU MODÈLE ---")

if SHOW_THINK:
    print("🤔 [MODE RAISONNEMENT]")
    print(brouillon)
    print("\n✅ [RÉPONSE FINALE]")

print(reponse_finale)

print(f"\n⏱️ Temps de réponse : {round(end_time - start_time, 2)} secondes")

