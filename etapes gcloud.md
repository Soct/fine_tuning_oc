# Guide de déploiement frugal : modèle Qwen 1.7B (vLLM) sur Google Cloud

Ce guide détaille comment déployer un modèle LLM fine-tuné avec adaptateur LoRA de manière ultra-économique en utilisant les crédits étudiants (Free Tier) sur Google Cloud Platform (GCP).

## Partie 1 : Préparation du compte Google Cloud

Par défaut, l'essai gratuit bloque l'accès aux cartes graphiques (GPU). Il faut donc débloquer le compte tout en sécurisant le budget.

### Activer la facturation complète

1. Aller dans `Facturation` (`Billing`).
2. Cliquer sur `Activer le compte` (`Upgrade`).

Note : Google utilisera toujours les 300 $ offerts en priorité.

### Demander le quota GPU

1. Aller dans `IAM et administration > Quotas`.
2. Filtrer par `NVIDIA T4 GPUs` ou `GPUs all regions`.
3. Sélectionner la ligne, cliquer sur `Modifier les quotas` et demander une limite de `1`.

Justification suggérée : projet étudiant, inférence LLM.

### Créer une alerte de sécurité anti-dépassement

1. Aller dans `Facturation > Budgets et alertes`.
2. Créer un budget fixé à `250 $`.
3. Configurer des alertes par e-mail, par exemple à `50 %`, `80 %` et `100 %`.

## Partie 2 : Création de la machine virtuelle

L'objectif est d'utiliser une instance Spot pour réduire la facture d'environ `70 %`, soit environ `0,19 $ à 0,26 $ / heure`.

1. Aller dans `Compute Engine > Instances de VM`.
2. Cliquer sur `Créer`.

### Configuration matérielle

- Type de machine : `Famille N1 -> n1-standard-4` (`4 vCPU`, `15 Go RAM`)
- GPU : `1 x NVIDIA T4`

### Disque de démarrage

- Système d'exploitation : `Deep Learning on Linux`
- Version : image avec `CUDA 12.x` et `PyTorch` préinstallés
- Taille : `50 Go` en SSD persistant

### Pare-feu

- Autoriser le trafic `HTTP` et `HTTPS`

### Option budget

1. Ouvrir `Options avancées > Gestion`.
2. Changer le modèle de provisionnement de `Standard` à `Spot`.

## Partie 3 : Préparation de l'environnement Linux

Une fois la VM démarrée, cliquer sur `SSH` pour ouvrir le terminal dans le navigateur.

### 1. Installer les outils pour l'environnement virtuel

Contournement de la sécurité `PEP 668` de Linux :

```bash
sudo apt update && sudo apt install python3.12-venv -y
```

### 2. Créer et activer l'environnement isolé

```bash
python3 -m venv vllm-env
source vllm-env/bin/activate
```

Le préfixe `(vllm-env)` doit apparaître dans le terminal.

### 3. Installer vLLM et les dépendances pour modèles compressés en 4-bit

```bash
pip install vllm bitsandbytes
```

## Partie 4 : Ouvrir le port 8000 dans le pare-feu Google Cloud

Par défaut, Google Cloud bloque les connexions entrantes sur les ports non standards. Il faut créer une règle pour laisser passer les requêtes vers vLLM avant de tester l'accès au serveur.

### Étapes dans la console Google Cloud

1. Utiliser la barre de recherche en haut de l'écran et taper `Pare-feu` ou `Firewall`.
2. Ouvrir `Pare-feu (Réseau VPC)`.
3. Cliquer sur `Créer une règle de pare-feu`.
4. Renseigner les champs suivants :

- Nom : `autoriser-vllm-8000`
- Cibles : `Toutes les instances du réseau`
- Filtre source : `Plages d'adresses IPv4`
- Plages d'adresses IPv4 sources : `0.0.0.0/0`
- Protocoles et ports : `Protocoles et ports spécifiés` puis `TCP` sur le port `8000`

5. Cliquer sur `Créer`.

La règle est active immédiatement.

## Partie 5 : Téléchargement et lancement de l'IA

Pour éviter les blocages de sécurité réseau de Hugging Face vis-à-vis des datacenters, on télécharge l'adaptateur localement avant de lancer le serveur.

### 1. Télécharger l'adaptateur LoRA médical

```bash
hf download Maphe/qwen3-1.7b-medical-finetuned --local-dir ./mon_lora_medical
```

### 2. Lancer le serveur d'inférence vLLM

`--dtype half` (`float16`) est obligatoire car le GPU `T4` ne supporte pas le `bfloat16`.

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model "unsloth/Qwen3-1.7B-unsloth-bnb-4bit" \
    --enable-lora \
    --lora-modules IA-Medicale=./mon_lora_medical \
    --dtype half \
    --max-model-len 4096 \
    --host 0.0.0.0 --port 8000
```

Attendre l'apparition du message suivant :

```text
INFO: Uvicorn running on http://0.0.0.0:8000
```

## Partie 6 : Démarrage automatique de vLLM avec systemd

Une fois le lancement manuel validé, on peut automatiser le démarrage pour éviter de se reconnecter en SSH et de relancer la commande à chaque allumage de la VM.

### 1. Créer le fichier de configuration du service

Depuis le terminal SSH de la VM, exécuter le bloc suivant.

Remplacer `soctbyswot` par le nom d'utilisateur réel de la VM si nécessaire, ainsi que les chemins `/home/soctbyswot/...`.

```bash
sudo tee /etc/systemd/system/vllm.service > /dev/null << 'EOF'
[Unit]
Description=Serveur vLLM IA Medicale
After=network.target

[Service]
User=soctbyswot
WorkingDirectory=/home/soctbyswot
Environment="PATH=/home/soctbyswot/vllm-env/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=/home/soctbyswot/vllm-env/bin/python3 -m vllm.entrypoints.openai.api_server --model "unsloth/Qwen3-1.7B-unsloth-bnb-4bit" --enable-lora --lora-modules IA-Medicale=/home/soctbyswot/mon_lora_medical --dtype half --max-model-len 4096 --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
```

### 2. Activer et démarrer le service

Exécuter ensuite ces commandes :

```bash
sudo systemctl daemon-reload
sudo systemctl enable vllm
sudo systemctl start vllm
```

### 3. Commandes utiles de gestion

Voir les logs en direct :

```bash
sudo journalctl -u vllm -f
```

Redémarrer le service :

```bash
sudo systemctl restart vllm
```

Arrêter le service :

```bash
sudo systemctl stop vllm
```

### Workflow final

La prochaine fois, il suffit de démarrer la VM dans Google Cloud, d'attendre environ deux minutes, puis le client Python peut appeler le serveur vLLM sans relancer manuellement la commande en SSH.

## Partie 7 : Règle d'or de survie financière

Ne jamais laisser la VM tourner inutilement.

Dès que la session de travail est terminée :

1. Aller dans la console Google Cloud.
2. Sélectionner la VM.
3. Cliquer sur `Arrêter` (`Stop`).

La facturation du processeur et du GPU s'arrête instantanément. Seul le stockage du disque reste facturé, à quelques centimes par jour.