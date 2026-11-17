# API de Tarification d'Assurance Auto - Industrialisation ML

*(English version below)*

Ce dépôt contient le projet complet d'industrialisation algorithmique (MLOps) réalisé dans le cadre du cours d'Industrialisation IA du Master 2 Fintech de CY Tech. Il vise à transformer un script monolithique d'entraînement de modèles (Machine Learning & Actuariat) en une API REST robuste, testée et prête pour la production.

## Stack Technique & Infrastructure
* **Langage & Environnement :** Python 3.12, `uv` (gestionnaire ultra-rapide de paquets et d'environnements)
* **Framework API :** FastAPI, Uvicorn, Pydantic (validation stricte des schémas de données)
* **Machine Learning :** Scikit-Learn, XGBoost, Optuna, Joblib
* **Qualité & Tests :** Pytest, HTTPX (TestClient)
* **DevOps & CI/CD :** Docker, GitHub Actions (CI pipeline), Render (Déploiement Continu)

## Architecture du Projet
Le projet sépare strictement la phase de recherche/entraînement de l'environnement d'inférence en production :

* `app/` : Code source de l'API en production.
  * `main.py` : Application FastAPI, définition des routes et schémas Pydantic.
  * `inference.py` : Chargement des modèles en mémoire (Singleton) et logique de prédiction.
  * `processing.py` : Feature engineering unitaire et déterministe pour les payloads JSON.
* `models_export/` : Artefacts des modèles sérialisés (`.joblib`). Contient les préprocesseurs, les modèles Ridge/XGBoost et les poids de blending. *(Note : exceptionnellement versionnés sur Git pour les besoins académiques du déploiement cloud).*
* `tests/` : Suite de tests automatisés (`pytest`) validant la cohérence mathématique, le contrat d'API et la gestion des erreurs HTTP 422.
* `training/` : Scripts de recherche originaux en Data Science (`step_final.py`). Intègre la logique actuarielle avancée (Capping, Blending, correction de Duan).
* `render.yaml` : Configuration du déploiement Render (Blueprint).
* `.github/workflows/` : Configuration de l'Intégration Continue (CI) pour installer les dépendances, exécuter les tests et tester le build Docker à chaque `push`.

## Logique Actuarielle & Machine Learning
L'API calcule la **prime pure** d'un contrat d'assurance automobile de façon unitaire, en respectant scrupuleusement la formule académique : `premium = frequency * amount`.

* **Modèle de Fréquence :** Blending optimisé entre une Régression Logistique bien calibrée et un XGBClassifier (optimisé via Optuna).
* **Modèle de Gravité (Montant) :** Blending entre une Régression Ridge log-transformée (avec correction de l'estimateur de Duan) et un XGBRegressor (fonction de perte Gamma). Entraîné exclusivement sur des données écrêtées au 99e percentile pour isoler le risque attritionnel.
* **Feature Engineering :** Les variables dynamiques (ex: extraction du département, ratio puissance/poids, calcul de l'ancienneté de permis) sont encapsulées directement dans le pipeline de traitement de l'API.

## Contrat d'API (Endpoints)
L'API impose une validation stricte des entrées. Des champs manquants ou mal typés (ex: une chaîne de caractères à la place d'un entier) déclencheront systématiquement une erreur `HTTP 422 Unprocessable Entity`, évitant ainsi les crashs serveurs (500).

| Endpoint | Méthode | Entrée | Exemple de Sortie |
| :--- | :--- | :--- | :--- |
| `/health` | `GET` | Aucune | `{"status": "ok"}` |
| `/predict_frequency` | `POST` | Contrat JSON | `{"frequency": 0.12}` |
| `/predict_amount` | `POST` | Contrat JSON | `{"amount": 1500.0}` |
| `/predict` | `POST` | Contrat JSON | `{"frequency": 0.12, "amount": 1500.0, "premium": 180.0}` |

*La documentation interactive est accessible nativement sur `/docs` (Swagger UI) et `/openapi.json`.*

## Format des entrées (contrat JSON)
Chaque route de prédiction reçoit **un seul contrat**. Le schéma complet, avec une description pour chaque champ, est visible dans le Swagger (`/docs`, schéma `ContractInput`).

| Champ | Type | Valeurs acceptées |
| :--- | :--- | :--- |
| `id_client`, `id_vehicule`, `id_contrat` | texte non vide | libre (non utilisés par les modèles) |
| `bonus` | nombre | 0,5 à 3,5 (1 = neutre) |
| `type_contrat` | texte | `Mini`, `Median1`, `Median2`, `Maxi` |
| `duree_contrat`, `anciennete_info` | entier | 0 à 100 |
| `freq_paiement` | texte | `Monthly`, `Quarterly`, `Biannual`, `Yearly` |
| `paiement` | texte | `Yes`, `No` |
| `utilisation` | texte | `WorkPrivate`, `Retired`, `Professional`, `AllTrips` |
| `code_postal` | **texte** de 5 caractères | `"06088"` (zéro initial conservé), Corse `"2A004"` / `"2B033"` |
| `conducteur2` | texte | `Yes`, `No` |
| `age_conducteur1` | entier | 18 à 120 |
| `age_conducteur2` | entier | **0 si `conducteur2 = "No"`**, sinon 18 à 120 |
| `sex_conducteur1` | texte | `"M"` ou `"F"` uniquement |
| `sex_conducteur2` | texte ou `null` | **`null` si `conducteur2 = "No"`**, sinon `"M"` ou `"F"` |
| `anciennete_permis1` | entier | 0 à 111 (**111 = inconnu**, convention des données) |
| `anciennete_permis2` | entier | **0 si `conducteur2 = "No"`**, 111 = inconnu |
| `anciennete_vehicule` | entier ou `null` | 0 à 100, `null` = inconnu (remplacé par la médiane) |
| `cylindre_vehicule` | entier | 0 à 10 000 cm³ (0 = non renseigné) |
| `din_vehicule` | entier | 1 à 2 000 ch |
| `essence_vehicule` | texte | `Gasoline`, `Diesel`, `Hybrid` |
| `marque_vehicule` | texte | ex. `"PEUGEOT"` (majuscules ou minuscules) ; une marque rare ou inconnue est regroupée en `"Autre"`, comme à l'entraînement |
| `modele_vehicule` | texte | libre (non utilisé par les modèles) |
| `debut_vente_vehicule`, `fin_vente_vehicule` | entier | 0 à 100 (non utilisés par les modèles) |
| `vitesse_vehicule` | entier | 1 à 500 km/h |
| `type_vehicule` | texte | `Tourism`, `Commercial` |
| `prix_vehicule` | nombre | 0 à 10 000 000 € (0 = non renseigné) |
| `poids_vehicule` | entier | 0 à 20 000 kg (0 = non renseigné) |

Les bornes sont plus larges que les valeurs observées : les 100 000 contrats de `train.csv` et `test.csv` sont tous acceptés.

**Règles de validation (erreur 422, jamais 500)** : champ manquant, mauvais type (`"30"` ou `true` pour un nombre, `30.7` pour un entier), catégorie inconnue (`"X"` pour le sexe), valeur hors bornes, corps vide ou JSON invalide, deuxième conducteur incohérent (`conducteur2 = "No"` avec un âge). Un entier envoyé sous forme décimale exacte (`10.0`) est accepté, car pandas exporte ainsi les colonnes qui contiennent des valeurs manquantes.

## Tests
`tests/test_api.py` contient environ 380 tests, en 8 parties :
1. **Monitoring** : `/health`, méthodes HTTP interdites (405), route inconnue (404).
2. **Swagger** : `/docs` et `/openapi.json`, valeurs autorisées documentées, description de chaque champ, exemple valide.
3. **Contrat d'API** : clés exactes, nombres JSON positifs, fréquence entre 0 et 1, `premium = frequency * amount`, cohérence entre `/predict` et les deux autres routes, déterminisme.
4. **Champs nullables et conventions métier** : `null` et champ absent, permis 111, valeurs 0, Corse, identifiants sans effet sur le prix.
5. **Marques** : marque inconnue regroupée en `"Autre"`, casse ignorée.
6. **Entrées invalides** : chaque champ manquant sur chaque route, mauvais types, catégories inconnues, bornes, codes postaux invalides, deuxième conducteur incohérent.
7. **Robustesse** : 200 contrats aléatoires valides, aucune erreur 500.
8. **Modules `processing` et `inference`** : variables dérivées, `null` → NaN, encodage identique à l'entraînement, test de non-régression sur l'exemple du TD.

## Déploiement Local

**1. Cloner le dépôt et initialiser l'environnement**
```bash
git clone https://github.com/matteo-dev/IA_Industrialisation_Auto_Insurance_Pricing_ML.git
cd IA_Industrialisation_Auto_Insurance_Pricing_ML
uv sync
```

**2. Lancer localement avec Uvicorn (Mode Développement)**
```bash
uv run uvicorn app.main:app --reload
```
L'interface Swagger sera disponible sur : `http://127.0.0.1:8000/docs`

**3. Exécuter la suite de tests automatisés**
```bash
uv run pytest tests/
```

**4. Construire et lancer via Docker (Mode Production)**
```bash
docker build -t api-tarification-auto:latest .
docker run -p 8000:8000 api-tarification-auto:latest
```

## Déploiement Cloud (CI/CD)
Ce projet est configuré pour un déploiement continu sur **Render**. Chaque `push` sur la branche `main` déclenche une GitHub Action qui valide les tests unitaires. En cas de succès, Render récupère le dernier commit, construit l'image Docker et la déploie sur une URL publique "Live".

**Configuration Render** (fichier `render.yaml`, lu par Render en mode *Blueprint*) :
* **Runtime** : Docker, à partir du `Dockerfile` à la racine (aucune commande de build ou de démarrage à saisir).
* **Port** : l'API écoute sur la variable d'environnement `PORT` (8000 par défaut). Render fournit cette variable, et le `Dockerfile` la lit au démarrage.
* **Health check** : `/health`. Render n'envoie le trafic vers une nouvelle version qu'une fois que cette route répond 200.
* **Déploiement automatique** : `autoDeployTrigger: checksPass`, c'est-à-dire seulement quand la CI GitHub est verte.
* **Offre gratuite** : le service se met en veille après 15 minutes sans requête. Le premier appel suivant peut prendre environ une minute (démarrage du conteneur et chargement des modèles).

**Mise en place** : Render → *New* → *Blueprint* → choisir ce dépôt. Render lit `render.yaml` et crée le service. Pour vérifier : `GET https://<service>.onrender.com/health` renvoie `{"status": "ok"}`, et la documentation est sur `https://<service>.onrender.com/docs`.

---
---

# Auto Insurance Pricing API - MLOps Pipeline

This repository contains the end-to-end algorithmic industrialization (MLOps) project developed for the **AI Industrialization course (M2 Fintech, CY Tech)**. The objective is to transform a monolithic machine learning and actuarial training script into a robust, fully tested, and production-ready REST API.

## Tech Stack & Infrastructure
* **Language & Environment:** Python 3.12, `uv` (Fast Python package and project manager)
* **API Framework:** FastAPI, Uvicorn, Pydantic (Strict schema validation)
* **Machine Learning:** Scikit-Learn, XGBoost, Optuna, Joblib
* **Quality & Testing:** Pytest, HTTPX (TestClient)
* **DevOps & CI/CD:** Docker, GitHub Actions (CI pipeline), Render (CD deployment)

## Project Architecture
The project strictly separates the research/training phase from the production inference environment:

* `app/`: Production API source code.
  * `main.py`: FastAPI application, route definitions, and Pydantic schemas.
  * `inference.py`: Singleton model loading and prediction logic.
  * `processing.py`: Unitary, deterministic feature engineering for incoming JSON payloads.
* `models_export/`: Serialized model artifacts (`.joblib`). Contains fitted preprocessors, Ridge/XGBoost models, and blending weights. *(Note: Tracked via Git for academic deployment purposes).*
* `tests/`: Automated test suite (`pytest`) validating mathematical coherence, API contracts, and HTTP 422 error handling.
* `training/`: Original data science research scripts (`step_final.py`). Features advanced Actuarial/ML logic (capping, blending, Duan's correction).
* `render.yaml`: Render deployment configuration (Blueprint).
* `.github/workflows/`: Continuous Integration (CI) configuration to automatically install dependencies, run tests, and build the Docker image on every push.

## Actuarial & Machine Learning Logic
The API calculates the **pure premium** of an auto insurance contract on a unitary basis, strictly following the academic contract: `premium = frequency * amount`.

* **Frequency Model:** Optimized blending between a well-calibrated Logistic Regression and an XGBClassifier (optimized via Optuna).
* **Severity Model (Amount):** Blending between a Log-transformed Ridge Regression (with Duan's smearing estimator correction) and an XGBRegressor (Gamma loss function). Trained exclusively on the 99th percentile capped data to capture attrition risk.
* **Feature Engineering:** Dynamic features (e.g., zip code parsing, power-to-weight ratios, driving license age calculation) are encapsulated directly within the API's processing pipeline.

## API Contract (Endpoints)
The API strictly enforces input validation. Missing or wrongly typed fields (e.g., passing a string instead of an integer) will systematically trigger an `HTTP 422 Unprocessable Entity` error, preventing internal server errors (500).

| Endpoint | Method | Input | Output Example |
| :--- | :--- | :--- | :--- |
| `/health` | `GET` | None | `{"status": "ok"}` |
| `/predict_frequency` | `POST` | Contract JSON | `{"frequency": 0.12}` |
| `/predict_amount` | `POST` | Contract JSON | `{"amount": 1500.0}` |
| `/predict` | `POST` | Contract JSON | `{"frequency": 0.12, "amount": 1500.0, "premium": 180.0}` |

*Interactive documentation is available natively at `/docs` (Swagger UI) and `/openapi.json`.*

## Input format (JSON contract)
Each prediction route takes **one contract**. The full schema, with a description of every field, is shown in Swagger (`/docs`, `ContractInput` schema).

| Field | Type | Accepted values |
| :--- | :--- | :--- |
| `id_client`, `id_vehicule`, `id_contrat` | non-empty string | free (not used by the models) |
| `bonus` | number | 0.5 to 3.5 (1 = neutral) |
| `type_contrat` | string | `Mini`, `Median1`, `Median2`, `Maxi` |
| `duree_contrat`, `anciennete_info` | integer | 0 to 100 |
| `freq_paiement` | string | `Monthly`, `Quarterly`, `Biannual`, `Yearly` |
| `paiement` | string | `Yes`, `No` |
| `utilisation` | string | `WorkPrivate`, `Retired`, `Professional`, `AllTrips` |
| `code_postal` | 5-character **string** | `"06088"` (leading zero kept), Corsica `"2A004"` / `"2B033"` |
| `conducteur2` | string | `Yes`, `No` |
| `age_conducteur1` | integer | 18 to 120 |
| `age_conducteur2` | integer | **0 if `conducteur2 = "No"`**, otherwise 18 to 120 |
| `sex_conducteur1` | string | `"M"` or `"F"` only |
| `sex_conducteur2` | string or `null` | **`null` if `conducteur2 = "No"`**, otherwise `"M"` or `"F"` |
| `anciennete_permis1` | integer | 0 to 111 (**111 = unknown**, data convention) |
| `anciennete_permis2` | integer | **0 if `conducteur2 = "No"`**, 111 = unknown |
| `anciennete_vehicule` | integer or `null` | 0 to 100, `null` = unknown (replaced by the median) |
| `cylindre_vehicule` | integer | 0 to 10,000 cm³ (0 = not provided) |
| `din_vehicule` | integer | 1 to 2,000 hp |
| `essence_vehicule` | string | `Gasoline`, `Diesel`, `Hybrid` |
| `marque_vehicule` | string | e.g. `"PEUGEOT"` (case-insensitive); a rare or unknown brand is grouped as `"Autre"`, as in training |
| `modele_vehicule` | string | free (not used by the models) |
| `debut_vente_vehicule`, `fin_vente_vehicule` | integer | 0 to 100 (not used by the models) |
| `vitesse_vehicule` | integer | 1 to 500 km/h |
| `type_vehicule` | string | `Tourism`, `Commercial` |
| `prix_vehicule` | number | 0 to 10,000,000 € (0 = not provided) |
| `poids_vehicule` | integer | 0 to 20,000 kg (0 = not provided) |

Bounds are wider than the observed values: all 100,000 contracts of `train.csv` and `test.csv` are accepted.

**Validation rules (error 422, never 500)**: missing field, wrong type (`"30"` or `true` for a number, `30.7` for an integer), unknown category (`"X"` for sex), out-of-range value, empty body or invalid JSON, inconsistent second driver (`conducteur2 = "No"` with an age). An integer sent as an exact decimal (`10.0`) is accepted, because pandas exports columns containing missing values this way.

## Tests
`tests/test_api.py` contains about 380 tests in 8 parts: monitoring, Swagger documentation, API contract (exact keys, JSON numbers, `premium = frequency * amount`, determinism), nullable fields and domain conventions, vehicle brands, invalid inputs (always 422), robustness on 200 random contracts, and unit tests of `processing` / `inference` (same encoding as in training, regression test on the coursework example).

## Local Deployment

**1. Clone the repository and initialize the environment**
```bash
git clone https://github.com/matteo-dev/IA_Industrialisation_Auto_Insurance_Pricing_ML.git
cd IA_Industrialisation_Auto_Insurance_Pricing_ML
uv sync
```

**2. Run locally with Uvicorn (Development mode)**
```bash
uv run uvicorn app.main:app --reload
```
Access the Swagger UI at: `http://127.0.0.1:8000/docs`

**3. Run the automated test suite**
```bash
uv run pytest tests/
```

**4. Build and run with Docker (Production mode)**
```bash
docker build -t api-tarification-auto:latest .
docker run -p 8000:8000 api-tarification-auto:latest
```

## Cloud Deployment (CI/CD)
This project is configured for continuous deployment on **Render**. Every push to the `main` branch triggers a GitHub Action that runs the unit tests. If successful, Render automatically pulls the latest commit, builds the Docker image, and deploys it on a live URL.

**Render configuration** (`render.yaml`, read by Render as a *Blueprint*):
* **Runtime**: Docker, from the root `Dockerfile` (no build or start command to fill in).
* **Port**: the API listens on the `PORT` environment variable (8000 by default), which Render provides and the `Dockerfile` reads at startup.
* **Health check**: `/health`. Render only routes traffic to a new version once this route answers 200.
* **Auto-deploy**: `autoDeployTrigger: checksPass`, i.e. only when the GitHub CI is green.
* **Free plan**: the service sleeps after 15 minutes without requests, so the first call afterwards can take about a minute (container start and model loading).

**Setup**: Render → *New* → *Blueprint* → select this repository. Render reads `render.yaml` and creates the service. Check: `GET https://<service>.onrender.com/health` returns `{"status": "ok"}`, and the documentation is at `https://<service>.onrender.com/docs`.
