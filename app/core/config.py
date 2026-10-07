"""Configuration application - Toutes les variables d'environnement."""

from typing import Literal, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings

# Parametres TLS que le pilote asyncpg ne connait pas. Au niveau module et non
# dans la classe : pydantic transforme tout attribut de classe a underscore
# initial en ModelPrivateAttr, qui n'est pas iterable.
_PARAMS_QUI_FUIENT = ("sslmode", "channel_binding")

# Dimension de la colonne articles.embedding, `vector(768)`. La migration la
# fige en dur, comme toute migration ; Article.embedding la repete, et un test
# verifie que modele, configuration et service s'accordent.
_DIMENSION_DU_SCHEMA = 768


class Settings(BaseSettings):
    """Settings de l'application."""

    # App
    APP_NAME: str = "JuriX API"
    VERSION: str = "2.1.0"
    DEBUG: bool = True
    ENVIRONMENT: str = "development"

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://jurix:jurix_dev_password_change_in_prod@localhost:5432/jurix_db"

    # Recherche et cache assures par PostgreSQL (tsvector, pg_trgm, query_cache).

    # LLM Provider for Chat (RAG)
    LLM_PROVIDER: Literal["mistral", "gemini"] = "mistral"

    # Mistral AI (LLM for RAG Chat - rapide, bilingue FR/EN, sans thinking bloquant)
    MISTRAL_API_KEY: str = ""
    MISTRAL_MODEL: str = "ministral-14b-latest"
    MISTRAL_TIMEOUT_S: float = 60.0

    # Groq (LPU pour Routage / Système 1 ultra-rapide < 100 ms)
    GROQ_API_KEY: str = ""
    # Modele du classement d'intention (chaque message du chat).
    GROQ_MODEL: str = "qwen/qwen3.8-27b"
    # Modele du classement des lois (pipeline, reclassement). Distinct de celui
    # du chat : chez Groq, les quotas sont PAR MODELE, et un reclassement du
    # corpus ne doit pas vider celui des utilisateurs. gpt-oss-120b est en
    # production ; qwen3.8-27b n'est qu'en « Preview ».
    GROQ_MODEL_CLASSEMENT: str = "openai/gpt-oss-120b"
    GROQ_TIMEOUT_S: float = 10.0
    # Plafonds du palier gratuit, PAR MODELE (verifies le 07/10/2026 dans les
    # en-tetes x-ratelimit-* et la documentation). Le limiteur fait attendre
    # avant d'envoyer plutot que de collectionner les 429.
    GROQ_REQUETES_PAR_MINUTE: int = 30
    # Departs autorises d'affilee avant que l'espacement (2 s a 30/min)
    # s'impose : deux messages simultanes ne doivent pas attendre l'un l'autre.
    GROQ_RAFALE: int = 5
    GROQ_JETONS_PAR_MINUTE: int = 8_000
    GROQ_JETONS_ENTREE_PAR_MINUTE: int = 7_000
    GROQ_REQUETES_PAR_JOUR: int = 1_000
    # Absent des en-tetes : le depassement ne se voit qu'au message du 429.
    GROQ_JETONS_PAR_JOUR: int = 200_000
    # Au-dela de cette attente imposee par un 429, c'est un quota epuise, pas
    # une saturation : l'appel echoue (GroqQuotaError) et plus aucun n'est
    # envoye au modele avant l'echeance.
    GROQ_ATTENTE_MAX_429_S: float = 120.0

    # Gemini API (optionnel désormais)
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3-flash-preview"
    GEMINI_EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    # Dimension des embeddings : 768, et aucune autre (validateur plus bas).
    # C'est la sortie native d'EmbeddingGemma, et une dimension que
    # gemini-embedding-001 produit sur demande (output_dimensionality,
    # troncature Matryoshka que le service renormalise). Un seul schema pour
    # les deux fournisseurs : passer de l'un a l'autre ne migre rien.
    # Sous le plafond de 2000 du type `vector`, l'index HNSW se pose sur la
    # colonne elle-meme (migration c4d5e6f7a8b9).
    EMBEDDING_DIM: int = _DIMENSION_DU_SCHEMA

    # ---- Fournisseur d'embeddings (app/services/embedding_service.py) ----
    # "gemma", le defaut : EmbeddingGemma execute EN LOCAL par onnxruntime,
    # gratuit, sans reseau. Il exige le modele sur disque (GEMMA_MODEL_DIR) et
    # ~1,5 Go de RAM dans le processus de l'API. Gemini ne sert plus alors
    # qu'au chat. "gemini" : l'API payante, a 768 dimensions elle aussi.
    #
    # Changer de fournisseur change l'espace vectoriel : TOUT le corpus doit
    # etre re-encode (scripts/regenerate_embeddings.py --all, qui reconnait
    # les vecteurs d'un autre fournisseur a leur colonne
    # articles.embedding_model). D'ici la, GET /search/health repond
    # « degraded ».
    #
    # PIEGE : `class Config` porte `extra = "ignore"`. Une cle mal
    # orthographiee dans .env est ignoree EN SILENCE et le defaut s'applique.
    EMBEDDING_PROVIDER: Literal["gemini", "gemma"] = "gemma"

    # Dossier du modele EmbeddingGemma exporte en ONNX
    # (onnx-community/embeddinggemma-300m-ONNX). Le `.onnx` et son
    # `.onnx_data` doivent rester DANS LE MEME DOSSIER et sous leur nom
    # d'origine : le graphe reference ses poids par nom. Telecharger avec
    # `hf download ... --local-dir`, jamais dans le cache Hugging Face, dont
    # les blobs partages separent les deux fichiers — onnxruntime refuse alors
    # le chargement (« External data path escapes model directory »).
    GEMMA_MODEL_DIR: str = "models/embeddinggemma-300m-onnx"
    # int8, et non q4 — MESURE, pas preference. Sur cette machine :
    #   int8 : requete 196 ms, 1,5 Go de RAM, quasi identique au fp32 de
    #          reference (ecart de score <= 0,005, meme classement) ;
    #   q4   : requete 35 ms, 434 Mo, mais sur 180 articles du Code Minier et
    #          15 questions, cosinus moyen 0,954 avec l'int8, premier resultat
    #          DIFFERENT pour 2 questions sur 15, ~12 % du top-10 change.
    # La fidelite prime pour un corpus juridique. q4 reste l'option mesuree
    # d'un hebergement contraint en memoire : l'empreinte du fournisseur rend
    # le changement sur — il declenche une regeneration, jamais un melange.
    # Les variantes fp16 (model_fp16, model_q4f16) ne sont pas supportees.
    GEMMA_ONNX_FILE: str = "onnx/model_quantized.onnx"
    # Revision EPINGLEE du depot onnx-community. Un nouvel export changerait
    # les vecteurs des requetes, mais pas ceux des documents deja stockes : les
    # deux espaces divergeraient sans erreur. Elle entre dans l'empreinte du
    # fournisseur, donc dans la cle de cache.
    GEMMA_REVISION: str = "5090578d9565bb06545b4552f76e6bc2c93e4a66"
    # Contexte du modele, <bos> et <eos> compris. Au-dela, onnxruntime leve une
    # erreur RotaryEmbedding : la troncature est obligatoire, et journalisee.
    GEMMA_MAX_TOKENS: int = 2048
    # Fils de calcul d'onnxruntime. 0 = coeurs physiques moins un, pour laisser
    # respirer la recherche pendant une ingestion.
    GEMMA_INTRA_OP_THREADS: int = 0

    # Delai maximal d'un appel Gemini, en SECONDES. Sans lui, le client
    # n'impose AUCUN delai : une prise reseau qui ne repond plus bloque
    # indefiniment, sans exception, donc sans declencher la moindre reprise.
    # Observe en conditions reelles : une ingestion figee 40 minutes sur un
    # appel d'embeddings, processus vivant, zero UC consommee.
    GEMINI_TIMEOUT_S: int = 120

    # ---- Re-ranking des chunks (app/services/reranker.py) ----
    # Etage 1 : traits lexicaux, sans reseau ni dependance. Quelques
    # millisecondes, actif par defaut.
    RERANK_ENABLED: bool = True
    # Etage 2 : notation des meilleurs chunks par Gemini. Ajoute un appel
    # facture et 400 a 900 ms sur le chemin critique, d'ou le defaut a False :
    # a n'activer qu'apres l'avoir vu gagner sur le lot d'evaluation tenu a
    # l'ecart.
    RERANK_LLM_ENABLED: bool = False
    RERANK_LLM_TOP_N: int = 20
    RERANK_LLM_TIMEOUT_S: float = 4.0

    # ---- Routage d'intention (app/services/intent_classifier.py) ----
    # Interrupteur de repli. A False, `ask()` retrouve exactement le
    # comportement anterieur : toute question part au RAG. C'est le levier de
    # retour arriere sans redeploiement, sur le modele de RERANK_LLM_ENABLED.
    INTENT_ROUTING_ENABLED: bool = True
    # MESURE, PAS ESTIME. Premiere valeur posee au jugé : 3,0 s, « large pour
    # un verdict d'un seul mot ». Mesure sur gemini-3-flash-preview, six
    # classifications reelles : 2583, 2826, 2983, 3309, 3843, 6531 ms —
    # mediane 3,3 s, maximum 6,5 s. A 3,0 s, CINQ appels sur six expiraient,
    # et une expiration retombe silencieusement sur "juridique" : le routeur
    # aurait cesse de router sans qu'aucune erreur, aucun test et aucun
    # journal ne le signale.
    #
    # 10 s couvre le maximum observe avec de la marge, tout en restant douze
    # fois sous GEMINI_TIMEOUT_S : une prise reseau qui ne repond plus est
    # abandonnee vite, au lieu de bloquer deux minutes.
    INTENT_TIMEOUT_S: float = 10.0
    # PAS 16 jetons, malgre une sortie d'un seul mot. Le modele est un modele
    # a raisonnement : sa reflexion est facturee sur ce budget AVANT la
    # premiere ligne de sortie. Le piege est deja documente sur
    # ANSWER_MAX_TOKENS (rag_service.py) et il est ici PIRE, parce que
    # silencieux : un budget trop court rend une reponse vide, donc le repli
    # "juridique" a chaque appel, donc un routeur qui ne route jamais, sans
    # qu'aucun test ne le voie — ils doublent tous le modele.
    INTENT_MAX_TOKENS: int = 1024
    # « groq » : classement par Groq (GROQ_MODEL), consignes courtes et schema
    # strict, verdicts gardes en cache ; toute panne rend « juridique ».
    # « llm » : classement par le modele du chat (LLM_PROVIDER), un appel de
    # plus par question. L'ancienne valeur « gemini » y est ramenee.
    # Le classement local (« local ») a ete retire.
    INTENT_CLASSIFIER: Literal["groq", "llm"] = "groq"
    # Le classement retarde chaque message du chat : au-dela, « juridique ».
    GROQ_INTENTION_TIMEOUT_S: float = 3.0

    # ---- Reflexion du modele (Gemini 3) ----
    # La reflexion interne fait l'essentiel du temps d'attente. MESURE sur
    # gemini-3-flash-preview, une question au chat, 37 s au total :
    # classification 8,5 s (232 jetons de reflexion pour une sortie de 18),
    # recherche 0,26 s, reponse 28 s (3 104 jetons de reflexion pour 247 de
    # reponse). En « minimal », la classification descend a 3,6 s.
    #
    # La REPONSE garde la reflexion du modele, par choix de l'utilisateur.
    # Compare sur deux questions, meme contexte : « low » repondait en 3 a 5 s
    # au lieu de 11 a 12, avec les memes articles cites, mais un peu moins
    # precis par endroits. « low » reste disponible par .env.
    # Valeurs : minimal, low, medium, high ; vide = choix du modele.
    # N'agit que sur les modeles Gemini 3 (les 2.5 reglent un budget, pas un
    # niveau).
    GEMINI_REFLEXION_REPONSE: Optional[Literal["minimal", "low", "medium", "high"]] = None
    GEMINI_REFLEXION_CLASSIFICATION: Optional[Literal["minimal", "low", "medium", "high"]] = "minimal"

    # ---- Fusion hybride ----
    # Valeurs par defaut NON calibrees sur ce corpus : RRF_K = 60 vient du
    # papier d'origine sur des runs TREC. A remplacer par les valeurs issues
    # de scripts/eval/run_eval.py, en citant le fichier de run en commentaire.
    # Seuil de `word_similarity` pour la recherche floue sur les titres.
    # Le defaut PostgreSQL est 0,6 : mesure sur le corpus, il laisse passer
    # « nominaton » (0,700) mais perd « nominasion » (0,571). 0,5 rattrape les
    # deux sans faire entrer un seul faux positif (« fonciere » plafonne a
    # 0,333 et reste dehors).
    TITLE_TRIGRAM_THRESHOLD: float = 0.5

    RRF_K: int = 60
    TEXT_WEIGHT: float = 0.4
    SEMANTIC_WEIGHT: float = 0.6

    # LLAMA_CLOUD_API_KEY et LLAMA_PARSE_TIER ont ete retires : l'extraction
    # passe par Gemini (app/services/pdf_extraction_service.py), qui rend un
    # numero de page explicite la ou LlamaParse n'en donnait aucun.
    # Cache d'extraction par sha256 — evite de repayer un fichier deja traite
    OCR_CACHE_DIR: str = "./data/ocr_cache"

    # ---- Documents d'origine (PDF) ----
    # Racine HTTPS du magasin de documents. VIDE = les PDF sont lus sur le
    # disque local, comportement historique : le developpement ne change pas et
    # aucun test existant ne bouge. NON VIDE = ils sont recuperes en HTTPS.
    #
    # Meme discipline que GOOGLE_CLIENT_ID plus bas : vide, la fonctionnalite
    # est INERTE, elle ne degrade rien. C'est ce qui rend le magasin
    # interchangeable — Supabase Storage aujourd'hui, un autre demain, sans
    # toucher une ligne de code.
    DOCUMENTS_BASE_URL: str = ""
    # Plafond du cache disque des documents telecharges, en Mo. 0 le desactive,
    # et chaque rendu de page retelecharge alors le PDF entier.
    DOCUMENTS_CACHE_MAX_MB: int = 96
    # Repertoire du cache. /tmp est inscriptible dans le conteneur, et son
    # caractere ephemere est sans consequence : le cache se reconstruit.
    DOCUMENTS_CACHE_DIR: str = "/tmp/jurix-docs"
    DOCUMENTS_FETCH_TIMEOUT_S: int = 30
    # ---- Extraction des PDF ----
    # "docling", le defaut : en local, gratuit, OCR pleine page sur TOUTES les
    # pages (reglages EXTRACTION_* ci-dessous). "gemini" : l'API payante,
    # reglee par les PDF_EXTRACTION_*. Changer de moteur ne relit pas les
    # extractions de l'autre : chacun a son cache. Les scripts d'ingestion du
    # corpus utilisent Docling quel que soit ce reglage.
    PDF_EXTRACTION_ENGINE: Literal["docling", "gemini"] = "docling"
    # ---- Extraction locale par Docling (app/services/docling_extraction.py) ----
    # Prefixe EXTRACTION_ et non DOCLING_ : Docling lit lui-meme les variables
    # d'environnement DOCLING_* (AcceleratorOptions). Une valeur exportee pour
    # JuriX — DOCLING_NUM_THREADS=0, « coeurs physiques » ici — changeait aussi
    # Docling : verifie, son lecteur de PDF tournait alors sans fin.
    #
    # Moteur OCR. "rapidocr_torch" (defaut) : PP-OCRv6 sur torch, donc sur le
    # GPU. MESURE sur 97 pages du corpus contre Gemini et LlamaParse : il lit
    # les numeros d'article sans erreur la ou Tesseract en fausse (« ARTICLE
    # 16 » lu « 13 », « 42 » lu « 41 ») — inacceptable pour des citations —,
    # a 3,8 s par page (environ 15 h pour le corpus). "rapidocr" : le meme sur
    # onnxruntime, CPU seul, 5,5 s par page. "tesseract" : binaire
    # TESSERACT_PATH, langue fra, plus juste sur les mots mais faux sur les
    # chiffres.
    EXTRACTION_OCR_ENGINE: Literal["rapidocr_torch", "rapidocr", "tesseract"] = "rapidocr_torch"
    # "auto" prend le GPU des que torch le voit.
    EXTRACTION_DEVICE: str = "auto"
    # Pages converties et mises en cache ensemble. Petit : une interruption ne
    # coute que le lot en cours, et la memoire reste bornee sur les documents
    # de plusieurs centaines de pages. Changer cette valeur ne perd pas le
    # cache : les lots deja faits se relisent quelle que soit leur taille.
    EXTRACTION_PAGES_PAR_LOT: int = 8
    # Fils de calcul. 0 = coeurs physiques.
    EXTRACTION_FILS: int = 0
    # Delai maximal d'un lot. Au-dela, les pages restantes sont en echec et le
    # lot sera rejoue ; le superviseur tue un processus bloque au-dela de ce
    # delai et d'une marge.
    EXTRACTION_LOT_TIMEOUT_S: float = 900.0
    # Aucun appel reseau pour les poids : ils sont dans le cache Hugging Face.
    # Sans cela, chaque initialisation interroge le Hub et retelecharge en
    # silence si une revision change en amont — l'extraction ne serait plus
    # reproductible, et une coupure reseau la ferait echouer.
    EXTRACTION_HORS_LIGNE: bool = True

    # Pages envoyees par appel a Gemini. La limite du modele est de 65 536
    # jetons EN SORTIE ; a ~2400 caracteres par page, 20 pages produisent
    # ~13 000 jetons, avec de la marge pour la reflexion interne. Le palier
    # gratuit plafonnant a 20 appels par jour, agrandir le lot economise des
    # appels — au risque de tronquer si les pages sont denses.
    PDF_EXTRACTION_PAGES_PER_CALL: int = 20
    # Delai d'un appel d'extraction. Distinct de GEMINI_TIMEOUT_S : un lot de
    # vingt pages scannees demande plusieurs minutes, la ou une question de RAG
    # se compte en secondes. Mesure : un lot de 9,8 Mo depassait 120 s.
    PDF_EXTRACTION_TIMEOUT_S: int = 600
    # Poids maximal d'un lot. Le nombre de pages ne suffit pas : vingt pages
    # scannees pesent 10 Mo la ou vingt pages de texte en pesent 1. Au-dela, le
    # lot est redecoupe.
    PDF_EXTRACTION_MAX_BATCH_MB: float = 6.0

    # CORS — comma-separated list of extra allowed origins for production
    # Example: https://jurix.vercel.app,https://www.jurix.cm
    ALLOWED_ORIGINS: str = ""

    # Security
    SECRET_KEY: str = "dev_secret_key_change_in_production_with_openssl_rand_hex_32"
    ALGORITHM: str = "HS256"

    # DUREE DE SESSION, en minutes, et il y en a DEUX a dessein.
    #
    # 30 minutes rendait le produit inutilisable : l'interet d'un compte est de
    # retrouver ses conversations, or l'utilisateur etait deconnecte sans
    # preavis en pleine session. 30 jours est la norme des produits grand
    # public.
    #
    # Mais le meme reglage regissait aussi l'administration, et les JWT sont
    # SANS ETAT ici : /auth/logout ne revoque rien, le jeton vit en clair dans
    # localStorage, il n'existe aucune liste de revocation. Un jeton superadmin
    # vole serait donc exploitable un mois. D'ou la seconde valeur, bien plus
    # courte, appliquee des que le compte porte un role privilegie.
    #
    # Levier d'urgence a connaitre : `get_current_user` relit l'utilisateur en
    # base a chaque requete et leve 403 si `is_active` est faux. DESACTIVER UN
    # COMPTE REVOQUE DONC SES JETONS IMMEDIATEMENT.
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 43200  # 30 jours
    ADMIN_TOKEN_EXPIRE_MINUTES: int = 720  # 12 heures

    # Identifiant client OAuth de la console Google Cloud (type « Web
    # application »). VIDE = connexion Google desactivee, et POST /auth/google
    # repond 503.
    #
    # Il n'y a PAS de client secret : le mode « credential » de Google Identity
    # Services rend le jeton d'identite a une fonction JavaScript, sans
    # redirection. Un secret ici ne servirait a rien.
    #
    # PIEGE : `class Config` ci-dessous porte `extra = "ignore"`. Une cle du
    # .env qui n'est pas DECLAREE dans cette classe est silencieusement
    # ignoree — c'est pourquoi celle-ci doit y figurer.
    GOOGLE_CLIENT_ID: str = ""

    # ---- Envoi de courriel (Brevo) ----
    # VIDE = envoi desactive. L'inscription, la connexion et tout le reste
    # continuent de fonctionner ; seuls la verification d'adresse et le lien de
    # reinitialisation ne partent pas. Meme discipline que GOOGLE_CLIENT_ID.
    #
    # Le transport est l'API HTTPS de Brevo, JAMAIS smtplib : les hebergeurs
    # gratuits (Render, Railway) bloquent les ports sortants 25, 465 et 587.
    BREVO_API_KEY: str = ""
    # L'adresse VERIFIEE dans Brevo. Sur un domaine gratuit (gmail.com), Brevo
    # ne peut pas l'authentifier et reecrit l'expediteur en @brevosend.com —
    # d'ou la mention "verifiez vos spams" affichee a l'utilisateur.
    BREVO_SENDER_EMAIL: str = ""
    BREVO_SENDER_NAME: str = "JuriX"
    BREVO_TIMEOUT_S: int = 15
    # Le palier gratuit plafonne a 300 messages par jour, PARTAGES entre
    # transactionnel et campagnes. On s'arrete avant, pour garder de la marge.
    BREVO_BUDGET_QUOTIDIEN: int = 200

    # Racine des liens contenus dans les courriels. VIDE = le service REFUSE
    # d'envoyer un message porteur d'un lien, plutot que d'expedier un lien
    # mort : sans elle le lien se construirait sur l'hote de l'API, qui ne sert
    # aucune page.
    FRONTEND_BASE_URL: str = ""

    # Intervalle de purge des caches, en secondes. Sorti de main.py pour etre
    # reglable : une constante de module est figee a l'import, donc intestable.
    # Cette boucle a un effet de bord utile en production — elle touche la base
    # regulierement, ce qui empeche un hebergeur gratuit de mettre le projet en
    # pause pour inactivite.
    CACHE_CLEANUP_INTERVAL_S: int = 900

    # QODO_API_KEY, ZEROSTEP_API_KEY et CORS_ORIGINS ont ete retires : les deux
    # premiers n'etaient lus nulle part, et main.py lit ALLOWED_ORIGINS, pas
    # CORS_ORIGINS — cette liste, qui contenait un joker "*", ne s'appliquait a
    # rien tout en donnant a lire le contraire.

    # Upload limits
    MAX_UPLOAD_SIZE: int = 1073741824  # 1GB in bytes for batch upload

    # OCR — chemin du binaire tesseract. Vide par defaut : sous Linux et dans
    # l'image Docker il est sur le PATH. Le defaut precedent etait un chemin
    # Windows, et n'etait de toute facon lu par personne.
    TESSERACT_PATH: str = ""

    # ---- Gardes ----

    @field_validator("DATABASE_URL")
    @classmethod
    def _refuser_les_parametres_qui_fuient(cls, v: str) -> str:
        """
        Interdit dans l'URL les parametres TLS que le pilote asyncpg ne connait pas.

        MESURE, pas suppose. Le dialecte asyncpg de SQLAlchemy tient une liste
        FERMEE de parametres qu'il consomme et convertit ; tout le reste part
        tel quel en argument nomme vers `asyncpg.connect()`, dont la signature
        n'a ni `sslmode`, ni `channel_binding`, ni `**kwargs` :

            TypeError: connect() got an unexpected keyword argument 'sslmode'

        L'erreur ne survient qu'au PREMIER acces a la base, donc bien apres un
        demarrage en apparence reussi. Or c'est exactement ce que proposent les
        boutons « copier » de Neon et de Supabase, et ce que .env.example
        recommandait. D'ou ce refus au chargement de la configuration, avec le
        remede dans le message.

        `prepared_statement_cache_size` n'est PAS refuse : celui-la est bien
        consomme par le dialecte (asyncpg.py, `kw.pop` puis `coerce_kw_type`),
        et il est INDISPENSABLE au pooler Supabase en mode transaction, qui ne
        supporte pas les instructions preparees.
        """
        if not v.startswith("postgresql+asyncpg://"):
            return v
        fautifs = [p for p in _PARAMS_QUI_FUIENT if f"{p}=" in v]
        if fautifs:
            raise ValueError(
                f"DATABASE_URL porte {', '.join(fautifs)}, que le pilote asyncpg "
                "refuse : la connexion echouerait avec « connect() got an "
                "unexpected keyword argument ». Retirez ce parametre de l'URL et "
                "posez la variable d'environnement PGSSLMODE=require, lue aussi "
                "bien par asyncpg que par psycopg2."
            )
        return v

    @field_validator("INTENT_CLASSIFIER", mode="before")
    @classmethod
    def _classement_d_intention(cls, v):
        """
        « gemini » designait le classement par le modele du chat : c'est « llm ».
        « local » est refuse avec la raison, plutot qu'avec l'erreur generique
        de pydantic, illisible pour qui a garde un vieux .env.
        """
        if not isinstance(v, str):
            return v
        valeur = v.strip().lower()
        if valeur == "gemini":
            return "llm"
        if valeur == "local":
            raise ValueError(
                "INTENT_CLASSIFIER=local n'existe plus : le classement local de "
                "l'intention a ete retire. Utilisez groq (defaut) ou llm."
            )
        return valeur

    @field_validator("EMBEDDING_DIM")
    @classmethod
    def _exiger_la_dimension_du_schema(cls, v: int) -> int:
        """
        Refuse toute dimension autre que celle de la colonne.

        Le piege vise est concret : un .env d'avant la bascule porte
        EMBEDDING_DIM=3072, et il l'emporte sur le defaut. Sans ce refus, le
        service demanderait des vecteurs 3072 que la colonne vector(768)
        rejette — a la premiere ecriture pour l'ingestion, a la premiere
        question pour la recherche, que l'hybride avale en retombant sans
        bruit sur le plein texte. Mieux vaut ne pas demarrer.
        """
        if v != _DIMENSION_DU_SCHEMA:
            raise ValueError(
                f"EMBEDDING_DIM={v}, or la colonne articles.embedding est en "
                f"vector({_DIMENSION_DU_SCHEMA}) et les deux fournisseurs y sont "
                "regles. Retirez EMBEDDING_DIM du .env : un .env d'avant la "
                "bascule porte encore 3072."
            )
        return v

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields from .env


settings = Settings()
