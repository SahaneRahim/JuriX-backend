"""Configuration application - Toutes les variables d'environnement."""

from pydantic import field_validator
from pydantic_settings import BaseSettings

# Parametres TLS que le pilote asyncpg ne connait pas. Au niveau module et non
# dans la classe : pydantic transforme tout attribut de classe a underscore
# initial en ModelPrivateAttr, qui n'est pas iterable.
_PARAMS_QUI_FUIENT = ("sslmode", "channel_binding")


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

    # Gemini API (LLM for RAG)
    GEMINI_API_KEY: str = ""
    # "gemini-3-flash" n'existe PAS : l'API repond 404 NOT_FOUND. Le RAG ne
    # pouvait donc pas produire une seule reponse — chaque appel echouait avant
    # meme d'atteindre le modele. Verifie contre ListModels sur ce compte :
    # gemini-2.5-flash repond 404 lui aussi, seuls les modeles ci-dessous
    # acceptent generateContent.
    #   gemini-3-flash-preview   (retenu : le plus capable disponible ici)
    #   gemini-3.1-flash-lite    (repli stable si l'apercu est retire)
    #   gemini-2.5-flash-lite
    GEMINI_MODEL: str = "gemini-3-flash-preview"
    GEMINI_EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    # Dimension demandee a l'API (output_dimensionality) : la sortie native du
    # modele, sans troncature. Le plafond de 2000 dimensions de pgvector ne
    # s'applique qu'au type `vector` ; l'index est pose sur une expression
    # `halfvec(3072)`, qui monte a 4000 (migration f5a6b7c8d9e0). On garde donc
    # la pleine precision en stockage et un index utilisable.
    # Doit rester egal a la dimension declaree sur Article.embedding.
    EMBEDDING_DIM: int = 3072

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

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields from .env


settings = Settings()
