"""
Routes d'authentification.

`app/core/auth.py` implémentait déjà l'intégralité de la chaîne JWT — hachage
bcrypt, création de jeton, dépendances par rôle — mais n'était importé nulle
part, et aucune route de connexion n'existait. Les endpoints « admin » passaient
par deux stubs renvoyant un dictionnaire en dur.

Deux points d'entrée de connexion, volontairement :

- `POST /login` accepte un formulaire OAuth2. C'est exactement l'URL déclarée
  par `oauth2_scheme` (`tokenUrl="/api/v1/auth/login"`), donc le bouton
  *Authorize* de /docs fonctionne — précieux pour administrer l'instance tant
  que l'interface d'administration n'est pas terminée.
- `POST /login/json` accepte du JSON, plus naturel pour le front SvelteKit.

Les deux partagent le même `_authenticate`.

Deux points d'entrée de création de compte :

- `POST /signup` — inscription publique : nom, adresse, mot de passe. Le rôle
  est écrit `user` par le serveur ; `SignupRequest` n'expose tout simplement
  pas ce champ, et refuse les clés inconnues.
- `POST /google` — connexion par Google Identity Services. Crée le compte à la
  première venue, ou le **lie** à un compte existant portant la même adresse.

L'ancienne docstring justifiait l'absence d'inscription publique par la
facturation Gemini : « une inscription ouverte donnerait à n'importe qui
l'accès au RAG ». CE RAISONNEMENT ÉTAIT DÉJÀ CADUC quand il a été écrit :
`POST /api/v1/rag/ask` n'a aucune dépendance d'authentification ni aucune
limite de débit — la porte est ouverte sans compte. L'inscription ne change
rien à l'exposition ; elle change une chose, et dans le bon sens : le trafic
devient **attribuable** (`Conversation.user_id` enfin écrit), ce qui est la
condition préalable à toute limitation ultérieure.

Le seul rempart efficace serait une limite par IP au niveau du proxy sur
`/api/v1/rag/*` — elle couvrirait le trafic anonyme, là où l'abus se produit
réellement. Une limite par compte punirait les inscrits en laissant les
anonymes libres.

Il n'existe aucune infrastructure d'envoi d'e-mail : donc pas de vérification
d'adresse, et **pas de réinitialisation de mot de passe**. La seule remise à
zéro passe par `PUT /api/v1/admin/users/{id}`. Pour un compte lié à Google, se
reconnecter par Google reste possible et fait office de récupération.

Author: JuriX Team
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import (
    create_access_token,
    duree_de_session,
    get_current_active_user,
    hash_password,
    verify_password,
)
from app.core.database import get_db
from app.core.google_identity import (
    GoogleInjoignable,
    GoogleNonConfigure,
    JetonGoogleInvalide,
    verifier_jeton_google,
)
from app.models.email_token import (
    DUREE_REINITIALISATION,
    DUREE_VERIFICATION,
    REINITIALISATION_MOT_DE_PASSE,
    VERIFICATION_ADRESSE,
)
from app.models.user import User
from app.schemas.user import (
    DemandeReinitialisation,
    GoogleAuthRequest,
    MessageReponse,
    ReinitialisationMotDePasse,
    SignupRequest,
    UserLogin,
    UserResponse,
    UserWithToken,
    VerificationAdresse,
)
from app.services import email_service, email_templates
from app.services.email_tokens_service import (
    adresse_du_client,
    compte_etrangle,
    consommer_jeton,
    creer_jeton,
    invalider_les_autres,
    ip_etranglee,
)
from app.services.user_identity import normaliser_email, username_pour_email

logger = logging.getLogger(__name__)

router = APIRouter()

# Message unique pour « email inconnu » et « mot de passe faux » : les
# distinguer permettrait d'énumérer les comptes existants.
_INVALID = "Identifiants invalides"


async def _authenticate(db: AsyncSession, email: str, password: str) -> User:
    """
    Vérifie un couple email / mot de passe.

    Raises:
        HTTPException 401: identifiants invalides
        HTTPException 403: compte désactivé
    """
    result = await db.execute(select(User).where(User.email == email.lower().strip()))
    user = result.scalar_one_or_none()

    # `user.hashed_password` est NULL sur un compte cree par Google.
    # `verify_password` le traite deja, mais le rendre explicite ici evite que
    # quelqu'un « simplifie » un jour l'un sans voir l'autre. Meme 401 que pour
    # un mot de passe faux : ne jamais reveler qu'il s'agit d'un compte Google.
    if user is None or not user.hashed_password or not verify_password(
        password, user.hashed_password
    ):
        logger.warning(f"🔒 Échec de connexion pour {email!r}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_INVALID,
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Compte désactivé"
        )

    # Les colonnes DateTime du modele sont SANS fuseau : y ecrire un datetime
    # avec fuseau leve "can't subtract offset-naive and offset-aware datetimes"
    # au moment du flush. On stocke donc de l'UTC naif, comme le reste du schema.
    user.last_login_at = datetime.now(timezone.utc).replace(tzinfo=None)
    await db.commit()
    await db.refresh(user)

    logger.info(f"✅ Connexion de {user.email} (rôle {user.role})")
    return user


@router.post("/login", response_model=UserWithToken, status_code=status.HTTP_200_OK)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
) -> UserWithToken:
    """
    Connexion au format formulaire OAuth2 (utilisée par /docs).

    Le champ `username` porte l'adresse email.
    """
    user = await _authenticate(db, form_data.username, form_data.password)
    return UserWithToken(
        **UserResponse.model_validate(user).model_dump(),
        access_token=create_access_token(
            {"sub": user.email}, expires_delta=duree_de_session(user)
        ),
    )


@router.post("/login/json", response_model=UserWithToken, status_code=status.HTTP_200_OK)
async def login_json(
    credentials: UserLogin,
    db: AsyncSession = Depends(get_db),
) -> UserWithToken:
    """Connexion au format JSON (utilisée par le front)."""
    user = await _authenticate(db, credentials.email, credentials.password)
    return UserWithToken(
        **UserResponse.model_validate(user).model_dump(),
        access_token=create_access_token(
            {"sub": user.email}, expires_delta=duree_de_session(user)
        ),
    )


def _reponse_avec_jeton(user: User) -> UserWithToken:
    """Sérialise un compte et lui joint un jeton de la durée qui lui revient."""
    return UserWithToken(
        **UserResponse.model_validate(user).model_dump(),
        access_token=create_access_token(
            {"sub": user.email}, expires_delta=duree_de_session(user)
        ),
    )


def _maintenant_naif() -> datetime:
    """
    UTC sans fuseau.

    Les colonnes DateTime du modèle sont SANS fuseau : y écrire un datetime
    qui en porte un lève « can't subtract offset-naive and offset-aware
    datetimes » au moment du flush.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


@router.post("/signup", response_model=UserWithToken, status_code=status.HTTP_201_CREATED)
async def signup(
    payload: SignupRequest,
    request: Request,
    taches: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> UserWithToken:
    """
    Inscription publique : nom, adresse, mot de passe.

    Le rôle est écrit `user` PAR LE SERVEUR. `SignupRequest` n'expose ni `role`
    ni `is_active` et refuse les clés inconnues, donc `{"role": "superadmin"}`
    produit un 422 — pas un silence.

    L'adresse est normalisée À L'ÉCRITURE. `_authenticate` cherche avec
    `email.lower().strip()` : sans cette normalisation, un compte créé avec une
    majuscule ne pourrait jamais se connecter. C'est le défaut que portait
    `admin.create_user`, corrigé au passage.

    Raises:
        409: un compte existe déjà avec cette adresse
        422: données invalides (mot de passe faible, champ interdit)
    """
    email = normaliser_email(payload.email)

    existant = (
        await db.execute(select(User).where(User.email == email))
    ).scalar_one_or_none()
    if existant is not None:
        # Message neutre : ne jamais révéler qu'il s'agit d'un compte Google.
        # (Cette route énumère nécessairement les comptes — c'est le cas de
        # tout formulaire d'inscription sans confirmation par e-mail.)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Un compte existe déjà avec cette adresse",
        )

    # `username_pour_email` consulte la base, mais deux inscriptions simultanées
    # peuvent franchir ce contrôle ensemble : l'index unique tranche, et on
    # rejoue. Boucle bornée — un échec répété n'est plus une collision.
    for tentative in range(3):
        user = User(
            email=email,
            username=await username_pour_email(db, email),
            hashed_password=hash_password(payload.password),
            full_name=payload.full_name,
            role="user",
            is_active=True,
            # L'adresse n'est pas encore prouvée. Elle le devient par le
            # lien de confirmation envoyé juste après, par une affirmation
            # `email_verified` de Google, ou par une réinitialisation de mot de
            # passe réussie — trois preuves du contrôle de la boîte.
            is_verified=False,
            last_login_at=_maintenant_naif(),
        )
        db.add(user)
        try:
            await db.commit()
            break
        except IntegrityError:
            await db.rollback()
            if tentative == 2:
                logger.error("Inscription impossible pour %s après 3 tentatives", email)
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Un compte existe déjà avec cette adresse",
                )

    await db.refresh(user)

    # Le lien de confirmation part EN TÂCHE DE FOND, et son échec est avalé.
    # Une panne de l'expéditeur ne doit pas faire échouer une inscription : le
    # compte est créé, la session est ouverte, et l'utilisateur peut demander un
    # renvoi. Sans clé configurée, aucun jeton n'est même créé et cette route se
    # comporte exactement comme avant.
    await _programmer_verification(taches, db, user, adresse_du_client(request))

    logger.info("🆕 Inscription : %s", user.email)
    return _reponse_avec_jeton(user)


@router.post("/google", response_model=UserWithToken, status_code=status.HTTP_200_OK)
async def connexion_google(
    payload: GoogleAuthRequest,
    db: AsyncSession = Depends(get_db),
) -> UserWithToken:
    """
    Connexion par Google Identity Services.

    Résolution en trois temps : par `google_sub` (l'identifiant stable), puis
    par adresse (→ **liaison** d'un compte mot de passe existant), puis
    création.

    LA LIAISON EST ASYMÉTRIQUE, ET C'EST DÉLIBÉRÉ. Google, en affirmant
    `email_verified`, prouve le contrôle de la boîte aux lettres — une preuve
    plus forte que tout ce que notre propre inscription établit, puisque nous
    n'envoyons aucun e-mail. L'inverse serait une prise de contrôle de compte :
    `POST /signup` refuse donc en 409 une adresse déjà rattachée à un compte
    Google, sans jamais fusionner.

    Le mot de passe existant est CONSERVÉ : le compte gagne une porte, il n'en
    perd pas.

    L'adresse n'est jamais mise à jour depuis Google : le JWT maison porte
    `{"sub": <email>}`, la changer invaliderait tous les jetons vivants.

    Raises:
        401: jeton invalide, expiré, d'une autre application, ou adresse non vérifiée
        403: compte désactivé
        503: connexion Google non configurée, ou certificats Google injoignables
    """
    try:
        claims = await verifier_jeton_google(payload.credential)
    except GoogleNonConfigure as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)
        )
    except GoogleInjoignable as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
            headers={"Retry-After": "30"},
        )
    except JetonGoogleInvalide:
        # Message unique, comme `_INVALID` : détailler la cause aiderait
        # surtout qui cherche à fabriquer un jeton acceptable.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Connexion Google refusée",
            headers={"WWW-Authenticate": "Bearer"},
        )

    google_sub = claims["sub"]
    email = normaliser_email(claims["email"])

    user = (
        await db.execute(
            select(User).where(
                or_(User.google_sub == google_sub, User.email == email)
            )
        )
    ).scalars().first()

    if user is None:
        user = User(
            email=email,
            username=await username_pour_email(db, email),
            hashed_password=None,  # compte sans mot de passe : la colonne est nullable
            full_name=claims.get("name") or email.split("@")[0],
            google_sub=google_sub,
            role="user",
            is_active=True,
            is_verified=True,  # Google a affirmé email_verified
            last_login_at=_maintenant_naif(),
        )
        db.add(user)
        logger.info("🆕 Compte Google créé : %s", email)
    else:
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN, detail="Compte désactivé"
            )
        if user.google_sub is None:
            user.google_sub = google_sub
            user.is_verified = True
            logger.info("🔗 Compte lié à Google : %s", user.email)
        user.last_login_at = _maintenant_naif()

    await db.commit()
    await db.refresh(user)
    return _reponse_avec_jeton(user)


@router.get("/me", response_model=UserResponse, status_code=status.HTTP_200_OK)
async def read_me(current_user: User = Depends(get_current_active_user)) -> User:
    """Renvoie l'utilisateur associé au jeton fourni."""
    return current_user


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(current_user: User = Depends(get_current_active_user)) -> None:
    """
    Déconnexion.

    Les JWT sont sans état : rien n'est révoqué côté serveur, le client jette son
    jeton. L'endpoint existe pour donner un point d'appel explicite au front et
    pour tracer la déconnexion.
    """
    logger.info(f"👋 Déconnexion de {current_user.email}")
    return None


# ==================== Courriel : vérification et réinitialisation ====================

# Réponse unique des deux routes de demande. Toutes les branches la rendent —
# adresse inconnue, compte désactivé, étranglement atteint, envoi réel — parce
# qu'une réponse qui varierait ferait de ces routes un oracle permettant de
# savoir quelles adresses possèdent un compte.
_ACCUSE_RECEPTION = (
    "Si un compte existe pour cette adresse, un lien vient d'être envoyé. "
    "Pensez à vérifier vos courriers indésirables."
)

# Message unique pour toutes les causes de refus d'un jeton : illisible, expiré,
# déjà utilisé, ou destiné à un autre usage. Même raisonnement que `_INVALID` :
# détailler la cause n'aide que celui qui cherche à en fabriquer un.
_JETON_REFUSE = "Ce lien n'est plus valide. Demandez-en un nouveau."


async def _envoyer_sans_faire_echouer(destinataire: str, sujet: str, texte: str, html: str) -> None:
    """
    Envoie un message et avale toute panne.

    Appelée en tâche de fond. Une panne Brevo ne doit JAMAIS remonter jusqu'à la
    réponse HTTP : une inscription doit réussir même si l'expéditeur est en
    panne, et une demande de réinitialisation doit rendre le même accusé de
    réception que d'habitude, sinon l'échec devient lui-même un signal.
    """
    try:
        await email_service.envoyer(destinataire, sujet, texte, html)
    except email_service.CourrielNonConfigure as exc:
        # État normal en développement et en test. Pas une erreur.
        logger.debug("Envoi désactivé : %s", exc)
    except (email_service.CourrielRefuse, email_service.CourrielInjoignable) as exc:
        logger.warning("Courriel non délivré à %s : %s", destinataire, exc)


async def _programmer_verification(
    taches: BackgroundTasks,
    db: AsyncSession,
    user: User,
    ip: str | None = None,
    langue: str = "fr",
) -> None:
    """
    Crée un jeton de vérification et programme son envoi.

    Ne crée AUCUN jeton si l'envoi n'est pas configuré : une ligne en base pour
    un message qui ne partira jamais est un déchet, et elle fausserait le compte
    de l'étranglement.

    L'ÉTRANGLEMENT PAR IP EST ICI, ET C'EST LE SEUL ENDROIT OÙ IL PEUT ÊTRE.
    L'étranglement par compte ne protège rien sur ce chemin : chaque inscription
    crée un compte NEUF, donc son compteur repart de zéro. Sans cette garde,
    inscrire cent adresses différentes enverrait cent messages et épuiserait le
    quota quotidien — qui est un plafond dur et partagé. La garde ne bloque pas
    l'inscription elle-même, seulement le message : le compte est créé, la
    session est ouverte, et un renvoi reste possible plus tard.
    """
    if not email_service.lien_configure():
        return
    if ip is not None and ip_etranglee(ip):
        logger.info("Envoi de vérification étranglé pour %s", ip)
        return
    jeton = await creer_jeton(db, user, VERIFICATION_ADRESSE, DUREE_VERIFICATION, ip)
    url = email_service.url_du_front(f"/verify-email/{jeton}")
    sujet, texte, html = email_templates.verification_adresse(
        user.full_name or user.username, url, langue
    )
    taches.add_task(_envoyer_sans_faire_echouer, user.email, sujet, texte, html)


@router.post(
    "/password/forgot",
    response_model=MessageReponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def demander_reinitialisation(
    payload: DemandeReinitialisation,
    request: Request,
    taches: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
) -> MessageReponse:
    """
    Demande un lien de réinitialisation de mot de passe.

    RÉPOND TOUJOURS 202, AVEC LE MÊME CORPS. Adresse inconnue, compte désactivé,
    étranglement atteint, compte Google, envoi réel : cinq chemins, une seule
    réponse. Un code ou un message différent transformerait cette route en
    oracle d'énumération de comptes — et elle est bien plus facile à balayer
    qu'un formulaire d'inscription.

    LE CAS DU COMPTE GOOGLE. Un compte sans mot de passe n'a rien à
    réinitialiser. Lui en greffer un lui ajouterait une porte d'entrée dont son
    propriétaire n'a jamais voulu — donc une surface d'hameçonnage et de
    bourrage d'identifiants supplémentaire. Il reçoit à la place un message qui
    dit la vérité : reconnectez-vous avec Google. Ce message part quand même,
    sans quoi l'absence de courriel distinguerait un compte Google d'une adresse
    inconnue.
    """
    ip = adresse_du_client(request)
    email = normaliser_email(payload.email)

    # Étranglement par IP AVANT toute lecture : il attrape le balayage
    # d'adresses inconnues, qui ne crée aucune ligne et échappe donc entièrement
    # à l'étranglement par compte.
    if ip_etranglee(ip):
        logger.warning("Demandes de réinitialisation étranglées pour %s", ip)
        return MessageReponse(message=_ACCUSE_RECEPTION)

    user = (
        await db.execute(select(User).where(User.email == email))
    ).scalar_one_or_none()

    if user is None or not user.is_active:
        return MessageReponse(message=_ACCUSE_RECEPTION)

    if not email_service.lien_configure():
        logger.debug("Envoi non configuré : aucune réinitialisation possible")
        return MessageReponse(message=_ACCUSE_RECEPTION)

    if user.hashed_password is None:
        sujet, texte, html = email_templates.compte_google(
            user.full_name or user.username, email_service.url_du_front("/login")
        )
        taches.add_task(_envoyer_sans_faire_echouer, user.email, sujet, texte, html)
        return MessageReponse(message=_ACCUSE_RECEPTION)

    if await compte_etrangle(db, user.id, REINITIALISATION_MOT_DE_PASSE):
        logger.info("Réinitialisation étranglée pour le compte %s", user.id)
        return MessageReponse(message=_ACCUSE_RECEPTION)

    jeton = await creer_jeton(db, user, REINITIALISATION_MOT_DE_PASSE, DUREE_REINITIALISATION, ip)
    url = email_service.url_du_front(f"/reset-password/{jeton}")
    sujet, texte, html = email_templates.reinitialisation(user.full_name or user.username, url)
    taches.add_task(_envoyer_sans_faire_echouer, user.email, sujet, texte, html)

    return MessageReponse(message=_ACCUSE_RECEPTION)


@router.post(
    "/password/reset",
    response_model=MessageReponse,
    status_code=status.HTTP_200_OK,
)
async def reinitialiser_mot_de_passe(
    payload: ReinitialisationMotDePasse,
    db: AsyncSession = Depends(get_db),
) -> MessageReponse:
    """
    Pose un nouveau mot de passe à partir d'un jeton reçu par courriel.

    Tout se passe dans UNE transaction : le mot de passe est écrit, le jeton est
    consommé, les autres jetons de réinitialisation du compte sont invalidés, et
    `is_verified` passe à vrai — recevoir ce courriel prouve le contrôle de la
    boîte, exactement comme l'affirmation `email_verified` de Google.

    LIMITE À CONNAÎTRE, ET ELLE N'EST PAS CACHÉE : les JWT sont sans état,
    `/logout` ne révoque rien, il n'existe aucune liste de révocation, et la
    session dure trente jours. **Changer son mot de passe ne déconnecte donc
    personne** : un jeton déjà volé survit à la réinitialisation. Le seul levier
    immédiat est de désactiver le compte, que `get_current_user` relit à chaque
    requête. Le correctif propre est un numéro de version dans le claim, qui
    dépasse ce lot — c'est une dette, pas un oubli.

    Raises:
        400: jeton illisible, expiré, déjà utilisé, ou d'un autre usage
    """
    ligne = await consommer_jeton(db, payload.token, REINITIALISATION_MOT_DE_PASSE)
    if ligne is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=_JETON_REFUSE)

    user = (
        await db.execute(select(User).where(User.id == ligne.user_id))
    ).scalar_one_or_none()
    if user is None or not user.is_active:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=_JETON_REFUSE)

    user.hashed_password = hash_password(payload.password)
    user.is_verified = True
    await invalider_les_autres(db, user.id, REINITIALISATION_MOT_DE_PASSE)
    await db.commit()

    logger.info("🔑 Mot de passe réinitialisé pour %s", user.email)
    return MessageReponse(message="Mot de passe modifié. Vous pouvez vous connecter.")


@router.post(
    "/verify-email",
    response_model=MessageReponse,
    status_code=status.HTTP_200_OK,
)
async def verifier_adresse(
    payload: VerificationAdresse,
    db: AsyncSession = Depends(get_db),
) -> MessageReponse:
    """
    Confirme une adresse à partir du jeton reçu par courriel.

    POST ET NON GET, ET CE N'EST PAS UN DÉTAIL DE STYLE. Un
    `GET /verify-email/<jeton>` serait déclenché par les analyseurs de liens —
    Outlook Safe Links, passerelles de messagerie d'entreprise, aperçus — AVANT
    que l'humain ne clique. Le jeton étant à usage unique, l'utilisateur
    arriverait alors systématiquement sur « lien déjà utilisé ». Le lien du
    courriel pointe donc vers une page du front, qui poste le jeton depuis le
    navigateur.

    Raises:
        400: jeton illisible, expiré, déjà utilisé, ou d'un autre usage
    """
    ligne = await consommer_jeton(db, payload.token, VERIFICATION_ADRESSE)
    if ligne is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=_JETON_REFUSE)

    user = (
        await db.execute(select(User).where(User.id == ligne.user_id))
    ).scalar_one_or_none()
    if user is None:
        await db.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=_JETON_REFUSE)

    user.is_verified = True
    await db.commit()

    logger.info("✅ Adresse vérifiée : %s", user.email)
    return MessageReponse(message="Adresse confirmée.")


@router.post(
    "/verify-email/resend",
    response_model=MessageReponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def renvoyer_verification(
    request: Request,
    taches: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> MessageReponse:
    """
    Renvoie le lien de vérification au compte connecté.

    Authentifiée, donc sans risque d'énumération — mais étranglée quand même :
    le quota d'envoi est un plafond dur et partagé, qu'un seul compte ne doit
    pas pouvoir épuiser.
    """
    if current_user.is_verified:
        return MessageReponse(message="Adresse déjà confirmée.")

    # L'étranglement par IP est appliqué par `_programmer_verification`, une
    # seule fois : le vérifier aussi ici consommerait deux jetons pour une même
    # requête et diviserait le plafond par deux sans que rien ne le dise.
    if await compte_etrangle(db, current_user.id, VERIFICATION_ADRESSE):
        return MessageReponse(message=_ACCUSE_RECEPTION)

    await _programmer_verification(
        taches, db, current_user, adresse_du_client(request)
    )
    return MessageReponse(message=_ACCUSE_RECEPTION)
