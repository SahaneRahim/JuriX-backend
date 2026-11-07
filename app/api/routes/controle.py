"""
Controle de production : GET /api/v1/controle/chat.

Le chat fait-il encore lire au modele les articles qui repondent ? Les
questions de reference (toutes les suggestions du front, des questions de
fond, et des questions qui nomment un article) passent par sa recuperation,
sans generation : voir app/services/controle_du_chat.py pour ce qui est
verifie, et pourquoi (incident du 29/10/2026). Seules les questions
GARANTIES font rendre 503 ; les faiblesses connues de la recuperation sont
rapportees a part, sans 503 (docstring de la route, plus bas).
"""

from functools import partial

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from app.core.database import AsyncSessionLocal
from app.services.controle_du_chat import (
    ControleEnCache,
    compter_les_vecteurs,
    controler_le_chat,
    sonder_le_fournisseur,
)

router = APIRouter()

_controle = ControleEnCache(
    partial(
        controler_le_chat,
        AsyncSessionLocal,
        sonde=sonder_le_fournisseur,
        comptes=compter_les_vecteurs,
    )
)


def controle_en_cache() -> ControleEnCache:
    """Le controle du processus ; un test le remplace (dependency_overrides)."""
    return _controle


@router.get("/chat")
async def controle_du_chat(controle: ControleEnCache = Depends(controle_en_cache)) -> JSONResponse:
    """
    LES CODES :
    - 200 : chaque question de reference GARANTIE fait lire au modele un
      article qui lui repond ET QUI PORTE LA REGLE DEMANDEE (la peine du vol
      est a l'art. 318, pas a l'art. 320 qui la double), parmi les 4 premiers
      extraits, sans autre texte avant lui (hors extraits toleres par la
      question). L'article qu'une question nomme est lu en premier, par la
      lecture directe. Le fournisseur d'embeddings repond, la recherche
      exacte du chat tient, et sa table des vecteurs compte autant de lignes
      que d'articles vectorises. Regle et mesures : controle_du_chat,
      RANG_MAX_DE_LA_REPONSE. Les FAIBLESSES CONNUES (plus bas) n'y changent
      rien, qu'elles echouent ou qu'elles tiennent ;
    - 503 : une panne NOUVELLE, dite dans `raisons` : une question garantie
      en echec (detail question par question dans `echecs` : ce qui manque,
      ce que le modele aurait lu, les autres textes lus avant la reponse), la
      sonde du fournisseur, le repli de la recherche exacte sur l'index, la
      derive de la table des vecteurs (`table_des_vecteurs`), ou un calcul en
      erreur ou abandonne ;
    - 202 : aucun rapport encore (premier appel apres un redemarrage, calcul
      de plus de 25 s) ; le calcul se poursuit et sert l'appel suivant. Pour
      cron-job.org, c'est un succes : un deploiement ne fait pas sonner
      l'alerte.

    QUESTIONS GARANTIES ET FAIBLESSES CONNUES (revue du tour 5, 30/10/2026).
    Une question est garantie si elle tient aujourd'hui, ses attendus relus :
    son echec fait rendre 503. Une FAIBLESSE CONNUE echoue aujourd'hui, a
    raison : la recuperation fait lire au modele un autre regime avant la
    reponse, trop tard, ou jamais. Elle est dans le rapport (`faiblesses` :
    rang de la reponse, ce que le modele lit avant, raisons, diagnostic
    connu), mais ne fait JAMAIS rendre 503 ; si elle tient de nouveau, le
    rapport dit « faiblesse résolue » (`faiblesses_resolues`), sans 503 non
    plus, et elle est a retirer de la liste pour que sa prochaine panne fasse
    503. POURQUOI : comptees avec les autres, elles faisaient rendre 503 a
    chaque calcul, et l'index perdu (l'incident du 29/10/2026) comme Voyage
    en panne rendaient le meme code ; cron-job.org, qui ne lit que lui,
    n'aurait plus rien signale de neuf. Chaque panne rejouee par la revue
    (index perdu, Voyage en panne, poids de fusion, RRF_K, lecture directe de
    l'article nomme coupee, art. 318 ecarte, morceau qui repond perdu, repli
    sur l'index) fait echouer une question garantie ou le controle entier :
    503. LA LISTE (questions_de_reference.FAIBLESSES_CONNUES, ou est le
    diagnostic de chacune ; elle ne s'allonge jamais pour faire passer un
    envoi) :
    - « Quel est le montant de l'amende prévue pour le vol ? » : l'art. 318
      jamais lu ;
    - « Quel est le délai de préavis en cas de licenciement ? » : l'art. 34 au
      rang 6, apres l'AUPCAP et le preavis d'un agent commercial ;
    - « How much compensation is allocated for unfair dismissal? » : le
      l'art. 39 au rang 4, apres l'indemnite du fonctionnaire.
    Le vol (fr, en) et l'acces non autorise (en) tiennent depuis le 07/11/2026
    (redecoupage du corpus, lecture de l'article renvoye).

    SANS DANGER EN ACCES PUBLIC : un calcul au plus par 30 minutes, quels que
    soient les appels, qui lisent sinon le dernier rapport (`age_s`) ; calcul
    borne a 2 minutes ; reponse en 25 s au plus (au-dela, le rapport
    precedent, marque `perime`) ; aucune ecriture, aucun appel a Mistral ni a
    Groq ; ni secret ni message d'erreur dans le rapport (une erreur n'y est
    nommee que par sa classe), seulement des questions, des references de
    textes et des mesures.

    ETAT AU 07/11/2026 (copie reparee) : 200, les 48 questions garanties
    tiennent, et les 3 faiblesses connues echouent (`faiblesses_en_echec` : 3). Un 503 dit donc
    une panne nouvelle. Lire `faiblesses` chaque semaine : c'est a la
    recuperation de les corriger, et `faiblesses_resolues` dit celles qui
    tiennent, a retirer de la liste. UNE GARANTIE FRAGILE : la question du
    taux de la TVA tient avec le vecteur enregistre et avec le vrai Voyage,
    mais un vecteur a peine different lui fait lire d'abord l'art. 142 d'une
    ancienne loi de finances (7 tirages sur 40 au cosinus 0,9994, 10 sur 40 a
    0,9985 ; quand les 51 vecteurs bougent ensemble, 2 calculs sur 10 rouges
    par elle seule ; data/fiabilite/tmp/controle6/ et controle7/). Un 503 pour
    elle seule, raison « un autre texte est lu avant l'extrait qui repond »,
    est ce risque-la, que l'utilisateur vit aussi ; pas l'index ni Voyage.

    BRANCHER CRON-JOB.ORG (gratuit, sans carte), APRES avoir publie au front
    les suggestions anglaises reformulees le 29/10/2026 (voir
    app/services/questions_de_reference.py) : avant, le controle juge des
    questions que le front ne propose pas encore.
    1. « Create cronjob » : URL https://<api>/api/v1/controle/chat, methode
       GET.
    2. Execution schedule, « Custom » : minutes 0 ET 5 ; heures 7 a 18 ; tous
       les jours ; fuseau Africa/Douala. Ce sont les heures de la sonde de
       reveil deja en place (GET /health/db toutes les 4 minutes, 7 h a 18 h) :
       l'API et Neon sont eveilles, le controle n'ajoute pas d'heures de
       calcul (CU-h) au forfait gratuit de Neon, deja a ~93 CU-h sur 100 par
       mois avec cette sonde (calcul du 28/10/2026). Hors de ces heures, chaque
       calcul reveillerait Neon pour 6 minutes environ (le calcul, puis 5 min
       avant la mise en veille), a 0,25 CU : ~9 CU-h de plus par mois sur les
       12 heures restantes, au-dela du forfait. L'appel de la minute 0 lance
       le calcul (plus de 25 s sur Neon) et rend le rapport de l'heure
       d'avant ; celui de la minute 5 lit le nouveau, sans second calcul.
    3. Notifications : cocher « execution of the cronjob fails » (le courriel
       d'alerte), « succeeds after it failed before » et « will be disabled
       because of too many failures ». Tout code hors 2xx compte comme un
       echec, et une reponse au-dela de 30 s aussi.
    4. Save, puis « Test run » : 200, ou 202 au premier appel. Un 503 se lit
       dans `raisons`, puis dans `echecs` ; les faiblesses connues, jamais en
       503, dans `faiblesses`.

    COUT D'UN CALCUL : les recherches du chat pour chaque question (51), sans
    generation ; deux comptes, ceux de la table des vecteurs ; un appel a
    Voyage pour la sonde (deux s'il echoue), plus un
    par question au premier calcul apres un demarrage (leurs vecteurs restent
    ensuite en memoire) ; 4 connexions sur les 10 du pool, deux minutes au
    plus : les utilisateurs gardent les autres, meme en demonstration.
    """
    code, rapport = await controle.lire()
    return JSONResponse(status_code=code, content=rapport, headers={"Cache-Control": "no-store"})
