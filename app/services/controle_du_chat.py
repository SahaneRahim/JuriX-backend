"""
Controle du chat : les questions de reference passent par la recuperation du
chat, SANS generation, et le controle dit si le modele lirait bien les
articles qui leur repondent.

POURQUOI. Incident du 29/10/2026 : « Quelles indemnités en cas de licenciement
abusif ? », une question que le chat propose, a recu une reponse batie sur
des tableaux de lois de finances. Mesure sur la base de production : son index
HNSW rendait 0 voisin juste sur 10 pour cette question (hnsw.ef_search 160),
la ou l'index local en rendait 10. Un utilisateur l'a vu avant nous. Ce
controle le voit avant lui : GET /api/v1/controle/chat
(app/api/routes/controle.py), appele chaque heure par cron-job.org, et le
test tests/test_integration/test_questions_de_reference.py, avant chaque
envoi du code.

CE QUI EST VERIFIE, pour chaque question de reference
(app/services/questions_de_reference.py), sur les extraits que le modele
lirait (ceux que le prompt retient, budget du contexte compris) :
1. LES TEXTES ATTENDUS sont lus. Une liste vide (barriere du chat, rien de
   trouve) est un echec : le chat repondrait qu'il n'a rien trouve a une
   question qu'il propose lui-meme ;
2. UN ARTICLE QUI REPOND est lu (porte_l_article), et il porte la REGLE
   demandee (porte_la_regle : la peine, la duree, la definition...) : le bon
   code peut arriver par un article hors sujet, ou par un article qui ne fait
   que renvoyer a la regle (« Les peines de l'article 318 ci-dessus sont
   doublées »). Un article qui repond par renvoi (« Est puni des peines
   prévues à l'alinéa 1 ») ne repond qu'une fois l'article vise lu aussi ;
3. CE QUE LE MODELE LIT AVANT LUI (juger, regle detaillee a
   RANG_MAX_DE_LA_REPONSE) : aucun extrait d'un AUTRE texte, sauf ceux que
   la question tolere un par un, et l'article qui repond parmi les
   RANG_MAX_DE_LA_REPONSE premiers extraits. Une question qui NOMME un
   article le fait lire EN PREMIER, par la lecture directe de l'article
   nomme. Derriere l'indemnite d'un fonctionnaire, le modele repondrait sur
   le mauvais regime ;
4. LE VOLET SEMANTIQUE du chat, quel que soit le chemin qui le sert
   (recherche exacte, index, repli) : ses VOISINS_SEMANTIQUES premiers voisins
   comptent un texte attendu. Le plein texte peut sauver le contexte d'une
   question dont les vecteurs se perdent ; ce point le voit quand meme ;
5. pour les questions marquees `rappel`, le RAPPEL@10 DE L'INDEX HNSW a
   chaque valeur de EF_SEARCH_DU_RAPPEL, contre une recherche exacte, sans
   index (meme vecteur, articles publies). L'incident : 0 sur 10.
Et pour le controle entier :
6. le FOURNISSEUR D'EMBEDDINGS repond, hors de la memoire des vecteurs de
   questions (sonder_le_fournisseur), qui masquerait sinon une panne de
   Voyage ;
7. la recherche exacte du chat n'est pas retombee sur l'index pour la
   PLUPART des questions (PART_MAX_EN_REPLI) : c'est elle qui protege le chat
   d'un index qui se perd, et une base trop lente la coupe sans bruit.

LE CODE DE LA ROUTE : QUESTIONS GARANTIES ET FAIBLESSES CONNUES (revue du
tour 5, 30/10/2026). Chaque question de reference est GARANTIE, ou FAIBLESSE
CONNUE : l'une des questions que la recuperation fait echouer aujourd'hui, a
raison, ses attendus etant justes (questions_de_reference.FAIBLESSES_CONNUES,
avec leur diagnostic ; 6 le 30/10/2026). Le rapport est « echec », et la
route rend 503, si :
- une question GARANTIE echoue, quelle qu'en soit la raison ;
- la sonde du fournisseur echoue (6), ou la recherche exacte est en repli
  pour la plupart des questions (7) ;
- aucune question garantie n'est verifiee, ou le calcul est en erreur ou
  abandonne a son delai (ControleEnCache).
Sinon « ok » et 200, MEME SI des faiblesses connues echouent. Le rapport les
donne a part (`faiblesses` : rang de la reponse, ce que le modele lit avant,
raisons, diagnostic connu). Une faiblesse qui tient de nouveau y est dite
« faiblesse résolue » (`faiblesses_resolues`, et un avertissement au
journal), sans 503 non plus : elle est a retirer de la liste, pour que sa
prochaine panne fasse 503. Une faiblesse ne fait jamais rendre 503, quelle
que soit la raison de son echec : une panne qui la toucherait (volet
semantique, delai, index) touche aussi des questions garanties, ou la sonde.
POURQUOI. Comptees avec les autres, les 6 faiblesses faisaient rendre 503 a
chaque calcul (revue du tour 5) : l'index perdu, l'incident du 29/10/2026,
et Voyage en panne rendaient le meme 503 qu'un jour sans panne. Or
cron-job.org ne lit que le code HTTP : l'alerte ne pouvait plus signaler une
panne NOUVELLE, et il desactive une tache qui echoue trop souvent. Chaque
panne rejouee par la revue (index perdu, Voyage en panne, poids de fusion,
RRF_K, lecture directe de l'article nomme coupee, art. 318 ecarte, morceaux
35.1, 318 ou 33 perdus, repli sur l'index) fait echouer au moins une
question garantie, ou une raison du controle entier : elle rend 503
(tests/test_integration/test_questions_de_reference.py, et les mesures de
data/fiabilite/tmp/controle6/ et controle7/, la seconde sur la recuperation
finale du tour 6).

Un volet semantique en echec (fournisseur, requete vectorielle) est un
echec : la mesure ne dirait rien de l'etat nominal. Mais un seul appel lent
ne doit pas faire sonner l'alerte : la sonde et chaque question ont droit a
un second essai.

CE QUI N'EST PAS FAIT. Aucune generation, aucun appel a Mistral ni a Groq :
le classement d'intention, qui precede la recherche dans le chat, n'est donc
pas controle, ni le re-ranking par le modele (coupe par defaut,
RERANK_LLM_ENABLED), ni la lecture des citations. Aucune ecriture : ni
journal des recherches, ni cache de recherche (lu ou ecrit), ni
conversation. Seuls appels externes : la base, et Voyage, pour la sonde et
pour les questions dont le vecteur n'est pas deja en memoire.
"""

import asyncio
import logging
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import literal, literal_column, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.law import Article, Law
from app.schemas.rag import RAGRequest
from app.schemas.search import ChunkResult
from app.services import rag_service, search_service
from app.services.embedding_service import EmbeddingService
from app.services.questions_de_reference import QUESTIONS_DE_REFERENCE, QuestionDeReference
from app.services.rag_service import RAGService
from app.services.search_service import SearchService

logger = logging.getLogger(__name__)

# Questions verifiees en meme temps. Chacune tient une connexion pour le
# plein texte et, le temps de son volet semantique, une seconde : 2 en
# prennent 4 au plus des 10 du pool (app/core/database.py, 5 + 5), et en
# laissent 6 aux utilisateurs. Le controle passe aussi pendant une
# demonstration : il ne doit pas la ralentir.
#
# Le controle entier, mesure en local (jurix_dev, 29/10/2026, par la route,
# recherche semantique exacte du chat, vrai Voyage, sonde comprise, machine
# chargee par les suites d'autres agents) : 27,8 s pour les 49 questions au
# premier calcul (49 vecteurs demandes a Voyage), 734 ms de mediane par
# question, 3,1 s au plus : les mots-cles anglais font tourner les replis du
# plein texte sur un corpus francais. Neon est plus lent (recherche exacte ~250 ms a chaud contre
# ~110 ms ici, reveil +1,2 s, base froide ~4 s sur la premiere recherche,
# mesures des 22 et 29/10/2026) : le calcul de production depassera
# ATTENTE_MAX_S, d'ou le rapport garde (ControleEnCache) et les deux appels
# par heure (app/api/routes/controle.py).
CONCURRENCE = 2
# Borne du calcul entier. Au-dela, les questions encore en cours sont
# arretees et comptent comme des echecs : rien ne doit tourner sans fin.
# Plus de cinq fois le calcul local ; au-dela, le chat lui-meme serait trop
# lent pour ses utilisateurs, et l'alerte est juste.
DELAI_DU_CALCUL_S = 120.0
# CE QUE LE MODELE LIT AVANT L'ARTICLE QUI REPOND (juger). Le premier extrait
# qui repond : un article attendu qui porte la regle demandee (et, s'il
# renvoie a un autre pour sa peine, le rang ou l'article vise est lu aussi :
# la reponse n'est complete qu'alors), ou, pour une question sans articles
# declares, un texte attendu.
#
# POURQUOI LA REGLE, ET PAS SEULEMENT L'ARTICLE. Revue du tour 4 (29/10/2026) :
# « Quelle peine encourt l'auteur d'un vol ? » restait verte par l'art. 320
# du Code penal (« Les peines de l'article 318 ci-dessus sont doublées »), la
# ou le modele lisait le vol d'effets MILITAIRES et le vol de matieres
# nucleaires avant la seule peine du vol, a l'art. 318 ; et « Quel est le
# délai de préavis en cas de licenciement ? » par le morceau 35.1 (les
# obligations PENDANT le preavis), la ou il lisait d'abord le preavis d'un
# AGENT COMMERCIAL. L'article attendu est celui qui porte la regle, et
# QuestionDeReference.regles en dit les mots : un morceau du meme numero qui
# ne les porte pas (l'art. 134-1, range sous « 134 », qui renvoie aux peines
# de l'art. 134) ne repond pas.
#
# POURQUOI. Revue du tour 3 (29/10/2026) : la regle d'alors tolerait deux
# extraits d'autres textes avant l'article qui repond, a n'importe quel
# rang. Le plein texte prive du morceau 35.1 du Code du travail, le modele
# lisait pour « Quelles indemnités en cas de licenciement abusif ? » :
# 1. l'indemnite du FONCTIONNAIRE licencie (Statut general de la fonction
#    publique, art. 120 : 12, 3 ou 24 mois de traitement) ;
# 2. le Code civil, art. 107 ;
# 3. enfin le Code du travail.
# Le controle restait vert. Un autre regime lu en tete, avec ses montants,
# fait une reponse fausse pour un salarie du prive.
#
# LA REGLE, pour toute question :
# 1. AUCUN extrait d'un autre texte avant l'article qui repond. Seule
#    exception : un extrait que la question tolere
#    (QuestionDeReference.toleres_avant), lu et juge un par un. Soit il est
#    sans rapport avec la question (les mentions d'un marche public avant la
#    definition du contrat a duree determinee), soit c'est une version
#    anterieure que le prompt signale reecrite par un texte attendu. Un
#    texte que rien ne declare est un autre regime : echec ;
# 2. l'article qui repond est parmi les RANG_MAX_DE_LA_REPONSE premiers
#    extraits lus. Un meme texte porte plusieurs regimes : le Code de
#    procedure penale traite la garde a vue, la detention provisoire et la
#    garde du mineur. Le modele lit 8 extraits (9 au plus) : au-dela de la
#    premiere moitie, il a deja lu quatre articles qui ne repondent pas ;
# 3. une question qui NOMME un article (QuestionDeReference.article_nomme)
#    le lit EN PREMIER, et par la lecture directe
#    (RAGService._article_nomme, origine _SOURCE_ARTICLE_NOMME). Le meme
#    numero d'un autre texte lu en tete, c'est la forme de l'incident.
#
# MESURE du 29/10/2026 (jurix_dev, vecteurs enregistres, recuperation de
# cette branche a 22 h 15 ; data/fiabilite/tmp/controle4/), base et 10
# tirages de gigue par question au cosinus 0,9993. Rang de l'article qui
# repond : 1 a 4, rang 4 pour deux questions (droits en garde a vue, GIE).
# Pannes rejouees : le plein texte sans 35.1 et les poids de fusion 0,1 / 0,9
# font lire le Statut de la fonction publique d'abord ; la lecture directe
# coupee fait echouer les cinq questions qui nomment un article, l'art. 318
# ecarte par _sans_doute la sienne ; RRF_K a 500 fait echouer par le rang
# seul (la duree de la garde a vue derriere la detention provisoire et la
# garde du mineur). L'ancienne regle restait verte.
#
# MESURE du 30/10/2026, articles attendus relus (ceux qui portent la regle),
# 51 questions ; memes conditions, recuperation du tour 5 (rag_service de
# 01 h 13, de 04 h 18 pour la gigue, 77b552d7 de 05 h 10 pour la route
# finale : memes verdicts), gigue de 10 tirages au cosinus 0,9985 (l'ecart
# du vrai Voyage mesure par la revue du tour 4) ; data/fiabilite/tmp/controle5/
# et controle5_reprise/ ; remesure au tour 6 sur la recuperation finale
# (rag_service d079c404, search_service c914d9bb ; controle7/) : memes rangs,
# memes verdicts, contexte par contexte :
# - la recuperation fait echouer, a raison, 6 questions dont la regle arrive
#   derriere un autre regime, trop tard, ou pas du tout, dans tous les tirages
#   et avec le vrai Voyage : la peine du vol (art. 318 au rang 7 derriere le
#   Code de justice militaire ; jamais lu en anglais, ni pour l'amende), le
#   preavis en francais (art. 34 au rang 6 derriere l'AUPCAP et le preavis
#   d'un agent commercial), la peine de l'acces non autorise en anglais (65.1
#   au rang 5, apres l'art. 69 et son amende de 10 a 100 millions) et
#   l'indemnite du licenciement abusif demandee autrement (le Statut de la
#   fonction publique, sans 35.1). Ce sont les faiblesses connues, avec leur
#   diagnostic : questions_de_reference.FAIBLESSES_CONNUES. Elles restent au
#   rapport, mais ne font pas rendre 503 (voir la docstring du module) ;
# - pour les 45 autres, l'extrait qui repond reste au rang 1 a 4, sans autre
#   texte avant lui hors tolerances, dans tous les tirages ; seul « Quel est
#   le taux de la TVA ? » bascule (2 tirages sur 10 ; 4 sur 10 dans la gigue
#   de la revue du tour 4), un vrai risque connu : question GARANTIE, elle
#   rend alors la route rouge a elle seule (docstring de la route, « UNE
#   GARANTIE FRAGILE »). Quatre sont au rang 4 dans tous les tirages, a la
#   limite, derriere des articles du meme texte : droits en garde a vue, GIE,
#   acces non autorise et duree de la garde a vue en francais ;
# - pannes : aucune de celles du tour 4 n'est perdue ; la perte du morceau
#   318 fait desormais echouer l'escroquerie, celle du morceau 33 le preavis
#   en anglais (elles restaient vertes), et les poids 0,1 / 0,9 la duree de la
#   garde a vue en anglais (le Code de justice militaire avant l'art. 119).
RANG_MAX_DE_LA_REPONSE = 4
VOISINS_SEMANTIQUES = 10
K_RAPPEL = 10
# Le rappel de l'index est mesure a deux parcours. 400 : la valeur de /search
# (search_service.EF_SEARCH_MIN) ; en dessous du seuil, /search et le repli du
# chat liraient de faux voisins. 200 : une MARGE, pour alerter avant qu'une
# recherche ne se trompe. L'index de l'incident rendait 0 voisin juste sur 10
# a 160 et 200, et 10 sur 10 a partir de 300 : il serait rouge ici, et il
# devait etre reconstruit (il l'a ete : m = 24, ef_construction = 200).
# Mesure sur l'index local (m = 16, ef_construction = 64, moins bon que celui
# de production), 44 questions, rappel@10 minimum : 0,0 a 40 (« corruption »
# et « incitations », la forme meme de l'incident, d'ou leur place parmi les
# questions `rappel`), 0,8 a 160, 0,9 a 200, 1,0 a 400 et 1000.
EF_SEARCH_DU_RAPPEL = (200, 400)
# L'incident : 0 ; l'index reconstruit : 1. 0,8 laisse la place a deux ex
# aequo ou a un parcours legerement approche, pas a un index qui se perd.
SEUIL_RAPPEL = 0.8
# Au-dela de cette part des questions verifiees servies par le repli sur
# l'index (recherche exacte annulee, trop lente), le controle echoue : la
# protection du chat est coupee. Une question ou deux en repli au reveil de
# Neon ne disent rien ; la plupart, si.
PART_MAX_EN_REPLI = 0.5

# Essais de la sonde du fournisseur, et du volet semantique d'une question.
# Le 29/10/2026, Voyage a expire 4 fois sur ~330 appels, isolement : un
# second essai ecarte ce bruit, deux echecs de suite sont une vraie panne.
ESSAIS = 2
PAUSE_ENTRE_DEUX_ESSAIS_S = 1.0

# Le rapport est garde 30 minutes, et le calcul ne part pas plus souvent,
# quels que soient le nombre d'appels et leur issue : la route est publique.
DUREE_DE_VIE_S = 30 * 60
# Un appel attend le calcul en cours au plus ce temps. cron-job.org coupe
# une requete a 30 s (son delai par defaut) et la compte en echec.
ATTENTE_MAX_S = 25.0
# Un calcul plus long que l'attente rend le rapport PRECEDENT, marque
# `perime`, s'il a moins que cet age : un controle lent ne doit pas sonner
# l'alarme a lui seul. Sans rapport assez recent (premier appel apres un
# redemarrage, ou trois heures sans appel), « en_cours », en 202 : ni
# succes ni echec, le resultat sert l'appel suivant (voir ControleEnCache).
AGE_MAX_DU_PRECEDENT_S = 3 * 3600
# Les codes HTTP du controle.
CODE_OK = 200
CODE_EN_COURS = 202
CODE_ECHEC = 503
# L'etat d'une faiblesse connue dans le rapport (`faiblesses`) : elle echoue
# encore, ou elle tient de nouveau. Ni l'un ni l'autre ne fait rendre 503.
FAIBLESSE_CONNUE = "faiblesse connue"
FAIBLESSE_RESOLUE = "faiblesse résolue"

# Les chemins du volet semantique (search_service.CHEMIN_*). Lus sans les
# exiger : le controle doit tourner aussi sur une recherche qui ne nomme pas
# ses chemins (d99e23c).
_CHEMIN_ECHEC = getattr(search_service, "CHEMIN_ECHEC", "echec")
_CHEMIN_REPLI = getattr(search_service, "CHEMIN_REPLI_HNSW", "repli_hnsw")
# L'origine (ChunkResult.source) d'un article lu par la lecture directe de
# l'article nomme. Meme lecture. Un renommage cote chat ferait echouer les
# questions qui nomment un article, bruyamment : le controle ne se tait pas.
_SOURCE_ARTICLE_NOMME = getattr(rag_service, "SOURCE_ARTICLE_NOMME", "article_nomme")


# ==================== UN EXTRAIT PORTE-T-IL UN ARTICLE ? ====================


def _en_tete(numero: str) -> "re.Pattern[str]":
    """
    L'en-tete de l'article `numero` dans un morceau qui en porte plusieurs :
    « Article 37.- », « Article 122 — », « Art. 1er : ». Un renvoi
    (« l'article 37 ci-dessus ») n'en est pas un : il faut le mot Article (ou
    Art.) puis un signe d'en-tete apres le numero ; et « Article 37-1 » est un
    autre article. La forme de RAGService._motif_d_en_tete, en Python.
    """
    premier = "premier|PREMIER|1er|" if numero == "1" else ""
    return re.compile(
        r"(?:^|[^\w'’])(?:Article|ARTICLE|Art\.)\s+"
        rf"(?:{premier}{re.escape(numero)})(?:er)?\s*(?:\.\s*[-–—]|[-–—:](?!\s*\d)|\.\s*\()"
    )


def porte_l_article(extrait: ChunkResult, reference: str, numero: str) -> bool:
    """
    L'extrait est-il (ou contient-il) l'article `numero` du texte `reference` ?

    - Son numero : « 318 » ; ou « 35.1 » pour 35, le PREMIER morceau d'un
      article long que le decoupage range sous « N.1 », « N.2 »... ;
    - ou un en-tete « Article 37.- » dans son contenu : le corpus range les
      articles 36 a 39 du Code du travail sous le morceau « 35.1 », et
      l'article 122 du Code de procedure penale sous « 120.2 ».
    Lu ainsi, l'article survit a un redecoupage du corpus, qui changerait les
    numeros des morceaux.

    UN MORCEAU « N.2 » ET AU-DELA NE COMPTE PAS POUR N par son numero. Il
    continue un article, mais lequel, et par quels alineas ? « 35.2 » du Code
    du travail commence par l'alinea 6 de l'article 39 (son en-tete est dans
    « 35.1 »), puis porte l'article 40, et rien de l'article 35. Compte pour
    35, il gardait vertes les questions du preavis sans le morceau 35.1, la
    ou le modele lisait le preavis d'un AGENT COMMERCIAL (AUDCG art. 228)
    avant celui du Code du travail (revue du tour 3 ; mesure du 29/10/2026).
    Compte pour 39, il gardait verte la question anglaise du licenciement
    sans 35.1, avec le seul salaire de reference. Quand la reponse est dans
    une suite, la question attend ce morceau par son numero exact : l'alinea
    2 de l'article 65 de la loi sur la cybercriminalite (l'acces non
    autorise) est le morceau « 65.2 » (questions_de_reference).
    """
    if extrait.reference != reference:
        return False
    numero_du_morceau = (extrait.number or "").strip()
    if numero_du_morceau in (numero, f"{numero}.1"):
        return True
    return _en_tete(numero).search(extrait.content or "") is not None


def _normaliser(texte: str) -> str:
    """Espaces et retours a la ligne reduits, apostrophe droite, casse ignoree."""
    return " ".join(texte.replace("’", "'").split()).casefold()


def porte_la_regle(extrait: ChunkResult, mots: Optional[str]) -> bool:
    """
    L'extrait porte-t-il les mots de la regle demandee
    (QuestionDeReference.regles) ? Sans mots declares, oui.

    POURQUOI. Porter l'article ne suffit pas a repondre. Le corpus range sous
    un meme numero des articles distincts : « 134 » du Code penal est aussi
    l'art. 134-1 (qui renvoie aux peines de l'art. 134) et l'art. 134-2
    (l'exemption de poursuites) ; « 311 » de l'AUSCGIE est aussi l'art.
    311-1 (la liberation des parts), sans le capital minimum. Et un
    redecoupage peut laisser l'en-tete d'un article dans un morceau et sa
    regle dans le suivant. Les mots de la regle, relus dans le morceau qui
    repond, tranchent : le meme numero sans eux ne repond pas. Lus au mot
    pres, apres reduction des espaces : le test d'integration verifie que
    chaque regle est bien dans un morceau de la base.
    """
    if not mots:
        return True
    return _normaliser(mots) in _normaliser(extrait.content or "")


def _repond(
    item: QuestionDeReference, extrait: ChunkResult, articles: Sequence[Tuple[str, str]]
) -> bool:
    """L'extrait porte-t-il l'un de ces articles, avec la regle que la question demande ?"""
    return any(
        porte_l_article(extrait, reference, numero)
        and porte_la_regle(extrait, item.regle_de(reference, numero))
        for reference, numero in articles
    )


def _premier_rang(lus: Sequence[ChunkResult], lu: Callable[[ChunkResult], bool]) -> Optional[int]:
    return next((i for i, c in enumerate(lus, start=1) if lu(c)), None)


def _lus_sans_la_regle(lus: Sequence[ChunkResult], articles: Sequence[Tuple[str, str]]) -> str:
    """
    « (lu sans la regle demandee : ...) » quand un extrait lu porte l'un de
    ces articles sans les mots de sa regle (un autre article range sous le
    meme numero, une autre part de l'article), sinon rien : la panne n'est
    pas la meme que l'article absent, et le proprietaire doit le lire.
    Appelee quand aucun extrait lu ne porte ces articles AVEC leur regle.
    """
    sans_la_regle = [
        _extrait(c) for c in lus if any(porte_l_article(c, ref, n) for ref, n in articles)
    ]
    return f" (lu sans la regle demandee : {', '.join(sans_la_regle)})" if sans_la_regle else ""


def _articles_en_clair(articles: Sequence[Tuple[str, str]]) -> str:
    """((ref, n), ...) -> « LOI-1992-007 art. 36, 37 ou 39 ou PRC-10679 art. 6 »."""
    par_texte: Dict[str, List[str]] = {}
    for reference, numero in articles:
        par_texte.setdefault(reference, []).append(numero)
    return " ou ".join(f"{ref} art. {', '.join(nums)}" for ref, nums in par_texte.items())


# ==================== RECUPERATION DU CHAT, SANS TRACE ====================


@dataclass
class ContexteDuChat:
    """Le contexte qu'une question donnerait au modele, et d'ou il vient."""

    extraits: List[ChunkResult]
    # Ceux que le prompt retient : le modele ne lit que ceux-la.
    lus: List[ChunkResult]
    # Le volet semantique de la recherche du chat, (article, texte) dans son
    # ordre ; None s'il n'a pas tourne.
    voisins: Optional[List[Tuple[int, str]]]
    semantique_degradee: bool
    # Qui a servi le volet semantique (recherche exacte, index...), quand le
    # code de la recherche le dit.
    chemin: Optional[str] = None
    # La barriere du chat a ecarte tout le contexte (rien d'assez pertinent).
    refus: bool = False

    @property
    def volet_en_panne(self) -> bool:
        """Le volet semantique n'a rien rendu d'exploitable : fournisseur ou requete."""
        return self.semantique_degradee or self.voisins is None or self.chemin == _CHEMIN_ECHEC


async def _sans_ecriture(*args: Any, **kwargs: Any) -> None:
    return None


def _sans_trace(recherche: SearchService) -> List[List[Tuple[int, str]]]:
    """
    La recherche du chat telle qu'il l'a construite (volet exact ou index,
    seuils...), moins ses ecritures, plus un regard sur son volet semantique.

    - Ni cache : une reponse de query_cache ne dirait rien de la base, et le
      controle n'y laisse rien.
    - Ni journal des recherches : 51 recherches par demi-heure fausseraient
      ses statistiques.
    - Les voisins du volet semantique sont gardes, dans leur ordre, tels qu'il
      les rend : avant tout filtre de la fusion.

    Sur CETTE instance seulement : les recherches des utilisateurs, servies en
    meme temps, n'en sont pas touchees.
    """
    recherche.use_cache = False
    recherche._ecrire_apres_la_reponse = _sans_ecriture
    vus: List[List[Tuple[int, str]]] = []
    chercher = recherche._chunks_semantiques

    async def _chunks_semantiques(*args: Any, **kwargs: Any) -> List[ChunkResult]:
        extraits = await chercher(*args, **kwargs)
        vus.append([(c.article_id, c.reference or "") for c in extraits])
        return extraits

    recherche._chunks_semantiques = _chunks_semantiques
    return vus


async def contexte_du_chat(
    db: AsyncSession, question: str, langue: str
) -> Tuple[ContexteDuChat, SearchService]:
    """
    Le chemin de `ask` et `ask_stream` jusqu'au prompt, generation exclue :
    _retrieve_and_merge_context, puis la construction du prompt, qui fixe les
    extraits lus. Le repli multilingue (_fallback_search) n'est pas rejoue :
    il ne joue que sur un contexte vide, que le controle compte deja comme un
    echec.

    `rag.llm = None` : aucun appel au modele, quoi que dise la configuration
    (le re-ranking par le modele ne part que si un modele est branche).
    """
    rag = RAGService(db)
    rag.llm = None
    recherche = rag.search_service
    vus = _sans_trace(recherche)
    requete = RAGRequest(question=question, language=langue)

    extraits, _ = await rag._retrieve_and_merge_context(requete, None)
    rag._build_prompt(question, extraits, [], requete.persona)
    volets = getattr(recherche, "volets", None)
    contexte = ContexteDuChat(
        extraits=list(extraits),
        lus=list(rag._extraits_lus(extraits)),
        voisins=vus[-1] if vus else None,
        semantique_degradee=bool(recherche.semantique_degradee),
        chemin=getattr(volets, "chemin", None) or getattr(recherche, "chemin_semantique", None),
        refus=bool(getattr(getattr(rag, "_recuperation", None), "refus", False)),
    )
    return contexte, recherche


# ==================== RAPPEL DE L'INDEX ====================


def _enonce_des_voisins(vecteur: List[float], candidats: int, k: int):
    """
    Les `k` articles publies les plus proches de `vecteur`, parmi `candidats`.

    L'idiome du volet semantique (SearchService._build_semantic_statement),
    ecrit ici pour que la mesure ne bouge pas avec lui : `<=>` sur la colonne
    nue, seul rattachable a l'index ; tri strict des candidats a l'exterieur
    (`+ 0`), l'identifiant departageant les ex aequo. Sous enable_indexscan
    off, le meme enonce est la recherche exacte.
    """
    distance = Article.embedding.cosine_distance(
        literal(vecteur, HALFVEC(EmbeddingService.EMBEDDING_DIM))
    )
    candidats_ = (
        select(Article.id.label("article_id"), distance.label("distance"))
        .join(Law, Law.id == Article.law_id)
        .where(Article.embedding.isnot(None), Law.status == "published")
        .order_by(distance)
        .limit(candidats)
        .subquery("candidats")
    )
    return (
        select(candidats_.c.article_id)
        .order_by(candidats_.c.distance + literal_column("0"), candidats_.c.article_id)
        .limit(k)
    )


async def voisins_par_l_index(
    db: AsyncSession, vecteur: List[float], k: int = K_RAPPEL, ef_search: int = 400
) -> List[int]:
    """
    Les `k` premiers voisins par l'index HNSW, a hnsw.ef_search `ef_search`,
    reglages poses comme le fait la recherche (parcours iteratif sous filtre,
    si pgvector le connait). Pour la transaction en cours : l'appelant
    l'annule ensuite.
    """
    await db.execute(
        text(SearchService._reglages_hnsw_sql(True)), {"hnsw_ef_search": str(int(ef_search))}
    )
    lignes = (await db.execute(_enonce_des_voisins(vecteur, ef_search, k))).all()
    return [ligne.article_id for ligne in lignes]


async def voisins_exacts(db: AsyncSession, vecteur: List[float], k: int = K_RAPPEL) -> List[int]:
    """
    Les `k` vrais plus proches voisins : l'index ecarte pour la transaction,
    PostgreSQL compare chaque article a la question et trie. Environ 250 ms
    sur la base de production (700 ms a froid), 190 a 300 ms en local.
    """
    await db.execute(text("SELECT set_config('enable_indexscan', 'off', true)"))
    lignes = (await db.execute(_enonce_des_voisins(vecteur, k, k))).all()
    return [ligne.article_id for ligne in lignes]


def rappel_a_k(index: Sequence[int], exacts: Sequence[int], k: int = K_RAPPEL) -> float:
    """Part des `k` voisins exacts que l'index a rendus parmi ses `k` premiers."""
    attendus = list(exacts)[:k]
    if not attendus:
        return 0.0
    return len(set(attendus) & set(list(index)[:k])) / len(attendus)


async def rappel_de_l_index(
    db: AsyncSession, recherche: SearchService, question: str
) -> Dict[int, float]:
    """
    Rappel@K_RAPPEL de l'index pour le vecteur de `question`, a chaque valeur
    de EF_SEARCH_DU_RAPPEL, chaque mesure dans sa transaction. Le vecteur vient
    du service d'embeddings du chat, qui vient de le calculer et le garde en
    memoire : aucun appel de plus.
    """
    service = recherche.embedding_service
    vecteur = await service.generate_embedding_async(question, task_type=service.TASK_QUERY)
    valeurs = [float(x) for x in vecteur]
    try:
        exacts = await voisins_exacts(db, valeurs)
    finally:
        await db.rollback()
    rappels: Dict[int, float] = {}
    for ef_search in EF_SEARCH_DU_RAPPEL:
        try:
            index = await voisins_par_l_index(db, valeurs, ef_search=ef_search)
        finally:
            await db.rollback()
        rappels[ef_search] = rappel_a_k(index, exacts)
    return rappels


# ==================== VERDICT D'UNE QUESTION ====================


def _extrait(chunk: ChunkResult) -> str:
    return f"{chunk.reference} art. {chunk.number}"


@dataclass
class Verdict:
    """Ce que le controle a vu pour une question, et pourquoi elle echoue."""

    question: QuestionDeReference
    raisons: List[str] = field(default_factory=list)
    # Rang (a partir de 1) du premier extrait lu d'un texte attendu.
    rang: Optional[int] = None
    # Rang ou la reponse est lue : le premier extrait qui porte un article
    # attendu avec sa regle, ou, s'il renvoie a un autre article, le rang ou
    # celui-ci est lu aussi (le plus tardif des deux).
    rang_article: Optional[int] = None
    # Extraits d'autres textes lus avant le premier qui repond, sans les
    # toleres : la regle en veut zero.
    intrus: Optional[int] = None
    # Ces extraits, « reference art. numero », pour le detail d'un echec.
    autres_textes_avant: List[str] = field(default_factory=list)
    # Les extraits toleres par la question, lus avant le premier qui repond.
    toleres_lus_avant: List[str] = field(default_factory=list)
    # Question qui nomme un article : l'a-t-il lu en premier, par la lecture
    # directe ? None pour les autres questions.
    article_nomme_en_tete: Optional[bool] = None
    # Rang du premier voisin semantique d'un texte attendu.
    rang_semantique: Optional[int] = None
    lus: List[str] = field(default_factory=list)
    # Rappel@10 de l'index, par hnsw.ef_search.
    rappels: Optional[Dict[int, float]] = None
    chemin: Optional[str] = None
    essais: int = 1
    duree_ms: int = 0

    @property
    def ok(self) -> bool:
        return not self.raisons

    def resume(self) -> Dict[str, Any]:
        sortie: Dict[str, Any] = {
            "question": self.question.question,
            "langue": self.question.langue,
            "ok": self.ok,
            "rang_dans_le_contexte": self.rang,
            "rang_de_l_article": self.rang_article,
            "intrus_avant_la_reponse": self.intrus,
            "rang_semantique": self.rang_semantique,
            "chemin_semantique": self.chemin,
            "essais": self.essais,
            "duree_ms": self.duree_ms,
        }
        # Seulement quand il y a lieu : le rapport en lit 51.
        if not self.question.garantie:
            # Son echec ne fait pas rendre 503 : le lecteur doit le savoir.
            sortie["faiblesse_connue"] = True
        if self.toleres_lus_avant:
            sortie["toleres_avant_la_reponse"] = list(self.toleres_lus_avant)
        if self.article_nomme_en_tete is not None:
            sortie["article_nomme_en_tete"] = self.article_nomme_en_tete
        if self.rappels is not None:
            sortie["rappel_a_10"] = {str(ef): r for ef, r in self.rappels.items()}
        return sortie

    def detail(self) -> Dict[str, Any]:
        """Pour un echec : ce qui manque, et ce que le modele aurait lu."""
        sortie = {
            **self.resume(),
            "attendus": [list(groupe) for groupe in self.question.textes],
            "articles_attendus": _articles_en_clair(self.question.articles_attendus),
            "raisons": list(self.raisons),
            "lus": list(self.lus),
        }
        if self.question.renvoi:
            sortie["renvoi_de_l_article_qui_repond"] = _articles_en_clair(self.question.renvoi)
        if self.autres_textes_avant:
            sortie["autres_textes_avant_la_reponse"] = list(self.autres_textes_avant)
        if self.question.article_nomme is not None:
            sortie["article_nomme"] = _articles_en_clair((self.question.article_nomme,))
        return sortie

    def en_faiblesse(self) -> Dict[str, Any]:
        """
        Pour une faiblesse connue : son etat (FAIBLESSE_CONNUE tant qu'elle
        echoue, FAIBLESSE_RESOLUE des qu'elle tient), le diagnostic de la
        liste, et ce que le modele lit AVANT la reponse : ce qui la retarde ou
        la remplace, que le proprietaire doit lire sans que la route rende
        503. Toute la liste lue quand la reponse ne l'est pas.
        """
        reponse = self.rang_article if self.question.articles_attendus else self.rang
        return {
            **self.detail(),
            "etat": FAIBLESSE_RESOLUE if self.ok else FAIBLESSE_CONNUE,
            "diagnostic_connu": self.question.faiblesse,
            "lus_avant_la_reponse": list(self.lus if reponse is None else self.lus[:reponse - 1]),
        }


def _cle_du_morceau(chunk: ChunkResult) -> Tuple[str, str]:
    """(reference, numero du morceau) : la forme de QuestionDeReference.toleres_avant."""
    return (chunk.reference or "", (chunk.number or "").strip())


def _l_article_nomme_en_tete(
    article_nomme: Tuple[str, str], lus: Sequence[ChunkResult]
) -> Optional[str]:
    """
    None si le premier extrait lu porte l'article que la question nomme, lu
    par la lecture directe ; sinon ce qui ne va pas.

    Lu en tete par la RECHERCHE seulement, il ne suffit pas : la lecture
    directe est ce qui protege toutes les questions qui nomment un article,
    et ces questions sont la pour la surveiller. Coupee, aucune des cinq
    n'avait son article en tete (mesure du 29/10/2026) ; mais un reglage de
    la recherche pourrait l'y remettre pour l'une d'elles, et la panne
    passerait inapercue.
    """
    reference, numero = article_nomme
    nomme = _articles_en_clair((article_nomme,))
    rang = next(
        (i for i, c in enumerate(lus, start=1) if porte_l_article(c, reference, numero)), None
    )
    if rang is None:
        tete = f"en tete : {_extrait(lus[0])}" if lus else "contexte vide"
        return f"l'article nomme par la question ({nomme}) n'est pas lu ({tete})"
    if rang != 1:
        return (
            f"l'article nomme par la question ({nomme}) n'est pas le premier extrait lu : "
            f"lu au rang {rang}, derriere {_extrait(lus[0])}"
        )
    if lus[0].source != _SOURCE_ARTICLE_NOMME:
        return (
            f"l'article nomme par la question ({nomme}) est lu en tete par la recherche "
            f"(origine {lus[0].source}), pas par la lecture directe de l'article nomme"
        )
    return None


def juger(
    item: QuestionDeReference,
    contexte: ContexteDuChat,
    rappels: Optional[Dict[int, float]] = None,
) -> Verdict:
    """
    Le verdict d'une question, d'apres son contexte et, s'il y a lieu, le
    rappel de l'index. La regle de ce que le modele lit avant l'article qui
    repond : voir RANG_MAX_DE_LA_REPONSE.
    """
    verdict = Verdict(
        question=item, lus=[_extrait(c) for c in contexte.lus], rappels=rappels,
        chemin=contexte.chemin,
    )
    admises = set(item.attendues())
    toleres = set(item.toleres_avant)
    lus = contexte.lus

    verdict.rang = next((i for i, c in enumerate(lus, start=1) if c.reference in admises), None)
    rang_de_l_article: Optional[int] = None
    rang_du_renvoi: Optional[int] = None
    if item.articles_attendus:
        rang_de_l_article = _premier_rang(lus, lambda c: _repond(item, c, item.articles_attendus))
        rang_du_renvoi = _premier_rang(lus, lambda c: _repond(item, c, item.renvoi))
        # Un article qui renvoie a un autre pour sa peine ne repond qu'avec
        # lui : la reponse est lue au plus tardif des deux.
        if rang_de_l_article is not None and (not item.renvoi or rang_du_renvoi is not None):
            verdict.rang_article = max(rang_de_l_article, rang_du_renvoi or 0)
    # Le premier extrait qui repond : un article attendu, ou a defaut d'articles
    # declares, un texte attendu.
    reponse = verdict.rang_article if item.articles_attendus else verdict.rang

    manquants = item.exigences_manquantes(c.reference for c in lus)
    if manquants:
        raison = (
            "textes attendus absents du contexte lu par le modele : "
            + " ; ".join(" ou ".join(groupe) for groupe in manquants)
        )
        if contexte.refus:
            raison += " (la barriere du chat a ecarte tout le contexte)"
        elif not lus:
            raison += " (contexte vide)"
        verdict.raisons.append(raison)
    elif reponse is None and rang_de_l_article is not None:
        verdict.raisons.append(
            f"l'extrait qui repond ({_extrait(lus[rang_de_l_article - 1])}, rang "
            f"{rang_de_l_article}) renvoie a {_articles_en_clair(item.renvoi)}, que le modele "
            "ne lit pas" + _lus_sans_la_regle(lus, item.renvoi)
        )
    elif reponse is None:
        verdict.raisons.append(
            "aucun article qui repond n'est lu par le modele : "
            + _articles_en_clair(item.articles_attendus)
            + _lus_sans_la_regle(lus, item.articles_attendus)
        )
    else:
        avant = [c for c in lus[:reponse - 1] if c.reference not in admises]
        verdict.toleres_lus_avant = [_extrait(c) for c in avant if _cle_du_morceau(c) in toleres]
        verdict.autres_textes_avant = [
            _extrait(c) for c in avant if _cle_du_morceau(c) not in toleres
        ]
        verdict.intrus = len(verdict.autres_textes_avant)
        if verdict.autres_textes_avant:
            verdict.raisons.append(
                f"un autre texte est lu avant l'extrait qui repond (rang {reponse}) : "
                + ", ".join(verdict.autres_textes_avant)
            )
        if reponse > RANG_MAX_DE_LA_REPONSE:
            verdict.raisons.append(
                f"extrait qui repond au rang {reponse}, au-dela des {RANG_MAX_DE_LA_REPONSE} "
                "premiers extraits lus"
            )

    if item.article_nomme is not None:
        raison_du_nomme = _l_article_nomme_en_tete(item.article_nomme, lus)
        verdict.article_nomme_en_tete = raison_du_nomme is None
        if raison_du_nomme is not None:
            verdict.raisons.append(raison_du_nomme)

    if contexte.volet_en_panne:
        verdict.raisons.append(
            "volet semantique indisponible : le chat repondrait du plein texte seul"
        )
    else:
        verdict.rang_semantique = next(
            (i for i, (_, ref) in enumerate(contexte.voisins or [], start=1) if ref in admises),
            None,
        )
        if verdict.rang_semantique is None or verdict.rang_semantique > VOISINS_SEMANTIQUES:
            verdict.raisons.append(
                f"aucun texte attendu parmi les {VOISINS_SEMANTIQUES} premiers voisins "
                "du volet semantique"
            )

    for ef_search, rappel in sorted((rappels or {}).items()):
        if rappel < SEUIL_RAPPEL:
            verdict.raisons.append(
                f"rappel@{K_RAPPEL} de l'index HNSW a ef_search {ef_search} : "
                f"{rappel:.1f}, sous le seuil de {SEUIL_RAPPEL}"
            )
    return verdict


FabriqueDeSessions = Callable[[], Any]


async def verifier_une_question(
    fabrique: FabriqueDeSessions, item: QuestionDeReference
) -> Verdict:
    """
    Une question de bout en bout, sur sa propre session, rendue intacte
    (rollback) : contexte, verdict, et pour les questions `rappel` la mesure
    de l'index, dans des transactions a part.

    Un volet semantique en panne a droit a un SECOND ESSAI : au premier
    calcul apres un redemarrage, chaque question demande son vecteur a
    Voyage, et un seul appel lent ne doit pas faire sonner l'alerte pour une
    demi-heure. Deux echecs de suite en sont une.

    Une erreur devient un echec qui la NOMME par sa classe, jamais par son
    message : le rapport est public, et un message de pilote peut citer
    l'hote de la base.
    """
    debut = time.monotonic()
    essais = 0
    try:
        async with fabrique() as db:
            for essais in range(1, ESSAIS + 1):
                if essais > 1:
                    await asyncio.sleep(PAUSE_ENTRE_DEUX_ESSAIS_S)
                try:
                    contexte, recherche = await contexte_du_chat(db, item.question, item.langue)
                finally:
                    await db.rollback()
                if not contexte.volet_en_panne:
                    break
                logger.warning(
                    f"⚠️ Controle du chat : « {item.question} », volet semantique en panne "
                    f"(essai {essais} sur {ESSAIS})"
                )
            rappels: Optional[Dict[int, float]] = None
            echec_du_rappel: Optional[str] = None
            if item.rappel and not contexte.volet_en_panne:
                try:
                    rappels = await rappel_de_l_index(db, recherche, item.question)
                except Exception as erreur:
                    echec_du_rappel = type(erreur).__name__
                    logger.warning(
                        f"⚠️ Controle du chat : mesure de l'index en echec ({erreur!r})"
                    )
        verdict = juger(item, contexte, rappels)
        if echec_du_rappel:
            verdict.raisons.append(f"mesure de l'index en echec ({echec_du_rappel})")
    except Exception as erreur:
        logger.warning(f"⚠️ Controle du chat : « {item.question} » en echec ({erreur!r})")
        verdict = Verdict(
            question=item, raisons=[f"recuperation en echec ({type(erreur).__name__})"]
        )
    verdict.essais = max(essais, 1)
    verdict.duree_ms = int((time.monotonic() - debut) * 1000)
    return verdict


# ==================== LE FOURNISSEUR D'EMBEDDINGS ====================


async def sonder_le_fournisseur(
    question: str = QUESTIONS_DE_REFERENCE[0].question,
    *,
    essais: int = ESSAIS,
    pause_s: float = PAUSE_ENTRE_DEUX_ESSAIS_S,
) -> Optional[str]:
    """
    None si le fournisseur d'embeddings du chat rend un vecteur, sinon ce qui
    ne va pas.

    POURQUOI. Le service d'embeddings garde en memoire le vecteur de chaque
    question deja vue (_VecteursRecents) : des le deuxieme calcul, les
    questions de reference ne passent plus par Voyage, et une panne de Voyage
    passerait inapercue. Or elle fait repondre le chat du plein texte seul,
    qui ne sait rien trouver pour une question anglaise dans un corpus
    francais. Vu le 29/10/2026 vers 17 h : Voyage repondait en 6,5 a 7 s,
    au-dela du delai d'une question (3 s), et le chat lisait le plein texte
    seul (« Semantic search failed in hybrid, plein texte seul »). Un appel,
    au fournisseur meme du chat, hors de cette memoire, borne par son delai.

    DEUX ESSAIS : un appel lent isole (une poignee TLS de plus de 2 s, mesuree
    depuis le poste) rendait le controle rouge pour une demi-heure, puis vert :
    du bruit, et une alerte qui sonne pour rien finit ignoree. Deux echecs a
    une seconde d'intervalle, c'est ce que vivent les utilisateurs.
    """
    service = search_service._embedding_service_instance
    if service is None:
        return "service d'embeddings absent : le chat repondrait du plein texte seul"
    derniere: Optional[BaseException] = None
    for essai in range(1, essais + 1):
        if essai > 1:
            await asyncio.sleep(pause_s)
        try:
            sonde = EmbeddingService(use_cache=False, provider=service.provider)
            await sonde.generate_embedding_async(question, task_type=sonde.TASK_QUERY)
            return None
        except Exception as erreur:
            derniere = erreur
            logger.warning(
                f"⚠️ Controle du chat : fournisseur d'embeddings en echec, essai {essai} "
                f"sur {essais} ({erreur!r})"
            )
    return f"fournisseur d'embeddings en echec, {essais} essais ({type(derniere).__name__})"


# ==================== LE CONTROLE ENTIER ====================


def _maintenant_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


async def controler_le_chat(
    fabrique: FabriqueDeSessions,
    questions: Sequence[QuestionDeReference] = QUESTIONS_DE_REFERENCE,
    *,
    concurrence: int = CONCURRENCE,
    delai_s: float = DELAI_DU_CALCUL_S,
    sonde: Optional[Callable[[], Awaitable[Optional[str]]]] = None,
) -> Dict[str, Any]:
    """
    Le rapport du controle : « ok » si les questions GARANTIES et le controle
    entier tiennent, « echec » sinon, avec ses `raisons` et les questions
    garanties en echec (`echecs`). Les faiblesses connues sont rapportees a
    part (`faiblesses`, `faiblesses_resolues`) et ne font jamais « echec » :
    voir la docstring du module. Borne a `delai_s` : les questions encore en
    cours a l'echeance sont arretees et comptent comme des echecs.

    `sonde` : la sonde du fournisseur d'embeddings (sonder_le_fournisseur, en
    production), lancee D'ABORD. En echec, les questions ne sont pas
    verifiees : sans vecteur, elles ne diraient que la panne deja connue, et
    chacune tiendrait un fil d'appel a Voyage (_EXECUTEUR_QUESTIONS, 4 fils
    partages avec les utilisateurs) jusqu'a son delai.
    """
    debut = time.monotonic()
    verifie_le = _maintenant_iso()

    fournisseur: Optional[str] = None
    if sonde is not None:
        try:
            async with asyncio.timeout(delai_s):
                fournisseur = await sonde()
        except TimeoutError:
            fournisseur = f"sonde du fournisseur sans reponse en {delai_s:.0f} s"
        except Exception as erreur:
            fournisseur = f"sonde du fournisseur en erreur ({type(erreur).__name__})"
    a_verifier = list(questions) if fournisseur is None else []

    places = asyncio.Semaphore(concurrence)

    async def bornee(item: QuestionDeReference) -> Verdict:
        async with places:
            return await verifier_une_question(fabrique, item)

    taches = [asyncio.create_task(bornee(item)) for item in a_verifier]
    try:
        if taches:
            await asyncio.wait(taches, timeout=max(0.0, delai_s - (time.monotonic() - debut)))
    finally:
        # Echeance passee, ou calcul annule : rien ne doit lui survivre, ni
        # tenir une connexion.
        restantes = [t for t in taches if not t.done()]
        for tache in restantes:
            tache.cancel()
        if restantes:
            await asyncio.wait(restantes)

    verdicts: List[Verdict] = []
    for item, tache in zip(a_verifier, taches):
        if tache.cancelled() or tache.exception() is not None:
            verdicts.append(Verdict(
                question=item,
                raisons=[f"non verifiee : le controle a depasse {delai_s:.0f} s"],
            ))
        else:
            verdicts.append(tache.result())

    # Seules les questions GARANTIES font le 503 ; les faiblesses connues sont
    # rapportees a part, qu'elles echouent encore ou qu'elles tiennent de
    # nouveau (voir la docstring du module).
    garanties = [v for v in verdicts if v.question.garantie]
    faiblesses = [v for v in verdicts if not v.question.garantie]
    echecs = [v for v in garanties if not v.ok]
    resolues = [v for v in faiblesses if v.ok]
    # Le repli se compte sur toutes les questions : c'est un etat de la base,
    # pas celui d'une question.
    replis = sum(1 for v in verdicts if v.chemin == _CHEMIN_REPLI)
    raisons: List[str] = []
    if fournisseur is not None:
        raisons.append(f"{fournisseur} : questions non verifiees")
    elif not verdicts:
        raisons.append("aucune question verifiee")
    elif not garanties:
        # Des faiblesses seules ne garantissent rien : rien de verifie n'est
        # pas un succes.
        raisons.append("aucune question garantie verifiee")
    if echecs:
        raisons.append(f"{len(echecs)} question(s) garantie(s) en echec sur {len(garanties)}")
    if verdicts and replis > PART_MAX_EN_REPLI * len(verdicts):
        raisons.append(
            f"recherche exacte du chat en repli sur l'index HNSW pour {replis} questions "
            f"sur {len(verdicts)} : la base est trop lente pour elle, et le chat lit "
            "l'index, dont la perte a fait l'incident du 29/10/2026"
        )

    rapport: Dict[str, Any] = {
        # Rien de verifie n'est pas un succes.
        "statut": "ok" if not raisons else "echec",
        "raisons": raisons,
        "verifie_le": verifie_le,
        "duree_ms": int((time.monotonic() - debut) * 1000),
        "questions": len(questions),
        "verifiees": len(verdicts),
        # Les questions garanties verifiees, et celles qui echouent : elles
        # seules font le 503.
        "garanties": len(garanties),
        "en_echec": len(echecs),
        # Les faiblesses connues verifiees, celles qui echouent encore, et
        # celles qui tiennent de nouveau (« faiblesse résolue ») : jamais de
        # 503, le detail est dans `faiblesses`.
        "faiblesses_connues": len(faiblesses),
        "faiblesses_en_echec": len(faiblesses) - len(resolues),
        "faiblesses_resolues": [v.question.question for v in resolues],
        # « ok », « non sonde » (pas de sonde), ou ce qui ne va pas.
        "fournisseur_d_embeddings": (
            fournisseur if fournisseur is not None
            else ("ok" if sonde is not None else "non sonde")
        ),
        # Qui a servi le volet semantique des questions : le repli de la
        # recherche exacte sur l'index s'y compte.
        "chemins_semantiques": dict(Counter(v.chemin or "inconnu" for v in verdicts)),
        "echecs": [v.detail() for v in echecs],
        "faiblesses": [v.en_faiblesse() for v in faiblesses],
        "rappel_hnsw": [
            {
                "question": v.question.question,
                "rappel_a_10": (
                    None if v.rappels is None else {str(ef): r for ef, r in v.rappels.items()}
                ),
                "seuil": SEUIL_RAPPEL,
            }
            for v in verdicts if v.question.rappel
        ],
        "questions_verifiees": [v.resume() for v in verdicts],
    }
    _journaliser(rapport, echecs, faiblesses)
    return rapport


def _journaliser(
    rapport: Dict[str, Any], echecs: List[Verdict], faiblesses: List[Verdict]
) -> None:
    en_echec = [v for v in faiblesses if not v.ok]
    if en_echec:
        logger.info(
            f"🩺 Controle du chat : {len(en_echec)} faiblesse(s) connue(s) en echec, sans 503 : "
            + " ; ".join(f"« {v.question.question} »" for v in en_echec)
        )
    for verdict in faiblesses:
        if verdict.ok:
            # Une bonne nouvelle, mais une action : tant qu'elle reste dans la
            # liste, sa prochaine panne ne ferait pas rendre 503.
            logger.warning(
                f"⚠️ Controle du chat : {FAIBLESSE_RESOLUE}, « {verdict.question.question} » "
                "tient : la retirer de FAIBLESSES_CONNUES (questions_de_reference)"
            )
    if rapport["statut"] == "ok":
        logger.info(
            f"🩺 Controle du chat : {rapport['garanties']} questions garanties tiennent, en "
            f"{rapport['duree_ms'] / 1000:.1f} s, chemins {rapport['chemins_semantiques']}"
        )
        return
    logger.warning(f"⚠️ Controle du chat EN ECHEC : {' ; '.join(rapport['raisons'])}")
    if echecs:
        logger.warning(
            "⚠️ Controle du chat, questions en echec : " + " ; ".join(
                f"« {v.question.question} » ({', '.join(v.raisons)})" for v in echecs[:5]
            )
        )


# ==================== LE RAPPORT GARDE ====================


class ControleEnCache:
    """
    Le dernier rapport du controle, et le calcul en cours.

    POURQUOI. La route est publique et un calcul fait 51 recherches : il ne
    doit pas partir a la demande. Un calcul au plus par `duree_de_vie_s`
    (30 minutes), quels que soient le nombre d'appels et l'issue du
    precedent — succes, echec, delai depasse ou erreur. Entre deux, le
    dernier rapport est rendu tel quel, avec son age.

    Les appels qui arrivent pendant un calcul attendent le MEME calcul, au
    plus `attente_max_s`. Au-dela, le rapport precedent s'il a moins de
    `age_max_du_precedent_s` (marque `perime`), avec son propre code. Sinon
    « en_cours », en 202 et non en 503 : le premier appel apres chaque
    redemarrage (chaque deploiement) tombe la, des que le calcul depasse
    l'attente, ce qui sera la regle sur Neon ; en 503, cron-job.org enverrait
    une fausse alerte a chaque deploiement, et une alerte qui sonne pour rien
    finit ignoree. Le calcul est une tache a part : un appelant qui raccroche
    ne l'annule pas, et son rapport sert l'appel suivant.

    Le rapport vit dans la memoire du processus : un redemarrage le perd. Le
    planning conseille (app/api/routes/controle.py) appelle aux minutes 0 et
    5, pendant que la sonde de reveil garde l'API eveillee : l'appel de la
    minute 5 lit le calcul lance a la minute 0. Un 202 a la minute 5 veut
    dire un redemarrage entre les deux (un deploiement).

    Sans verrou : entre deux `await`, rien d'autre ne tourne dans la boucle
    d'evenements, et la decision de lancer un calcul n'en contient aucun.
    Un seul processus (uvicorn --workers 1, Dockerfile) : un seul calcul.
    """

    def __init__(
        self,
        calculer: Callable[[], Awaitable[Dict[str, Any]]],
        *,
        duree_de_vie_s: float = DUREE_DE_VIE_S,
        attente_max_s: float = ATTENTE_MAX_S,
        delai_max_s: float = DELAI_DU_CALCUL_S + 15.0,
        age_max_du_precedent_s: float = AGE_MAX_DU_PRECEDENT_S,
        horloge: Callable[[], float] = time.monotonic,
    ):
        self._calculer = calculer
        self.duree_de_vie_s = duree_de_vie_s
        self.attente_max_s = attente_max_s
        self.delai_max_s = delai_max_s
        self.age_max_du_precedent_s = age_max_du_precedent_s
        self._horloge = horloge
        self._tache: Optional["asyncio.Task[Dict[str, Any]]"] = None
        self._lance_a: Optional[float] = None
        self._rapport: Optional[Dict[str, Any]] = None
        self._fini_a: Optional[float] = None
        self.calculs = 0

    async def lire(self) -> Tuple[int, Dict[str, Any]]:
        """
        (code HTTP, rapport) : 200 si le rapport est « ok » (les questions
        garanties tiennent ; une faiblesse connue n'y change rien), 503 sinon,
        202 tant qu'aucun rapport assez recent n'existe.
        """
        boucle = asyncio.get_running_loop()
        if self._tache is not None and self._tache.get_loop() is not boucle:
            # Tache d'une boucle fermee (un test precedent) : elle ne finira
            # pas, et son rapport ne viendra jamais. Un calcul neuf repart.
            self._tache = None
            self._lance_a = None

        if self._tache is None and (
            self._lance_a is None or self._horloge() - self._lance_a >= self.duree_de_vie_s
        ):
            self._lance_a = self._horloge()
            self.calculs += 1
            self._tache = asyncio.create_task(self._calculer_sans_lever())
            self._tache.add_done_callback(self._ranger)

        tache = self._tache
        if tache is not None:
            if not tache.done():
                # asyncio.wait n'annule pas la tache a l'echeance, ni quand
                # l'appelant est lui-meme annule.
                await asyncio.wait({tache}, timeout=self.attente_max_s)
            if tache.done():
                self._ranger(tache)
            else:
                return self._pendant_le_calcul()

        if self._rapport is None:
            return CODE_EN_COURS, {"statut": "en_cours", "raison": "aucun rapport encore"}
        return self._rendre(self._rapport)

    def _pendant_le_calcul(self) -> Tuple[int, Dict[str, Any]]:
        """Le calcul depasse l'attente : le rapport precedent s'il est assez recent."""
        en_cours = int(self._horloge() - (self._lance_a or self._horloge()))
        if self._rapport is not None and self._fini_a is not None and (
            self._horloge() - self._fini_a <= self.age_max_du_precedent_s
        ):
            code, rapport = self._rendre(self._rapport)
            rapport["perime"] = True
            rapport["calcul_en_cours_depuis_s"] = en_cours
            return code, rapport
        return CODE_EN_COURS, {
            "statut": "en_cours",
            "raison": f"controle en cours depuis {en_cours} s, plus de "
                      f"{self.attente_max_s:.0f} s d'attente : le resultat servira "
                      "l'appel suivant",
        }

    def _rendre(self, rapport: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        sortie = dict(rapport)
        maintenant = self._horloge()
        sortie["age_s"] = int(maintenant - (self._fini_a or maintenant))
        sortie["prochain_calcul_dans_s"] = max(
            0, int(self.duree_de_vie_s - (maintenant - (self._lance_a or maintenant)))
        )
        return (CODE_OK if sortie.get("statut") == "ok" else CODE_ECHEC), sortie

    def _ranger(self, tache: "asyncio.Task[Dict[str, Any]]") -> None:
        """Garde le rapport d'une tache finie (une fois), et oublie la tache."""
        if tache is not self._tache:
            return
        self._tache = None
        self._fini_a = self._horloge()
        if tache.cancelled():
            self._rapport = {
                "statut": "echec", "verifie_le": _maintenant_iso(),
                "raison": "controle interrompu",
            }
            return
        self._rapport = tache.result()

    async def _calculer_sans_lever(self) -> Dict[str, Any]:
        """Le calcul, borne ; toute erreur devient un rapport d'echec, par sa classe."""
        try:
            async with asyncio.timeout(self.delai_max_s):
                return await self._calculer()
        except TimeoutError:
            logger.error(f"❌ Controle du chat : plus de {self.delai_max_s:.0f} s, abandonne")
            return {
                "statut": "echec", "verifie_le": _maintenant_iso(),
                "raison": f"controle abandonne apres {self.delai_max_s:.0f} s",
            }
        except Exception as erreur:
            logger.exception("❌ Controle du chat en erreur")
            return {
                "statut": "echec", "verifie_le": _maintenant_iso(),
                "raison": f"controle en erreur ({type(erreur).__name__})",
            }
