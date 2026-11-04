"""
Les questions de reference du chat, les textes que le modele DOIT lire pour
chacune, et les articles qui y REPONDENT.

POURQUOI. Incident du 29/10/2026 : « Quelles indemnités en cas de licenciement
abusif ? », une question que le chat propose lui-meme, a recu une reponse
batie sur des tableaux de lois de finances. L'index HNSW de production ne
rendait plus aucun voisin juste pour elle ; le modele a dit que les documents
ne contenaient rien, puis a cite de memoire un « article 76 du Code du
travail ». Rien ne l'a signale avant qu'un utilisateur le voie. Ces questions
sont le filet : si l'une d'elles ne fait plus lire ses articles au modele, un
test echoue avant l'envoi du code, et le controle de production le dit au
proprietaire avant un utilisateur.

TROIS FAMILLES :
- toutes les questions suggerees par le front (src/lib/suggestions.ts du
  depot JuriX-frontend-main) : tous les profils, les deux langues, sans
  doublon. Un clic les ENVOIE, elles sont donc les plus posees ;
- des questions de fond, sur les sujets les plus demandes : licenciement et
  preavis, vol, escroquerie, garde a vue, divorce, titre foncier, SARL, TVA,
  corruption ;
- des questions qui NOMMENT un article (« Que dit l'article 39 du Code du
  travail ? ») : la forme la plus courante chez les juristes, et celle de
  l'incident (« article 76 du Code du travail »). Elles passent par la lecture
  directe de l'article nomme (RAGService._article_nomme), qu'aucune autre
  question n'exerce : le controle exige que cet article soit le premier
  extrait lu, et par elle (champ `article_nomme`).

CE QUE LE MODELE DOIT LIRE, releve le 29/10/2026 sur jurix_dev (memes
articles et memes vecteurs que la production), par une recherche exacte sans
l'index HNSW et par le contexte du chat, puis EN LISANT les morceaux :
- `textes` : chaque element est une exigence, et une exigence a plusieurs
  references se lit « l'un de ces textes » ;
- `articles_attendus` : les articles qui repondent, (reference, numero). L'un
  d'eux au moins doit etre parmi les extraits lus. Le texte seul ne suffit
  pas : le bon code peut arriver par un article hors sujet. Une revue l'a
  montre : sans le morceau 35.1 du Code du travail (en-tetes des articles 36 a
  39), « Quelles indemnités en cas de licenciement abusif ? » restait verte
  au niveau du texte, et le modele lisait d'abord l'indemnite du FONCTIONNAIRE
  licencie (DEC-1994-199 art. 120). controle_du_chat.porte_l_article dit
  comment un morceau porte un article : son numero (« 35.1 », le premier
  morceau de l'article 35), ou un en-tete « Article 37.- » dans son contenu
  (le corpus range les articles 36 a 39 du Code du travail sous « 35.1 »).
  Un numero « N.x » designe un morceau precis, quand la reponse est dans la
  suite d'un article : l'alinea 2 de l'article 65 de la loi sur la
  cybercriminalite est le morceau « 65.2 » ;
- UN ARTICLE QUI REPOND PORTE LA REGLE DEMANDEE (la peine, la duree, la
  definition), et non un article qui en parle ou y renvoie. Un article qui ne
  fait que renvoyer aux peines d'un autre (« Les peines de l'article 318
  ci-dessus sont doublées ») n'est JAMAIS attendu seul : il ne repond
  qu'avec l'article vise (`renvoi`), ou pas du tout. Revue du tour 4
  (29/10/2026) : l'art. 320 du Code penal gardait verte la peine du vol, la
  ou le modele lisait le Code de justice militaire avant l'art. 318 ; le
  morceau 35.1 (les obligations PENDANT le preavis) gardait vert le delai de
  preavis, la ou il lisait d'abord le preavis d'un agent commercial. Chaque
  article attendu a ete relu le 30/10/2026 dans les morceaux de jurix_dev,
  et ceux qui ne portaient pas la regle sont sortis : art. 320 et 321 (renvoi
  aux peines de l'art. 318), art. 35 (preavis), 120 (prorogation de la garde
  a vue), 97 de l'AUSCGIE (obligation de s'immatriculer), 46 de l'AUDCG
  (mentions de la demande), 870 a 872 de l'AUSCGIE (traits du GIE, pas sa
  definition), 36 et 37 du Code du travail pour le licenciement ABUSIF, 123
  du CPP pour l'avocat, 69 de la loi sur la cybercriminalite (un autre delit,
  l'acces EN VIOLATION des mesures de securite, puni d'une autre amende) ; et
  le morceau 65.2 n'y repond plus seul (renvoi aux peines de l'alinea 1) ;
- `regles` : les mots de cette regle, (reference, numero, mots), relus au mot
  pres dans le morceau qui repond (controle_du_chat.porte_la_regle). Le meme
  numero range parfois plusieurs articles (« 134 » du Code penal porte aussi
  les art. 134-1 et 134-2, « 311 » de l'AUSCGIE l'art. 311-1) : les mots
  designent celui qui repond. Toute question qui ne nomme pas d'article en
  declare pour chacun de ses articles ;
- `renvoi` : quand l'article qui repond renvoie a un autre pour sa peine
  (« Est puni des peines prévues à l'alinéa 1 ci-dessus, tout accès non
  autorisé », morceau 65.2), l'article vise (morceau 65.1) : la reponse n'est
  lue qu'avec les deux, au rang du plus tardif ;
- `article_nomme` : pour une question qui nomme un article, (reference,
  numero). Il doit etre le PREMIER extrait lu, par la lecture directe. Un
  champ, et non le texte de la question : « l'article 76 » dit le numero,
  pas le texte que la lecture directe doit trouver ;
- `toleres_avant` : les extraits d'AUTRES textes, (reference, numero du
  morceau), que le modele peut lire AVANT l'article qui repond. Hors de
  cette liste, un autre texte lu avant lui fait echouer la question : c'est
  un autre regime (le Statut de la fonction publique avant le Code du
  travail). Chacun est lu, et juge incapable de fausser la reponse. Il est
  sans rapport avec la question, ou c'est une version anterieure que le
  prompt signale reecrite par un texte attendu. Seuls ceux que la mesure a
  montres y sont, voir leurs constantes.

LA REGLE DU CONTROLE pour ces champs : controle_du_chat, a
RANG_MAX_DE_LA_REPONSE (aucun autre texte avant l'article qui repond, hors
tolerances ; cet article, avec sa regle et son renvoi, parmi les 4 premiers
extraits lus ; l'article nomme en tete).

Les numeros sont pris tels que la base les porte, et seulement quand ils sont
surs. L'AUPSRVE range sous le numero « 1 » les articles 1 et 1-2 a 1-16
(notification, delais...) : « 1 » n'y designe pas l'article 1, il n'est donc
jamais attendu ; l'article 9 (opposition) y est un morceau « SECTION_3 » sans
numero ni en-tete : on attend 10 et 11.

UN TEXTE ADMIS DOIT PERMETTRE UNE REPONSE JUSTE ET A JOUR pour les
suggestions : la loi sur la cybercriminalite n'a aucun article sur
l'escroquerie, et la Constitution de 1996 (PRC-6283) dit encore le mandat
« renouvelable une fois » que la revision de 2026 et le Code electoral ne
disent plus ; ni l'une ni l'autre ne suffit donc seule. Les questions de fond
ne controlent que la RECUPERATION : deux d'entre elles ont une reponse
incomplete dans le corpus (SARL, titre foncier : voir leurs notes), raison
pour laquelle le front ne les suggere pas.

QUATRE SUGGESTIONS ANGLAISES REFORMULEES (29/10/2026). Mesure (jurix_dev,
vrai Voyage, recherche exacte et contexte du chat, code de production d99e23c
comme code de cette branche), les formulations du front faisaient lire au
modele :
- « What does the Penal Code say about obtaining money by false pretences on
  the internet? » : aucun extrait du Code penal. Art. 318 au 33e rang de la
  recherche exacte (similarite 0,319), derriere douze articles de la loi sur
  la cybercriminalite (0,36 a 0,44), qui n'en a aucun sur l'escroquerie. Sept
  variantes avec « internet » ou « online » echouent de meme ;
- « What compensation is due for unfair dismissal? » : le morceau 35.1 au 6e
  rang, derriere l'indemnite d'un agent commercial (AUDCG 231) et celle d'un
  fonctionnaire (DEC-1994-199 120, PRC-8798 20) ;
- « How can I make a customer pay an unpaid invoice? » : l'injonction de
  payer (AUPSRVE art. 2) au 6e rang, derriere le recouvrement des creances
  des ENTITES PUBLIQUES (PRC-9703 40 et 36) et une loi de finances ;
- « How do I register a company in the trade and personal property credit
  register? » : ni l'article 46 (societes) ni le 47 (pieces), dont les morceaux
  commencent par du bruit d'OCR ; le modele lisait l'article 44, celui des
  PERSONNES PHYSIQUES, et des inscriptions de suretes. Cinq formulations
  essayees, aucune ne les fait lire.
Les formulations retenues ci-dessous mettent l'article qui repond au rang 1
du contexte, avec les deux codes. Elles sont reportees au front
(suggestions.ts, commit 4733bb9 du depot du front, 29/10/2026), mot pour mot :
test_chaque_suggestion_du_front_est_une_question_de_reference le verifie.
Articles attendus relus dans les morceaux le 29/10/2026, puis le 30/10/2026
(seuls ceux qui portent la regle restent attendus) :
- licenciement : 35.1 porte les en-tetes des art. 36 (preavis non observe),
  37 (indemnite de licenciement) et 39 (dommages-interets : au moins 3 mois
  de salaire, au plus 1 mois par annee d'anciennete) : l'art. 39 repond ;
- escroquerie : art. 318 c (« en déterminant fallacieusement la victime »,
  5 a 10 ans) ; l'art. 321 ne fait que doubler ces peines ;
- personnalite juridique : AUSCGIE art. 98 (« à compter de son
  immatriculation ») ; l'art. 97 ne dit que l'obligation de s'immatriculer ;
- facture impayee : AUPSRVE art. 2 (creance certaine, liquide, exigible, de
  cause contractuelle), 3 et 4 (requete), 5 (ordonnance).

Mesure du 29/10/2026 pour les 49 questions d'alors, avec les articles
attendus d'alors (jurix_dev, code de cette branche, vecteurs enregistres de
Voyage) :
- l'article qui repond est lu au rang 1 pour 36 questions, 2 pour 7, 3 pour
  4, 4 pour 2. Ces deux-la (droits en garde a vue, GIE) sont a la limite de
  la regle (4), derriere des articles du meme texte ;
- aucun extrait d'un autre texte ne le precede, sauf l'article 6 de la
  Constitution de 1996 pour l'election en anglais, tolere ;
- l'article nomme est en tete, par la lecture directe, pour les cinq ;
- le texte attendu est le premier voisin semantique pour 47 questions, le 2e
  et le 3e pour les deux autres.
Avec 10 tirages de gigue par question (cosinus 0,9993 ; la revue du tour 4 a
mesure jusqu'a 0,9985 avec le vrai Voyage, d'ou la mesure du 30/10/2026
ci-dessous) : les rangs ne bougent pas au-dela de 4, et deux extraits sans
rapport passent en tete parfois, toleres pour leur question (PRC-6506 art.
84, CODE-CIVIL 1384.2). Un tirage sur 490 lit en tete l'article 142 de la
loi de finances 2016 pour la TVA : il echoue, a raison.

Mesure du 30/10/2026, avec les articles qui portent la regle (51 questions,
memes conditions, gigue de 10 tirages au cosinus 0,9985 ; vrai Voyage pour
les questions en echec ; data/fiabilite/tmp/controle5/ et controle5_reprise/,
recuperation du tour 5 de 01 h 13, 04 h 18, 04 h 43 et 05 h 10, puis celle
du tour 6, rag_service d079c404 de 07 h 16, dans controle6/ et controle7/ :
memes verdicts, aux memes rangs) : 6 questions echouent, a raison, dans tous
les tirages : ce sont les FAIBLESSES_CONNUES (plus bas, avec leur
diagnostic). Les 45 autres
tiennent ; seule la TVA bascule encore, 2 tirages sur 10. Quatre sont a la
limite, l'extrait qui repond au rang 4 dans tous les tirages, derriere des
articles du meme texte : droits en garde a vue, GIE, acces non autorise en
francais (65.1 apres 86 et 69.1), duree de la garde a vue en francais (119
apres 124.1, 120.1 et 118). Les deux questions de fond ajoutees ce jour-la
viennent de la revue du tour 4 : la meme recuperation, posee autrement,
lisait d'abord un autre regime.

QUESTIONS GARANTIES ET FAIBLESSES CONNUES (30/10/2026, revue du tour 5).
Chaque question est l'une ou l'autre (QuestionDeReference.garantie), et la
route ne rend 503 que pour les premieres :
- GARANTIE : elle tient aujourd'hui, ses attendus relus dans la base. Son
  echec est une panne NOUVELLE, et la route rend 503 : index perdu, Voyage
  en panne, poids de fusion ou RRF_K deregles, lecture directe de l'article
  nomme coupee, morceau qui repond perdu ou ecarte... ;
- FAIBLESSE CONNUE (FAIBLESSES_CONNUES) : elle echoue aujourd'hui, A RAISON,
  ses attendus etant justes. C'est la recuperation qui fait lire au modele
  un autre regime avant la reponse, la reponse trop tard, ou jamais. Le
  rapport la donne (`faiblesses` : rang de la reponse, ce que le modele lit
  avant, raisons), mais elle NE FAIT JAMAIS rendre 503. Si elle tient de
  nouveau, le rapport dit « faiblesse résolue » (`faiblesses_resolues`), et
  la route ne rend pas 503 non plus : une bonne nouvelle ne doit pas sonner.
POURQUOI. Revue du tour 5 : comptees comme les autres, les 6 faiblesses
faisaient rendre 503 a chaque calcul. L'index perdu, l'incident du
29/10/2026, invisible en local, et Voyage en panne rendaient le meme 503.
Or cron-job.org ne lit que le code HTTP : il ne pouvait plus signaler une
panne NOUVELLE, et il desactive une tache qui echoue trop souvent. Separees,
les faiblesses restent sous les yeux du proprietaire sans aveugler l'alerte.
LA LISTE NE S'ALLONGE JAMAIS POUR FAIRE PASSER UN ENVOI. Une question n'y
entre que si elle echoue aujourd'hui avec ses attendus justes, et que la
mesure l'a montre (rang, ce qui est lu avant) ; elle en sort des qu'elle
tient : le test d'integration l'exige (XPASS strict), pour que sa prochaine
panne fasse de nouveau 503. Les questions `rappel` et celles qui nomment un
article n'y entrent jamais (tests/test_services/test_controle_du_chat.py) :
elles seules voient l'index perdu et la lecture directe coupee, et leur
panne ne se verrait plus. La TVA reste GARANTIE : elle tient aujourd'hui,
avec le vecteur enregistre comme avec le vrai Voyage, meme si la gigue la
fait basculer (une ancienne loi de finances lue avant, 7 a 10 tirages sur 40
le 30/10/2026) ; un 503 pour elle dit ce risque, que l'utilisateur vit aussi.

Qui les lit :
- tests/test_integration/test_questions_de_reference.py, a lancer avant
  chaque envoi (une faiblesse connue y est un xfail strict) ;
- GET /api/v1/controle/chat (app/services/controle_du_chat.py), en
  production.

UNE QUESTION SUGGEREE CHANGE AU FRONT : la reporter ici, ses textes et ses
articles releves de la meme facon, puis recrire les vecteurs enregistres
(`pytest -m voyage_live` avec JURIX_REGENERER_VECTEURS=1, voir le test
d'integration). tests/test_services/test_controle_du_chat.py compare les deux
listes quand le depot du front est a cote de celui-ci. LE CORPUS CHANGE (une
loi de finances 2027 qui reecrit l'article 142, une reingestion qui redecoupe
le Code du travail) : un texte ou un article attendu peut changer aussi ; le
test d'integration le dira.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

SUGGESTION = "suggestion"
FOND = "fond"

# Les textes, par leur reference en base.
CODE_DU_TRAVAIL = "LOI-1992-007"     # Loi n°92/007 du 14 août 1992
CODE_PENAL = "PRC-4722"              # Loi n°2016/007 du 12 juillet 2016
CODE_DE_PROCEDURE_PENALE = "LOI-2005-007"
CYBERCRIMINALITE = "LOI-2010-012"    # Loi n°2010/012 du 21 décembre 2010
REVISION_DE_2026 = "PRC-10679"       # Loi n°2026/002, réécrit les art. 5 à 7
CODE_ELECTORAL = "LOI-2012-001"
CODE_CIVIL = "CODE-CIVIL"
TITRE_FONCIER = "DEC-1976-165"       # Décret n°76/165 du 27 avril 1976
INCITATIONS_2025 = "PRC-10455"       # Ordonnance n°2025/002 du 18 juillet 2025
LOI_DE_FINANCES_2026 = "PRC-10557"   # Loi n°2025/012 du 17 décembre 2025
AU_SOCIETES = "OHADA-AUSCGIE-2014"
AU_DROIT_COMMERCIAL = "OHADA-AUDCG-2010"
AU_RECOUVREMENT = "OHADA-AUPSRVE-2023"
# Textes seulement toleres avant l'article qui repond (`toleres_avant`).
CONSTITUTION_DE_1996 = "PRC-6283"    # Loi n°96/06 du 18 janvier 1996
MARCHES_DES_ENTREPRISES_PUBLIQUES = "PRC-6506"  # Décret n°2018/355 du 12 juin 2018


@dataclass(frozen=True)
class QuestionDeReference:
    """Une question, sa langue, les textes et les articles que le modele doit lire."""

    question: str
    langue: str
    textes: Tuple[Tuple[str, ...], ...]
    origine: str
    # Ce qui a ete lu, en clair : la raison des textes et des articles.
    articles: str
    # Les articles qui repondent, (reference, numero) : l'un d'eux doit etre
    # parmi les extraits lus (controle_du_chat.porte_l_article), avec les mots
    # de sa regle (`regles`).
    articles_attendus: Tuple[Tuple[str, str], ...] = ()
    # Parmi les questions dont le controle compare l'index HNSW a la
    # recherche exacte (rappel@10).
    rappel: bool = False
    # L'article que la question NOMME, (reference, numero) : le premier
    # extrait lu, par la lecture directe. None si elle n'en nomme pas.
    article_nomme: Optional[Tuple[str, str]] = None
    # Les extraits d'autres textes toleres avant l'article qui repond,
    # (reference, numero du morceau) : voir la docstring du module.
    toleres_avant: Tuple[Tuple[str, str], ...] = ()
    # Les mots de la regle demandee, (reference, numero, mots), par article
    # attendu ou vise par un renvoi : l'extrait qui repond doit les porter
    # (controle_du_chat.porte_la_regle). Voir la docstring du module.
    regles: Tuple[Tuple[str, str, str], ...] = ()
    # Les articles auxquels l'article qui repond RENVOIE pour sa peine
    # (« Est puni des peines prévues à l'alinéa 1 ») : l'un d'eux doit etre lu
    # aussi, la reponse n'est complete qu'alors.
    renvoi: Tuple[Tuple[str, str], ...] = ()
    # FAIBLESSE CONNUE de la recuperation : ce que le modele lit a la place de
    # la reponse, tel que mesure (FAIBLESSES_CONNUES). None : question
    # GARANTIE, dont l'echec fait rendre 503 a la route.
    faiblesse: Optional[str] = None

    @property
    def garantie(self) -> bool:
        """Son echec fait rendre 503 a la route ; celui d'une faiblesse connue, jamais."""
        return self.faiblesse is None

    def regle_de(self, reference: str, numero: str) -> Optional[str]:
        """Les mots de la regle que l'article (reference, numero) doit porter, s'il y en a."""
        return next((mots for ref, num, mots in self.regles if (ref, num) == (reference, numero)),
                    None)

    def exigences_manquantes(self, references: Iterable[str]) -> List[Tuple[str, ...]]:
        """Les exigences qu'aucune de ces references ne remplit."""
        presentes = set(references)
        return [groupe for groupe in self.textes if not presentes.intersection(groupe)]

    def attendues(self) -> Tuple[str, ...]:
        """Toutes les references admises, dans l'ordre, sans doublon."""
        return tuple(dict.fromkeys(ref for groupe in self.textes for ref in groupe))


def _art(reference: str, *numeros: str) -> Tuple[Tuple[str, str], ...]:
    """Les articles `numeros` d'un texte : `_art(CODE_PENAL, "318", "321")`."""
    return tuple((reference, numero) for numero in numeros)


# Un article attendu : (reference, numero), ou (reference, numero, mots de la
# regle qu'il doit porter), ecrit avec _regle.
Attendu = Union[Tuple[str, str], Tuple[str, str, str]]


def _regle(reference: str, numero: str, mots: str) -> Tuple[Tuple[str, str, str], ...]:
    """
    L'article `numero` de `reference`, qui ne repond que s'il porte `mots` :
    des mots de la regle demandee, relus dans le morceau de la base.
    """
    return ((reference, numero, mots),)


def _q(
    question: str,
    langue: str,
    *textes: Union[str, Tuple[str, ...]],
    origine: str,
    articles: str,
    repondent: Optional[Tuple[Attendu, ...]] = None,
    renvoi: Tuple[Attendu, ...] = (),
    nomme: Optional[Tuple[str, str]] = None,
    toleres: Tuple[Tuple[str, str], ...] = (),
    rappel: bool = False,
) -> QuestionDeReference:
    """
    `nomme` : l'article que la question nomme, qui est alors celui qui repond.
    `repondent` et `renvoi` : des articles, avec ou sans les mots de leur regle.
    La question est une faiblesse connue si FAIBLESSES_CONNUES la porte, au
    mot pres : une cle mal recopiee laisse la question garantie, donc une
    route rouge, jamais une panne passee sous silence.
    """
    if repondent is None:
        if nomme is None:
            raise ValueError(f"« {question} » : ni article qui repond, ni article nomme")
        repondent = (nomme,)
    return QuestionDeReference(
        question=question,
        langue=langue,
        textes=tuple(t if isinstance(t, tuple) else (t,) for t in textes),
        origine=origine,
        articles=articles,
        articles_attendus=tuple((a[0], a[1]) for a in repondent),
        rappel=rappel,
        article_nomme=nomme,
        toleres_avant=toleres,
        regles=tuple(a for a in (*repondent, *renvoi) if len(a) == 3),
        renvoi=tuple((a[0], a[1]) for a in renvoi),
        faiblesse=FAIBLESSES_CONNUES.get(question),
    )


# ---- Les articles qui repondent, et les mots de leur regle ----
# Chacun relu le 30/10/2026 dans les morceaux de jurix_dev (memes articles que
# la production) : c'est celui qui PORTE la regle demandee, jamais un article
# qui y renvoie seulement. Les mots de `_regle` sont dans ce morceau, au mot
# pres (test d'integration).
#
# L'election du President : la revision constitutionnelle de 2026 (art. 6
# nouveau : sept ans, reeligible) ou le Code electoral (art. 116 : sept ans,
# reeligible, scrutin majoritaire a un tour). La Constitution de 1996
# (PRC-6283, art. 6) dit « renouvelable une fois » : seule, elle ferait une
# reponse perimee.
_ELECTION_DU_PRESIDENT = (REVISION_DE_2026, CODE_ELECTORAL)
_ARTICLES_DE_L_ELECTION = (
    _regle(REVISION_DE_2026, "6", "élu pour un mandat de sept (07) ans. Il est rééligible")
    + _regle(CODE_ELECTORAL, "116", "élu pour un mandat de sept (07) ans au suffrage universel direct")
)
# Le licenciement abusif : l'art. 39, sous le morceau 35.1 (« Toute rupture
# abusive du contrat peut donner lieu à des dommages-intérêts » ; au moins 3
# mois de salaire, au plus 1 mois par annee d'anciennete). Les art. 36
# (preavis non observe) et 37 (indemnite de licenciement), dans le meme
# morceau aujourd'hui, sont dus pour tout licenciement : seuls, apres un
# redecoupage, ils ne diraient rien de l'abus, la ou l'incident a fait
# inventer au modele un « article 76 ».
_INDEMNITES_DU_LICENCIEMENT_ABUSIF = _regle(
    CODE_DU_TRAVAIL, "39", "ne peut être inférieur à trois (3) mois de salaire")
# Le delai de preavis : l'art. 34, dans le morceau 33 (en-tete « Article
# 34.- ») : le contrat a duree indeterminee ne se rompt que moyennant un
# preavis, notifie par ecrit, et « un arrêté du ministre chargé du Travail
# [...] détermine les conditions et la durée du préavis ». Pas l'art. 35
# (morceau 35.1) : les obligations PENDANT le preavis, puis les art. 36 a 39,
# aucune duree. Revue du tour 4 : compte pour l'art. 35, 35.1 gardait verte
# la question francaise, la ou le modele lisait d'abord le preavis d'un AGENT
# COMMERCIAL (AUDCG art. 228 : un, deux, trois mois).
_DUREE_DU_PREAVIS = _regle(
    CODE_DU_TRAVAIL, "34", "détermine les conditions et la durée du préavis")
# La peine du vol, de l'abus de confiance et de l'escroquerie : l'art. 318 (1)
# du Code penal, seul a la chiffrer (cinq a dix ans, 100 000 a 1 000 000 de
# francs). Les art. 320 (vol aggrave) et 321 (abus de confiance et escroquerie
# aggraves) disent seulement « Les peines de l'article 318 ci-dessus sont
# doublées » : jamais attendus. Revue du tour 4 : le vol restait vert par
# l'art. 320, le modele ne lisant la peine de l'art. 318 qu'au rang 7, derriere
# le vol d'effets MILITAIRES (Code de justice militaire, art. 54 et 56), ou
# jamais en anglais ; et, le morceau 318 perdu, l'escroquerie restait verte
# par l'art. 321, sans aucune peine lisible.
_PEINE_DE_L_ARTICLE_318 = _regle(CODE_PENAL, "318", "emprisonnement de cinq (5) à dix (10) ans")
_AMENDE_DE_L_ARTICLE_318 = _regle(
    CODE_PENAL, "318", "amende de cent mille (100 000) à un million (1 000 000) de francs")
_ABUS_DE_CONFIANCE = _regle(CODE_PENAL, "318", "par abus de confiance, c'est-à-dire en détournant")
# Les droits de la personne gardee a vue : art. 122 (informee des faits,
# traitee avec humanite, visite de l'avocat et de la famille) et 123 (examen
# medical), sous le morceau 120.2.
_DROITS_EN_GARDE_A_VUE = (
    _regle(CODE_DE_PROCEDURE_PENALE, "122",
           "doit être immédiatement informé des faits qui lui sont reprochés")
    + _regle(CODE_DE_PROCEDURE_PENALE, "123", "être examinée par un médecin")
)
# L'avocat en garde a vue : l'art. 122 (3), la visite de l'avocat. L'art. 123
# ne nomme l'avocat que pour demander l'examen medical.
_L_AVOCAT_EN_GARDE_A_VUE = _regle(
    CODE_DE_PROCEDURE_PENALE, "122", "recevoir aux heures ouvrables la visite de son avocat")
# La duree de la garde a vue : l'art. 119 (2), 48 heures renouvelable une
# fois, deux fois sur autorisation du procureur. Pas le morceau 120.1, la
# prorogation de 24 heures par 50 km « nonobstant les dispositions de
# l'article 119 alinéa (2) » : il ne dit pas la duree qu'il prolonge (revue du
# tour 4).
_DUREE_DE_LA_GARDE_A_VUE = _regle(
    CODE_DE_PROCEDURE_PENALE, "119", "ne peut excéder quarante huit (48) heures renouvelable une fois")
# L'injonction de payer : art. 2 (creance certaine, liquide et exigible), 3 et
# 4 (requete), 5 (ordonnance). « 1 » n'est pas attendu (voir la docstring).
_INJONCTION_DE_PAYER = (
    _regle(AU_RECOUVREMENT, "2", "créance certaine, liquide et exigible")
    + _regle(AU_RECOUVREMENT, "3", "La demande est formée par requête")
    + _regle(AU_RECOUVREMENT, "4", "La requête doit être déposée")
    + _regle(AU_RECOUVREMENT, "5", "rend l'ordonnance dans les trois jours")
)
# L'opposition : art. 10 (delai) et 11 (formes) ; le 9 n'a ni numero ni en-tete.
_OPPOSITION = (
    _regle(AU_RECOUVREMENT, "10", "L'opposition doit être formée dans les dix jours")
    + _regle(AU_RECOUVREMENT, "11", "de signifier son recours à toutes les parties")
)
# L'embauche : art. 27 (contrat ecrit, visa) et 115 (declaration de la
# main-d'oeuvre). Le 114 (declaration d'ouverture) est dans le meme morceau
# SECTION_8 que le 115, mais son en-tete est illisible (OCR : « Art ss e e e
# e ») : le 115 suffit a reconnaitre ce morceau.
_EMBAUCHE = (
    _regle(CODE_DU_TRAVAIL, "27", "doit être constaté par écrit")
    + _regle(CODE_DU_TRAVAIL, "115", "doit fournir à l'inspection du travail")
)
# L'acces non autorise : l'alinea 2 de l'article 65, le morceau 65.2 (« Est
# puni des peines prévues à l'alinéa 1 ci-dessus, tout accès non autorisé »).
# Il RENVOIE a l'alinea 1, le morceau 65.1 (cinq a dix ans, 5 a 10 millions
# de francs CFA), qui punit l'interception : la peine n'est lisible qu'avec
# les deux (`renvoi`). L'art. 69 n'est pas attendu seul : il punit l'acces EN
# VIOLATION DES MESURES DE SECURITE, d'une amende de 10 a 100 millions, et
# lu seul il ferait annoncer cette amende pour tout acces non autorise ; le
# morceau 65.2 porte aussi l'art. 68 (10 a 50 millions), d'ou le renvoi exige.
# Plus strict que la revue du tour 4 (« la peine en 65.1 ou 69 »), et voulu :
# admettre l'art. 69 seul remettrait au vert la question anglaise (69.1 au
# rang 2), ou la peine de l'acces simple, 65.1, n'est lue qu'au rang 5.
# Un morceau « N.2 » ne compte pas pour N a son numero
# (controle_du_chat.porte_l_article) : 65.2 est attendu par le sien, et 65.1
# l'est comme premier morceau de l'article 65.
_ACCES_NON_AUTORISE = _regle(CYBERCRIMINALITE, "65.2", "tout accès non autorisé")
_PEINES_DE_L_ALINEA_1 = _regle(
    CYBERCRIMINALITE, "65", "amende de 5.000.000 (cinq millions) à 10.000.000 (dix millions)")
# La definition du groupement d'interet economique : l'art. 869. Les art. 870
# a 872 n'en disent que des traits (pas de partage des benefices, qui peut le
# constituer, personnalite morale) : seuls, ils laisseraient la definition a
# la memoire du modele.
_DEFINITION_DU_GIE = _regle(AU_SOCIETES, "869", "a pour but exclusif de mettre en œuvre")
# Les avantages fiscaux de l'ordonnance de 2025 : art. 10 (phase
# d'installation) et 11 (phase d'exploitation, morceau 11.1).
_AVANTAGES_DE_2025 = (
    _regle(INCITATIONS_2025, "10", "pendant la phase d'installation")
    + _regle(INCITATIONS_2025, "11", "pendant la phase d'exploitation")
)
_TENTATIVE_ET_COMPLICITE = (
    _regle(CODE_PENAL, "94", "Toute tentative manifestée par un acte")
    + _regle(CODE_PENAL, "97", "Est complice d'une infraction qualifiée crime ou délit")
)
_LEGITIME_DEFENSE = _regle(
    CODE_PENAL, "84", "nécessité immédiate de la défense de soi-même ou d'autrui")
_CDD_ET_CDI = _regle(
    CODE_DU_TRAVAIL, "25", "dont le terme est fixé à l'avance par la volonté des deux parties")

# ---- Extraits toleres avant l'article qui repond (toleres_avant) ----
# Chacun lu le 29/10/2026, et vu en tete du contexte dans la mesure (base, ou
# 10 tirages de gigue au cosinus 0,9993, data/fiabilite/tmp/controle4/).
#
# L'article 6 de la Constitution de 1996 dit encore le mandat « renouvelable
# une fois ». Le prompt le donne au modele avec sa ligne de statut :
# « Statut : article modifié par la Loi N°2026/002 du 14 avril 2026
# (PRC-10679) : réécrit en entier [...] Pour ce qui a changé, ce texte fait
# foi » (liens_entre_textes). Suivent le Code electoral (art. 116 : « Il est
# rééligible ») et l'article 6 nouveau. Lu en tete pour l'election en anglais,
# en base et dans tous les tirages. Seule, cette Constitution reste une
# exigence manquante (_ELECTION_DU_PRESIDENT) : toleree avant la reponse,
# jamais a sa place. La ligne de statut, telle que le prompt la donne, est la
# condition de la tolerance : tests/test_services/test_controle_du_chat.py
# verifie qu'elle tient. Pour la question FRANCAISE, jamais lue avant la
# reponse (le Code electoral est en tete : 21 contextes au tour 4, base et 10
# tirages au cosinus 0,9985 le 30/10/2026) : pas toleree, une tolerance qu'aucune
# mesure n'a montree n'a pas lieu d'etre.
_CONSTITUTION_DE_1996_ART_6 = ((CONSTITUTION_DE_1996, "6"),)
# Les mentions obligatoires d'un MARCHE PUBLIC (objet, imputation
# budgetaire, parties...), en tete pour le contrat de travail a duree
# determinee en anglais dans 5 tirages sur 10 (6 sur 10 au cosinus 0,9985,
# le 30/10/2026). Un autre « contrat », sans
# rapport avec la duree d'un contrat de travail : l'article 25 du Code du
# travail, qui repond, suit au rang 2.
_MENTIONS_D_UN_MARCHE_PUBLIC = ((MARCHES_DES_ENTREPRISES_PUBLIQUES, "84"),)
# Un recueil de jurisprudence sur l'article 1384 du Code civil
# (responsabilite du fait des choses, des instituteurs), en tete pour l'abus
# de confiance dans 2 tirages sur 10 (3 sur 10 au cosinus 0,9985, le
# 30/10/2026). Il ne definit aucune infraction ;
# l'article 318 du Code penal suit au rang 2.
_JURISPRUDENCE_DE_L_ARTICLE_1384 = ((CODE_CIVIL, "1384.2"),)

# ---- Les faiblesses connues de la recuperation (voir la docstring) ----
# La question, au mot pres, et ce que le modele lit a la place de la reponse.
# Mesure du 30/10/2026 (jurix_dev, vecteurs enregistres, base et 10 tirages
# de gigue au cosinus 0,9985, puis le vrai Voyage ; recuperation du tour 5,
# rag_service 77b552d7, search_service c914d9bb ; remesure sur la
# recuperation finale du tour 6, rag_service d079c404, dans
# data/fiabilite/tmp/controle7/) : les 6 echouent dans tous les tirages, aux
# memes rangs, et les 45 autres questions tiennent. Ce sont les seules : une
# question n'entre ici que si elle echoue aujourd'hui avec ses attendus
# justes, jamais pour faire passer un envoi ; elle en sort des qu'elle tient,
# pour que sa prochaine panne fasse de nouveau rendre 503 a la route. C'est a
# la recuperation de les corriger.
FAIBLESSES_CONNUES: Dict[str, str] = {
    "Quelle peine encourt l’auteur d’un vol ?": (
        "en tete l'art. 320 (« Les peines de l'article 318 ci-dessus sont doublées », sans "
        "peine) ; la peine du vol (art. 318) au rang 7, apres le vol et le recel d'effets "
        "MILITAIRES (PRC-5695 art. 56 et 54, Code de justice militaire) et la surete nucleaire "
        "(PRC-7769 art. 108 : 10 a 20 ans)"),
    "What penalty does the Penal Code provide for theft?": (
        "l'art. 318 jamais lu ; l'art. 320 (peines doublees) au rang 3, et la seule peine de "
        "vol chiffree lue est celle du Code de justice militaire (PRC-5695 art. 54 et 56)"),
    "Quel est le montant de l'amende prévue pour le vol ?": (
        "l'art. 318 jamais lu ; en tete, le vol d'effets MILITAIRES (PRC-5695 art. 54 et 56)"),
    "Quel est le délai de préavis en cas de licenciement ?": (
        "l'art. 34 (morceau 33) au rang 6, apres l'AUPCAP art. 110, le preavis d'un AGENT "
        "COMMERCIAL (AUDCG art. 228 : un, deux, trois mois) et le Code civil art. 107"),
    "How does the law punish unauthorised access to a computer system?": (
        "la peine de l'alinea 1 de l'art. 65 (morceau 65.1) au rang 5, apres 65.2, l'art. 69 "
        "(10 a 100 millions, l'acces EN VIOLATION des mesures de securite) et les art. 86 et 84"),
    "How much compensation is allocated for unfair dismissal?": (
        "le morceau 35.1 (art. 39) jamais lu ; en tete, l'indemnite du FONCTIONNAIRE licencie "
        "(DEC-1994-199 art. 120)"),
}

QUESTIONS_DE_REFERENCE: Tuple[QuestionDeReference, ...] = (
    # ---- Suggestions du front, en francais ----
    _q("Quelles indemnités en cas de licenciement abusif ?", "fr", CODE_DU_TRAVAIL,
       origine=SUGGESTION, rappel=True, repondent=_INDEMNITES_DU_LICENCIEMENT_ABUSIF,
       articles="art. 39 (dommages-intérêts de la rupture abusive : 3 mois de salaire au "
                "moins), avec les art. 36 et 37 : morceau 35.1"),
    _q("Que risque l’auteur d’une escroquerie sur internet ?", "fr", CODE_PENAL,
       origine=SUGGESTION, rappel=True, repondent=_PEINE_DE_L_ARTICLE_318,
       articles="Code pénal art. 318 (1) c (escroquerie, cinq à dix ans) ; 321 (peines "
                "doublées) n'y fait que renvoyer"),
    _q("Quels sont mes droits en garde à vue ?", "fr", CODE_DE_PROCEDURE_PENALE,
       origine=SUGGESTION, repondent=_DROITS_EN_GARDE_A_VUE,
       articles="art. 122 (droits du suspect), 123 (examen médical) : morceau 120.2"),
    _q("Quand peut-on invoquer la légitime défense ?", "fr", CODE_PENAL,
       origine=SUGGESTION, repondent=_LEGITIME_DEFENSE, articles="art. 84"),
    _q("Quelle différence entre tentative et complicité en droit pénal ?", "fr", CODE_PENAL,
       origine=SUGGESTION, repondent=_TENTATIVE_ET_COMPLICITE,
       articles="art. 94 (tentative), 97 (complicité)"),
    _q("Comment est élu le Président de la République ?", "fr", _ELECTION_DU_PRESIDENT,
       origine=SUGGESTION, repondent=_ARTICLES_DE_L_ELECTION,
       articles="révision de 2026, art. 6 (nouveau) ; Code électoral art. 116"),
    _q("Qu’est-ce qui distingue un contrat de travail à durée déterminée d’un contrat "
       "à durée indéterminée ?", "fr", CODE_DU_TRAVAIL,
       origine=SUGGESTION, repondent=_CDD_ET_CDI, articles="art. 25"),
    _q("Qu’est-ce qu’un groupement d’intérêt économique ?", "fr", AU_SOCIETES,
       origine=SUGGESTION, repondent=_DEFINITION_DU_GIE,
       articles="art. 869 (définition) ; 870 à 872 n'en disent que des traits"),
    _q("Quels sont les éléments constitutifs de l’abus de confiance ?", "fr", CODE_PENAL,
       origine=SUGGESTION, repondent=_ABUS_DE_CONFIANCE,
       toleres=_JURISPRUDENCE_DE_L_ARTICLE_1384, articles="art. 318 (1) b"),
    _q("Quelles peines pour l’accès non autorisé à un système d’information ?", "fr",
       CYBERCRIMINALITE, origine=SUGGESTION, repondent=_ACCES_NON_AUTORISE,
       renvoi=_PEINES_DE_L_ALINEA_1,
       articles="art. 65 al. 2 (morceau 65.2), qui renvoie aux peines de l'al. 1 "
                "(morceau 65.1)"),
    _q("Comment former opposition à une injonction de payer en droit OHADA ?", "fr",
       AU_RECOUVREMENT, origine=SUGGESTION, repondent=_OPPOSITION,
       articles="art. 9 à 11 (opposition) ; 9 sans numéro en base"),
    _q("Quelle est la durée maximale de la garde à vue ?", "fr", CODE_DE_PROCEDURE_PENALE,
       origine=SUGGESTION, repondent=_DUREE_DE_LA_GARDE_A_VUE,
       articles="art. 119 (2) (48 heures renouvelable une fois) ; 120 (1) ne fait que "
                "le proroger selon la distance"),
    _q("Quels avantages fiscaux l’ordonnance de 2025 accorde-t-elle à un nouvel "
       "investissement ?", "fr", INCITATIONS_2025,
       origine=SUGGESTION, repondent=_AVANTAGES_DE_2025, articles="art. 10 et 11"),
    _q("Quelles pièces justificatives pour immatriculer une société commerciale ?", "fr",
       AU_DROIT_COMMERCIAL, origine=SUGGESTION,
       repondent=_regle(AU_DROIT_COMMERCIAL, "47", "sont jointes les pièces justificatives suivantes"),
       articles="art. 47 (pièces), RCCM ; l'art. 46 ne dit que les mentions de la demande"),
    _q("Quelles obligations pour l’employeur lors d’une embauche ?", "fr", CODE_DU_TRAVAIL,
       origine=SUGGESTION, repondent=_EMBAUCHE,
       articles="art. 27 (contrat écrit, visa), 114 et 115 (déclarations) : morceaux "
                "26.2 et SECTION_8"),
    _q("Comment obtenir une injonction de payer pour une facture impayée ?", "fr",
       AU_RECOUVREMENT, origine=SUGGESTION, repondent=_INJONCTION_DE_PAYER,
       articles="art. 2 à 5 (conditions, requête, ordonnance)"),
    # ---- Suggestions du front, en anglais (corpus en francais) ----
    _q("What damages does the Labour Code provide for unfair dismissal?", "en",
       CODE_DU_TRAVAIL, origine=SUGGESTION, repondent=_INDEMNITES_DU_LICENCIEMENT_ABUSIF,
       articles="art. 39 (morceau 35.1), au rang 1 ; reformulée le 29/10/2026"),
    _q("How does the Penal Code punish deceiving someone to obtain their money?", "en",
       CODE_PENAL, origine=SUGGESTION, repondent=_PEINE_DE_L_ARTICLE_318,
       articles="Code pénal art. 318 (1) c ; reformulée le 29/10/2026 (art. 318 au "
                "rang 1, similarité 0,450 contre 0,392 au suivant)"),
    _q("What are my rights in police custody?", "en", CODE_DE_PROCEDURE_PENALE,
       origine=SUGGESTION, repondent=_DROITS_EN_GARDE_A_VUE,
       articles="art. 122, 123 : morceau 120.2"),
    _q("When can self-defence be invoked?", "en", CODE_PENAL,
       origine=SUGGESTION, repondent=_LEGITIME_DEFENSE, articles="art. 84"),
    _q("What is the difference between attempt and complicity in criminal law?", "en",
       CODE_PENAL, origine=SUGGESTION, repondent=_TENTATIVE_ET_COMPLICITE,
       articles="art. 94, 97"),
    _q("How is the President of the Republic elected?", "en", _ELECTION_DU_PRESIDENT,
       origine=SUGGESTION, repondent=_ARTICLES_DE_L_ELECTION,
       toleres=_CONSTITUTION_DE_1996_ART_6,
       articles="révision de 2026, art. 6 (nouveau) ; Code électoral art. 116 ; la "
                "Constitution de 1996 (art. 6) lue d'abord, avec sa ligne de statut"),
    _q("What distinguishes a fixed-term employment contract from an open-ended one?", "en",
       CODE_DU_TRAVAIL, origine=SUGGESTION, repondent=_CDD_ET_CDI,
       toleres=_MENTIONS_D_UN_MARCHE_PUBLIC, articles="art. 25"),
    _q("What is an economic interest group under OHADA law?", "en", AU_SOCIETES,
       origine=SUGGESTION, repondent=_DEFINITION_DU_GIE, articles="art. 869"),
    _q("What are the constituent elements of the offence of breach of trust under the "
       "Penal Code?", "en", CODE_PENAL, origine=SUGGESTION,
       repondent=_ABUS_DE_CONFIANCE, articles="art. 318 (1) b"),
    _q("How does the law punish unauthorised access to a computer system?", "en",
       CYBERCRIMINALITE, origine=SUGGESTION, repondent=_ACCES_NON_AUTORISE,
       renvoi=_PEINES_DE_L_ALINEA_1,
       articles="art. 65 al. 2 (morceau 65.2), avec les peines de l'al. 1 (morceau 65.1)"),
    _q("How is an OHADA injunction to pay obtained and opposed?", "en", AU_RECOUVREMENT,
       origine=SUGGESTION, rappel=True, repondent=_INJONCTION_DE_PAYER + _OPPOSITION,
       articles="art. 2 à 5, 10 et 11"),
    _q("What is the maximum length of police custody?", "en", CODE_DE_PROCEDURE_PENALE,
       origine=SUGGESTION, repondent=_DUREE_DE_LA_GARDE_A_VUE,
       articles="art. 119 (2) ; 120 (1) ne fait que le proroger"),
    _q("What tax incentives does the 2025 investment incentives ordinance grant to a new "
       "investment?", "en", INCITATIONS_2025, origine=SUGGESTION, rappel=True,
       repondent=_AVANTAGES_DE_2025, articles="art. 10 et 11"),
    _q("From when does a company have legal personality under OHADA law?", "en",
       AU_SOCIETES, origine=SUGGESTION,
       repondent=_regle(AU_SOCIETES, "98",
                        "jouit de la personnalité juridique à compter de son immatriculation"),
       articles="art. 98 (personnalité juridique dès l'immatriculation au RCCM) ; l'art. "
                "97 ne dit que l'obligation de s'immatriculer ; remplace le 29/10/2026 la "
                "question sur l'immatriculation (art. 46-47 illisibles en anglais)"),
    _q("What are the employer's obligations when hiring under the Labour Code?", "en",
       CODE_DU_TRAVAIL, origine=SUGGESTION, repondent=_EMBAUCHE,
       articles="art. 27, 114 et 115 : morceaux 26.2 et SECTION_8"),
    _q("How can a business recover an unpaid invoice from a customer under OHADA law?", "en",
       AU_RECOUVREMENT, origine=SUGGESTION, repondent=_INJONCTION_DE_PAYER,
       articles="art. 2 à 5 (injonction de payer), art. 2 au rang 1 ; reformulée le "
                "29/10/2026"),
    # ---- Questions de fond ----
    _q("Quel est le délai de préavis en cas de licenciement ?", "fr", CODE_DU_TRAVAIL,
       origine=FOND, repondent=_DUREE_DU_PREAVIS,
       articles="art. 34 (préavis, sa durée fixée par arrêté) : morceau 33 ; pas l'art. 35 "
                "(obligations pendant le préavis)"),
    _q("Quelles sont les conditions d’un licenciement pour motif économique ?", "fr",
       CODE_DU_TRAVAIL, origine=FOND,
       repondent=_regle(CODE_DU_TRAVAIL, "40", "motifs non inhérents à la personne du travailleur"),
       articles="art. 40 : morceau 35.2"),
    _q("Quelle peine encourt l’auteur d’un vol ?", "fr", CODE_PENAL,
       origine=FOND, repondent=_PEINE_DE_L_ARTICLE_318,
       articles="art. 318 (1) a (vol, cinq à dix ans) ; 320 (vol aggravé) n'y fait que "
                "renvoyer"),
    _q("Quelle est la peine prévue pour l’escroquerie ?", "fr", CODE_PENAL,
       origine=FOND, repondent=_PEINE_DE_L_ARTICLE_318, articles="art. 318 (1) c ; 321 renvoie"),
    _q("Une personne gardée à vue peut-elle se faire assister d’un avocat ?", "fr",
       CODE_DE_PROCEDURE_PENALE, origine=FOND, repondent=_L_AVOCAT_EN_GARDE_A_VUE,
       articles="art. 122 al. 3 (visite de l'avocat) : morceau 120.2"),
    _q("Quelles sont les causes de divorce ?", "fr", CODE_CIVIL, origine=FOND,
       repondent=(
           _regle(CODE_CIVIL, "229", "pour cause d'adultère de sa femme")
           + _regle(CODE_CIVIL, "230", "pour cause d'adultère de son mari")
           + _regle(CODE_CIVIL, "231", "peine afflictive")
           + _regle(CODE_CIVIL, "232", "pour excès, sévices ou injures")
       ),
       articles="art. 229 à 232"),
    _q("Comment obtenir un titre foncier ?", "fr", TITRE_FONCIER, origine=FOND,
       repondent=(
           _regle(TITRE_FONCIER, "11", "solliciter l'obtention d'un titre foncier")
           + _regle(TITRE_FONCIER, "12", "Le dossier est déposé à la sous-préfecture")
           + _regle(TITRE_FONCIER, "13", "le préfet informe le public")
           + _regle(TITRE_FONCIER, "14", "un plan et un procès-verbal sont dressés")
           + _regle(TITRE_FONCIER, "15", "avis de clôture de bornage")
       ),
       articles="art. 11 et suivants (procédure). Récupération seule : rédaction de "
                "1976, sans le décret n°2005/481 qui réécrit la procédure"),
    _q("Quel est le capital social minimum d’une SARL ?", "fr", AU_SOCIETES, origine=FOND,
       repondent=_regle(AU_SOCIETES, "311", "d'un million (1.000.000) de francs CFA au moins"),
       articles="art. 311 (et non 311-1, rangé sous le même numéro). Récupération seule : "
                "« sauf dispositions nationales contraires », et la loi n°2016/014 qui y "
                "déroge manque au corpus"),
    _q("Quel est le taux de la TVA ?", "fr", LOI_DE_FINANCES_2026, origine=FOND,
       repondent=_regle(LOI_DE_FINANCES_2026, "142", "17,5 %"),
       articles="art. 142 du CGI réécrit (17,5 %, 10 %, 0 %) : seule loi de finances "
                "du corpus dont l'article 142 donne les taux"),
    _q("Quelle est la peine pour la corruption d’un fonctionnaire ?", "fr", CODE_PENAL,
       origine=FOND, rappel=True,
       repondent=_regle(CODE_PENAL, "134",
                        "deux cent mille (200 000) à deux millions (2 000 000) de francs"),
       articles="art. 134 (cinq à dix ans) ; 134-1 et 134-2, rangés sous le même numéro, "
                "n'en portent pas la peine"),
    _q("What is the notice period for dismissing an employee?", "en", CODE_DU_TRAVAIL,
       origine=FOND, repondent=_DUREE_DU_PREAVIS, articles="art. 34 : morceau 33"),
    _q("What penalty does the Penal Code provide for theft?", "en", CODE_PENAL,
       origine=FOND, repondent=_PEINE_DE_L_ARTICLE_318, articles="art. 318 (1) a"),
    # Deux questions de la revue du tour 4 (29/10/2026, vrai Voyage), posees
    # comme un utilisateur les tape, sur les deux sujets de l'incident et de
    # la revue : la recuperation y faisait lire d'abord le Code de justice
    # militaire sans l'art. 318, et l'indemnite du FONCTIONNAIRE licencie
    # (DEC-1994-199 art. 120) sans le morceau 35.1. Les reformulations qui
    # tiennent (suggestions) ne disent pas que la recuperation tient.
    _q("Quel est le montant de l'amende prévue pour le vol ?", "fr", CODE_PENAL,
       origine=FOND, repondent=_AMENDE_DE_L_ARTICLE_318,
       articles="art. 318 (1) (100 000 à 1 000 000 de francs)"),
    _q("How much compensation is allocated for unfair dismissal?", "en", CODE_DU_TRAVAIL,
       origine=FOND, repondent=_INDEMNITES_DU_LICENCIEMENT_ABUSIF,
       articles="art. 39 (morceau 35.1)"),
    # ---- Questions qui nomment un article : lu EN PREMIER (article_nomme) ----
    _q("Que dit l'article 39 du Code du travail ?", "fr", CODE_DU_TRAVAIL,
       origine=FOND, nomme=(CODE_DU_TRAVAIL, "39"),
       articles="art. 39 (dommages-intérêts) : en-tête dans le morceau 35.1"),
    # L'article de l'incident : le modele l'avait invente pour les indemnites,
    # et le lecteur de citations l'avait relie a l'article 76.37 d'une loi de
    # finances. Le vrai porte sur les retenues sur salaire.
    _q("Que dit l'article 76 du Code du travail ?", "fr", CODE_DU_TRAVAIL,
       origine=FOND, nomme=(CODE_DU_TRAVAIL, "76"),
       articles="art. 76 (retenues sur salaire) : en-tête dans le morceau SECTION_4.1"),
    _q("Que dit l'article 122 du Code de procédure pénale ?", "fr", CODE_DE_PROCEDURE_PENALE,
       origine=FOND, nomme=(CODE_DE_PROCEDURE_PENALE, "122"),
       articles="art. 122 (droits du gardé à vue) : en-tête dans le morceau 120.2"),
    _q("What does Article 318 of the Penal Code say?", "en", CODE_PENAL,
       origine=FOND, nomme=(CODE_PENAL, "318"), articles="art. 318 (morceau 318)"),
    _q("What does Article 37 of the Labour Code say?", "en", CODE_DU_TRAVAIL,
       origine=FOND, nomme=(CODE_DU_TRAVAIL, "37"),
       articles="art. 37 (indemnité de licenciement) : en-tête dans le morceau 35.1"),
)
