"""
Ce que l'intitule officiel d'un texte dit de lui : son numero, sa date et
son objet.

POURQUOI. Les listes (domaines, recherche) affichaient l'intitule entier :
« Décret N°2018/763 du 10 décembre 2018 portant révocation d'un magistrat ».
Coupe a deux lignes sur un telephone, il perdait justement l'objet, et toutes
les fiches se ressemblaient. Les listes affichent desormais une ligne de
reperes (nature, numero, date) puis l'objet en clair : « Révocation d'un
magistrat ». L'intitule complet reste sur la page du texte.

PRUDENCE. Une decoupe ratee deforme ce qu'un texte dit faire, ce qui est pire
qu'un intitule long. `objet_du_titre` rend donc None des qu'elle doute :
en-tete qui contient autre chose qu'une nature, un numero et une date
(« Code des assurances … et Traité instituant … »), verbe qu'elle ne sait pas
nominaliser (« déclarant fériée … »), intitule tire d'un nom de fichier. Le
client affiche alors l'intitule entier. Les regles ont ete eprouvees sur les
2 287 intitules de la base de developpement (voir les tests).

Fonctions pures, sans base ni reseau : appelees pour chaque ligne d'une liste.
"""

import re
import unicodedata
from datetime import date
from typing import Optional

# ==================== NORMALISATION ====================

# Fautes de saisie relevees dans le corpus : « port an t », « p9rtant », et
# l'annee collee au verbe (« 06 septembre 2017portant »).
_PORTANT_ABIME = re.compile(r"\bp[o9]rt\s?an\s?t\b", re.IGNORECASE)
_VERBE_COLLE = re.compile(
    r"(?<=\d)(portant|fixant|modifiant|autorisant|habilitant|ratifiant|relati(?:f|ve))\b",
    re.IGNORECASE,
)

# « decret_n_2014_543_du_10.12.2014_nomination_dg_apme » : un nom de fichier
# pris pour intitule. Un souligne isole (« d'installation_ du personnel ») est
# une faute de frappe, retiree par _nettoyer.
_NOM_DE_FICHIER = re.compile(r"_\w*_")
# « de I'Accord », « I ‘Accord » : un l lu comme un I par l'OCR.
_L_LU_COMME_I = re.compile(r"(?<![\w'’])[Ii]\s?['’‘]\s?(?=\w)")


def _nettoyer(titre: str) -> str:
    """Espaces normalises, trait d'union conditionnel retire, fautes connues reparees."""
    t = titre.replace("\N{SOFT HYPHEN}", "").replace("\N{NO-BREAK SPACE}", " ").replace("_", " ")
    # « ratification du · Protocole » : un point median egare.
    t = re.sub(r"\s+", " ", t.replace(" · ", " ")).strip()
    t = _PORTANT_ABIME.sub("portant", t)
    t = re.sub(r"\b(portant)\s+portant\b", r"\1", t, flags=re.IGNORECASE)
    # « N°202l/345 » : un 1 lu comme un l dans l'annee du numero.
    t = re.sub(r"\b(\d{3})l(?=\s*/)", r"\g<1>1", t)
    t = _L_LU_COMME_I.sub("l'", t)
    return _VERBE_COLLE.sub(r" \1", t)


def _replier(texte: str) -> str:
    """Minuscules sans accents : « Arrêté » et « ARRETE » se comparent."""
    decompose = unicodedata.normalize("NFKD", texte.lower())
    return "".join(c for c in decompose if not unicodedata.combining(c))


# ==================== EN-TETE : NATURE, NUMERO, DATE ====================

# Mots qu'un en-tete peut porter avant l'objet, replies. Tout autre mot dans
# l'en-tete fait douter : « Code des assurances des États membres de la CIMA,
# édition 2019, et Traité instituant … » n'a pas « instituant » pour verbe
# principal.
_NATURES = {
    "decret", "arrete", "loi", "cadre", "ordonnance", "circulaire", "decision",
    "acte", "uniforme", "revise", "traite", "pleins", "pouvoirs",
}
_MOIS = {
    "janvier", "janv", "jan", "fevrier", "fevr", "fev", "mars", "mar", "avril", "avr",
    "mai", "juin", "juillet", "juil", "aout", "septembre", "sept", "sep", "octobre",
    "oct", "novembre", "nov", "decembre", "dec",
}
# Les quatre premieres lettres suffisent, comme a l'ingestion : « juilet »,
# « SEP. ».
_MOIS_4 = {"janv", "fevr", "mars", "avri", "juin", "juil", "aout", "sept", "octo", "nove", "dece"}
_MOTS_DE_DATE = {"du", "of", "en", "date", "le", "er", "n", "no", "numero"}
# Sigles ecrits en minuscules (« N°0591/cab/pr ») ; en capitales, tout sigle
# court passe (voir _SIGLE).
_SIGLES = {"cab", "pr", "pm", "prc", "sg", "pn", "rdpc", "df", "lf"}

# Sigles d'un numero d'acte : « N°0342/CAB/PR », « N°002/RDPC/PN », « N°69/DF/228 ».
_SIGLE = re.compile(r"^[A-Z]{1,6}$")


def _en_tete_sans_doute(en_tete: str) -> bool:
    """
    Vrai si l'en-tete ne contient qu'une nature, un numero et une date.

    Les jetons sont lus sur le texte d'origine (pour reconnaitre un sigle en
    capitales) puis replies. Le premier doit etre une nature : un intitule qui
    commence par « Accord entre … » ou « Code civil … » est un nom, pas un
    acte suivi de son objet.
    """
    jetons = re.findall(r"[^\W\d_]+|\d+", en_tete)
    if not jetons or _replier(jetons[0]) not in _NATURES:
        return False
    for jeton in jetons:
        if jeton.isdigit() or _SIGLE.match(jeton):
            continue
        replie = _replier(jeton)
        mois = replie in _MOIS or (len(replie) <= 9 and replie[:4] in _MOIS_4)
        if not (mois or replie in _NATURES or replie in _MOTS_DE_DATE or replie in _SIGLES):
            return False
    return True


# « N°2018/763 », « N° 0342/CAB/PR », « No. 2020/038 », « N1078/CAB/PR »,
# « N°67-LF-7 », et sans « N° » : « Décret 2017/209 du … ».
_NUMERO = re.compile(
    r"^[^\W\d_]+(?:[\s-]+(?:uniforme|cadre|pouvoirs|r[ée]vis[ée]))*\s*"
    r"(?:n\s*[°ºo]?\s*\.?\s*)?"
    r"(?P<numero>\d[\w.-]*(?:\s*/\s*[\w.-]+)*)",
    re.IGNORECASE,
)


def numero_du_titre(titre: Optional[str]) -> Optional[str]:
    """
    Le numero de l'acte tel que l'intitule l'ecrit : « 2018/763 »,
    « 0342/CAB/PR ». None si l'intitule n'en porte pas en tete (codes,
    traites, accords, actes uniformes) ou s'il vient d'un nom de fichier.

    Espaces autour des barres retires (« 2015/ 437 », « 0292 /CAB/PR »), et la
    barre qui colle parfois le « du » au numero aussi (« 0095/CAB/PR/du »).
    """
    if not titre or _NOM_DE_FICHIER.search(titre):
        return None
    t = _nettoyer(titre)
    trouve = _NUMERO.match(t)
    premier = re.match(r"[^\W\d_]+", t)
    if not trouve or not premier or _replier(premier.group()) not in _NATURES:
        return None
    numero = re.sub(r"\s*/\s*", "/", trouve.group("numero"))
    numero = re.sub(r"/(?:du|of)$", "", numero, flags=re.IGNORECASE)
    numero = numero.rstrip("/.-")
    # Un numero est fait de chiffres d'abord : « 2018 » seul, suivi d'un mois,
    # n'en est pas un (« Décret N°2024/193 juin 10 » donne « 2024/193 »).
    return numero or None


def date_du_texte(publication_date: Optional[date], titre: Optional[str]) -> Optional[date]:
    """
    La date de l'acte, telle que son intitule la donne.

    `publication_date` est lue dans l'intitule a l'ingestion
    (scripts/ingest_corpus._infer_date) ; faute de jour et de mois, elle y vaut
    le 1er janvier de l'annee du numero. Cette date de remplissage n'est PAS
    une date juridique : None, et seule l'annee de `publication_date` est sure.
    Meme regle que prompts._date_du_texte, pour que le chat et les listes
    disent la meme chose.
    """
    if publication_date is None:
        return None
    if (publication_date.day, publication_date.month) == (1, 1) and "janv" not in _replier(
        titre or ""
    ):
        return None
    return publication_date


# ==================== OBJET ====================

# Verbe ou locution qui ouvre l'objet, et ce qu'on en fait :
# - None : l'objet est ce qui suit, tel quel (« portant nomination de … ») ;
# - "" : ce qui suit, sans son article (« fixant les modalités … » donne
#   « Modalités … ») ;
# - un nom : nominalisation, avec « de » contracte (« modifiant le décret … »
#   donne « Modification du décret … »).
# Les plus longues d'abord : « modifiant et complétant » avant « modifiant ».
_OUVERTURES = (
    (r"portant\s+sur", ""),
    (r"portant", None),
    (r"modifiant\s+et\s+compl[ée]tant", "Modification"),
    (r"modifiant\s+et\s+abrogeant", "Modification et abrogation"),
    (r"modifiant", "Modification"),
    (r"compl[ée]tant", "Complément"),
    (r"abrogeant", "Abrogation"),
    (r"ratifiant", "Ratification"),
    (r"approuvant\s+et\s+rendant\s+ex[ée]cutoire", "Approbation"),
    (r"approuvant", "Approbation"),
    (r"transformant", "Transformation"),
    (r"cr[ée]ant", "Création"),
    (r"organisant", "Organisation"),
    (r"prorogeant", "Prorogation"),
    (r"r[ée]glementant", "Réglementation"),
    (r"convoquant", "Convocation"),
    (r"nommant", "Nomination"),
    (r"fixant", ""),
    (r"d[ée]terminant", ""),
    (r"pr[ée]cisant", ""),
    (r"d[ée]finissant", ""),
    (r"r[ée]gissant", ""),
    (r"instituant", ""),
    (r"ordonnant", ""),
    (r"relati(?:f|ve)\s+(?:à|a|au|aux)\b", "RELATIF"),
    (r"d['’]orientation", "Orientation"),
    (r"autorisant", "AUTORISATION"),
    (r"habilitant", "HABILITATION"),
)
_OUVERTURE = re.compile(
    r"(?<![\w'’])(?:" + "|".join(f"({motif})" for motif, _ in _OUVERTURES) + r")(?![\w'’])",
    re.IGNORECASE,
)

# Noms qui ouvrent l'objet quand le verbe manque (« Décret N°2019/233 du 08 mai
# 2019 nomination d'un Inspecteur Général … ») : ceux qui suivent le plus
# souvent « portant » dans le corpus.
_NOMS_D_OBJET = re.compile(
    r"(?:nomination|promotion|inscription|avancement|organisation|r[ée]organisation"
    r"|attribution|int[ée]gration|cr[ée]ation|approbation|titularisation|adh[ée]sion"
    r"|admission|agr[ée]ment|convocation|r[ée]vocation|rappel|classement"
    r"|renouvellement|ratification)\b",
    re.IGNORECASE,
)

_ARTICLE = re.compile(r"^(?:les|le|la|l['’]|des|du|une|un)\s*(?=\w)", re.IGNORECASE)

# Apres « habilitant <qui> », l'infinitif qui dit ce qui est permis.
_HABILITE_A = re.compile(
    r"^(?P<qui>.{0,160}?)[\s,]*\bà\s+(?P<inf>signer|recourir|conclure|n[ée]gocier"
    r"|contracter|emprunter|[ée]mettre|ratifier)\b[\s,]*",
    re.IGNORECASE,
)
# « autorisant le Président de la République à ratifier … », « le PR à … ».
_AUTORISE_LE_PR = re.compile(
    r"^(?:le\s+)?(?:pr[ée]sident\s+de\s+la\s+r[ée]publique|pr)\s*,?\s+à\s+(?P<inf>\w+)\s*",
    re.IGNORECASE,
)

_MOIS_RE = (
    r"(?:janv(?:ier)?|f[ée]v(?:r(?:ier)?)?|mars|avr(?:il)?|mai|juin|juil(?:l(?:et)?)?"
    r"|ao[uû]t|sept(?:embre)?|oct(?:obre)?|nov(?:embre)?|d[ée]c(?:embre)?)\.?"
)
# Date finale d'un acte uniforme ou d'un traite, dont l'intitule ne porte pas
# la date en tete : « Acte uniforme relatif à la médiation du 23 novembre 2017 ».
_DATE_FINALE = re.compile(
    rf",?\s+du\s+\d{{1,2}}(?:er)?\s+{_MOIS_RE}\s+\d{{4}}\.?$", re.IGNORECASE
)
_PUBLIE_AU_JO = re.compile(r",?\s*publié\s+au\s+journal\s+officiel\b.*$", re.IGNORECASE)
_VOYELLE = re.compile(r"^[aeiouyàâäéèêëîïôöùûüœæ]", re.IGNORECASE)
# Ce qu'un decret habilite directement, quand le titre omet le signataire.
_ACTE_HABILITE = re.compile(
    r"^(?:les?|la|l['’]|une?|des)\s+(?:accords?|pr[êe]ts?|contrats?|financement|[ée]missions?"
    r"|conventions?|cr[ée]dits?)\b",
    re.IGNORECASE,
)
# Un objet ne commence pas par une preposition : « portant de la médaille de
# la Vaillance … » a perdu son nom (« attribution »), et la decoupe le dirait
# mal.
_DEBUT_DOUTEUX = re.compile(
    r"^(?:de|du|d['’]|à|au|aux|en|pour|par|sur|et|ou|avec|dans|entre|\.\.\.|…)(?![\w])",
    re.IGNORECASE,
)
# Articles d'une enumeration de tete : « les attributions, l'organisation et
# le fonctionnement du Conseil … ». Le premier est retire avec le verbe ; les
# suivants, laisses, donnaient « Attributions, l'organisation et le
# fonctionnement du Conseil … ».
_ARTICLE_ENUMERE = re.compile(r"(,\s*|\set\s+)(?:les|le|la|l['’])\s*(?=\w)", re.IGNORECASE)
_PREMIER_COMPLEMENT = re.compile(r"\s(?:du|des|de)\s|\sd['’]", re.IGNORECASE)


# « relative au don, au prélèvement et à la greffe de … » : chaque terme de
# l'enumeration repete la preposition que le verbe a deja emportee.
_PREPOSITION_ENUMEREE = re.compile(
    r"(,\s*|\set\s+)(?:à\s+la|à\s+l['’]|aux|au|à)\s*(?=\w)", re.IGNORECASE
)
# Un second verbe coordonne au premier (« précisant certaines attributions et
# fixant les avantages … ») : l'objet en garderait la moitie conjuguee. Pas
# « modifiant et complétant », locution traitee comme un tout.
_SECOND_VERBE = re.compile(
    r"(?:,|(?<!modifiant)\set)\s+(?:portant|fixant|modifiant|compl[ée]tant|abrogeant"
    r"|ratifiant|approuvant|pr[ée]cisant|d[ée]terminant|d[ée]finissant|r[ée]gissant"
    r"|instituant|autorisant|habilitant)\b",
    re.IGNORECASE,
)


def _sans_prepositions_d_enumeration(texte: str) -> str:
    """« don, au prélèvement et à la greffe de … » → « don, prélèvement et greffe de … »."""
    complement = _PREMIER_COMPLEMENT.search(texte)
    fin = complement.start() if complement else len(texte)
    return _PREPOSITION_ENUMEREE.sub(r"\1", texte[:fin]) + texte[fin:]


def _sans_articles_d_enumeration(texte: str) -> str:
    """« attributions, l'organisation et le fonctionnement du … » → « attributions, organisation et fonctionnement du … »."""
    complement = _PREMIER_COMPLEMENT.search(texte)
    fin = complement.start() if complement else len(texte)
    return _ARTICLE_ENUMERE.sub(r"\1", texte[:fin]) + texte[fin:]


def _sans_article(texte: str) -> str:
    return _ARTICLE.sub("", texte, count=1)


def _de(texte: str) -> str:
    """
    « de » devant un groupe nominal, contracte : de le → du, de les → des,
    de un → d'un, de des → de, de l' → de l', de + voyelle → d'.
    """
    article = re.match(r"^(les|le|la|l['’]|des|une|un)\b\s*", texte, re.IGNORECASE)
    if article:
        mot = article.group(1).lower()
        reste = texte[article.end():]
        if mot == "le":
            return "du " + reste
        if mot == "les":
            return "des " + reste
        if mot in ("un", "une"):
            return f"d'{mot} " + reste
        if mot == "des":
            return ("d'" if _VOYELLE.match(reste) else "de ") + reste
        return "de " + texte
    return ("d'" if _VOYELLE.match(texte) else "de ") + texte


def _a(texte: str) -> str:
    """« à » devant un groupe nominal, contracte : à le → au, à les → aux."""
    article = re.match(r"^(les|le)\b\s*", texte, re.IGNORECASE)
    if article:
        reste = texte[article.end():]
        return ("au " if article.group(1).lower() == "le" else "aux ") + reste
    return "à " + texte


def _nominaliser(nom: str, reste: str) -> Optional[str]:
    if nom == "AUTORISATION":
        pr = _AUTORISE_LE_PR.match(reste)
        if pr:
            # « Autorisation de ratifier la Convention … » : l'objet de la loi
            # est l'autorisation, pas la ratification elle-meme.
            infinitif = pr.group("inf")
            suite = reste[pr.end():]
            lien = "d'" if _VOYELLE.match(infinitif) else "de "
            return f"Autorisation {lien}{infinitif} {suite}"
        return "Autorisation " + _de(reste)
    if nom == "HABILITATION":
        habilite = _HABILITE_A.match(reste)
        if habilite:
            # Qui est habilite (le ministre, cite avec tout son titre) n'est
            # pas l'objet : ce qu'il peut signer, si.
            return f"Habilitation à {habilite.group('inf')} {reste[habilite.end():]}"
        # « habilitant un Accord de Prêt pour … » : l'acte vise est l'objet.
        # Mais « habilitant le Ministre … » sans infinitif reconnu nomme une
        # personne, pas un objet : on doute.
        if _ACTE_HABILITE.match(reste):
            return _sans_article(reste)
        return None
    if nom == "Complément":
        return "Complément " + _a(reste)
    if nom == "Orientation":
        return "Orientation " + reste
    return f"{nom} " + _de(reste)


def objet_du_titre(titre: Optional[str]) -> Optional[str]:
    """
    L'objet d'un texte, tire de son intitule officiel : « Décret N°2018/763 du
    10 décembre 2018 portant révocation d'un magistrat » donne « Révocation
    d'un magistrat ». None quand la decoupe n'est pas sure : le client affiche
    alors l'intitule entier.

    L'objet commence au premier verbe qui suit l'en-tete (nature, numero,
    date) : « portant », « fixant », « modifiant », « autorisant »,
    « relatif à »… Ce verbe doit suivre l'en-tete et rien d'autre : un verbe
    plus loin appartient a un acte cite, ou a un intitule qui n'est pas de
    cette forme. Les verbes sont nominalises (« modifiant le décret … » donne
    « Modification du décret … ») ou retires avec leur article
    (« fixant les modalités … » donne « Modalités … »). La premiere lettre est
    mise en capitale ; le reste garde la casse de l'intitule, noms propres
    compris.
    """
    if not titre or _NOM_DE_FICHIER.search(titre):
        return None
    t = _nettoyer(titre)

    ouverture = _OUVERTURE.search(t)
    if ouverture:
        en_tete, reste = t[:ouverture.start()], t[ouverture.end():].strip()
        motif = next(i for i, groupe in enumerate(ouverture.groups()) if groupe is not None)
        traitement = _OUVERTURES[motif][1]
    else:
        # Verbe absent : un nom d'objet frequent peut ouvrir l'objet.
        en_tete_seul = re.match(r"^(.*?)(?=\b" + _NOMS_D_OBJET.pattern + ")", t, re.IGNORECASE)
        if not en_tete_seul:
            return None
        en_tete, reste, traitement = en_tete_seul.group(1), t[en_tete_seul.end():], None

    en_tete = en_tete.strip(" ,")
    if not _en_tete_sans_doute(en_tete):
        return None
    reste = reste.lstrip(" ,:;-–—")
    if len(reste) < 3:
        return None

    if traitement is None:
        objet = reste
    elif traitement == "":
        objet = _sans_articles_d_enumeration(_sans_article(reste))
    elif traitement == "RELATIF":
        objet = _sans_prepositions_d_enumeration(_sans_article(reste))
    else:
        objet = _nominaliser(traitement, reste)
        if objet is None:
            return None

    objet = _PUBLIE_AU_JO.sub("", objet)
    objet = re.sub(r",\s*révisé$", "", objet, flags=re.IGNORECASE)
    # La date finale n'est retiree que pour un acte uniforme ou un traite, dont
    # l'intitule la porte a la fin faute de la porter en tete. Ailleurs,
    # « … du décret n° 2013/092 du 03 avril 2013 » est la date d'un AUTRE acte,
    # qui fait partie de l'objet.
    if _replier(en_tete).startswith(("acte", "traite")):
        objet = _DATE_FINALE.sub("", objet)
    objet = re.sub(r"[\s.;,:\-–—]+$", "", objet).strip()
    if len(objet) < 3 or _DEBUT_DOUTEUX.match(objet) or _SECOND_VERBE.search(objet):
        return None
    return objet[0].upper() + objet[1:]
