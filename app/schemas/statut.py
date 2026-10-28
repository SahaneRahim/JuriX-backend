"""
Schemas du statut juridique d'un texte et de chacun de ses articles.

Tout y est ADDITIF : GET /laws/{id} gagne un bloc facultatif. Le front deploye
ignore ce qu'il ne connait pas.

Le front compose lui-meme les phrases (« Modifié par la loi n° 2026/002 du 14
avril 2026 ») dans la langue de l'interface : le site est bilingue. D'ou des
pieces separees (nature, numero, date, codes) plutot que des phrases francaises
toutes faites. Seule la `portee`, redigee en francais par le releve des liens,
est un texte libre.
"""

from datetime import date
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

TypeDeChangement = Literal["modifie", "complete", "abroge", "abroge_partiellement", "remplace"]


class TexteLie(BaseModel):
    """
    Un texte nomme par un lien : celui qui modifie, ou celui qui est modifie.

    `id` n'est rempli que pour un texte publie dans JuriX : c'est l'adresse
    /laws/{id}. Un texte absent du corpus (« la loi n° 2012/015 », connue par
    les visas d'autres textes) n'a que son intitule, sa nature et sa date.
    """

    id: Optional[int] = Field(None, description="Identifiant JuriX, si le texte est publié dans JuriX")
    reference: Optional[str] = Field(None, description="Cote en base (« PRC-10679 »), si le texte y est")
    nature: str = Field(
        ...,
        description="Nature de l'acte, comme laws.type : loi, décret, ordonnance, arrêté, "
        "acte uniforme, autre…",
    )
    numero: Optional[str] = Field(None, description="Numéro de l'acte lu dans l'intitulé (« 2026/002 »)")
    titre: str = Field(..., description="Intitulé complet")
    date_signature: Optional[date] = Field(
        None,
        description="Date de l'acte (signature), lue dans son intitulé ; null quand elle n'est pas "
        "sûre au jour près (voir annee) ou que l'intitulé ne date que sa publication au Journal "
        "officiel (voir date_publication_jo). Mêmes règles que GET /laws/{id}",
    )
    annee: Optional[int] = Field(None, description="Année de l'acte, sûre même quand le jour ne l'est pas")
    date_publication_jo: Optional[date] = Field(
        None,
        description="Date de publication au Journal officiel, quand l'intitulé ne donne que celle-là",
    )
    consultable: bool = Field(
        False, description="Le texte est publié dans JuriX : un lien vers /laws/{id} est possible"
    )


class Changement(BaseModel):
    """
    Un lien prouve entre ce texte et un autre, vu depuis ce texte.

    `sens` dit qui touche qui. « modifie_par » : `texte` a touche CE texte, aux
    articles `articles_cibles` de ce texte. « modifie » : CE texte a touche
    `texte`, aux articles `articles_cibles` de cet autre texte, par ses propres
    articles `articles_source`.
    """

    id: str = Field(..., description="Identifiant dans cette réponse (« c3 »), pour par_article et par_morceau")
    sens: Literal["modifie_par", "modifie"]
    type: TypeDeChangement = Field(..., description="Ce que le texte récent fait au texte ancien")
    nature_modification: Literal[
        "reecrit_en_entier", "modifie", "complete", "abroge", "abroge_partiellement", "remplace"
    ] = Field(
        ...,
        description="`type`, précisé en « reecrit_en_entier » quand le relevé le dit "
        "(sinon la portée dit l'étendue)",
    )
    libelle: str = Field(
        ..., description="Libellé français de nature_modification : « réécrit en entier », « abrogé en partie »…"
    )
    texte: TexteLie = Field(..., description="L'AUTRE texte du lien")
    etendue: Literal["articles", "texte", "inconnue"] = Field(
        ...,
        description="« articles » : articles_cibles en donne la liste ; « texte » : le texte "
        "entier (ou la liste est dans la portée) ; « inconnue » : les articles touchés ne sont "
        "pas connus",
    )
    articles_cibles: List[str] = Field(
        default_factory=list,
        description="Articles touchés, tels que relevés (« 6 », « 34 à 101 ») ; vide : voir etendue",
    )
    section: Optional[str] = Field(
        None,
        description="Début de l'intitulé de section des articles visés, pour une compilation "
        "qui reprend les mêmes numéros dans plusieurs titres (Code civil)",
    )
    articles_source: List[str] = Field(
        default_factory=list,
        description="Articles du texte récent qui portent le changement ; vide : non précisé",
    )
    portee: Optional[str] = Field(
        None, description="Ce qui change, en une phrase française du relevé, sans point final"
    )
    depasse: bool = Field(
        False,
        description="Un texte plus récent a depuis réécrit ou abrogé les mêmes dispositions : "
        "ce changement relève de l'historique, il ne dit plus le droit en vigueur",
    )
    depasse_par: Optional[TexteLie] = Field(None, description="Ce texte plus récent")
    integre_au_texte: bool = Field(
        False,
        description="Le texte affiché par JuriX contient déjà ce changement (version consolidée)",
    )


class VersionEnBase(BaseModel):
    """
    Ce qu'on sait du texte affiche, quand on sait quelque chose.

    Absent (null) : rien d'etabli. Present : le document en base est une
    version consolidee, qui contient deja les changements de `textes_integres`.
    La mention « version d'origine, non consolidée » serait fausse pour lui.
    """

    consolidee: bool = True
    textes_integres: List[TexteLie] = Field(default_factory=list)


class StatutJuridique(BaseModel):
    """
    Le statut juridique d'un texte d'apres les liens PROUVES du releve.

    Ne dit jamais « en vigueur » : un texte sans lien connu est un texte dont
    JuriX n'a releve aucune modification a `releve_le`, pas un texte intact.
    """

    releve_le: date = Field(..., description="Date du relevé des liens entre textes")
    etat: Literal["aucune_modification_connue", "modifie", "abroge", "remplace"] = Field(
        ...,
        description="Résumé des changements actuels (ni dépassés ni intégrés) subis par ce texte. "
        "« abroge » ou « remplace » : le texte entier",
    )
    types: List[TypeDeChangement] = Field(
        default_factory=list, description="Types distincts des changements actuels subis"
    )
    abroge: bool = Field(False, description="Le texte entier est abrogé ou remplacé")
    abroge_par: Optional[TexteLie] = Field(None, description="Le texte qui l'abroge ou le remplace")
    version_en_base: Optional[VersionEnBase] = None
    modifie_par: List[Changement] = Field(
        default_factory=list,
        description="Ce que d'autres textes ont fait à celui-ci, du texte le plus ancien au plus "
        "récent, historique compris. Plusieurs changements peuvent venir du même texte : les "
        "regrouper par texte.reference (ou texte.titre) pour une ligne par texte",
    )
    modifie: List[Changement] = Field(
        default_factory=list,
        description="Ce que celui-ci a fait à d'autres textes, même ordre",
    )
    par_article: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Numéro d'article normalisé (majuscules, « 1er » -> « 1 », sans suffixe de "
        "morceau « .2 ») -> identifiants des changements qui le touchent ou qu'il porte. "
        "Un changement qui a une `section` ne vaut que dans cette section",
    )
    par_morceau: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Identifiant d'article (articles[].id) -> identifiants des changements, "
        "calculés morceau par morceau avec la règle des lignes de statut du chat",
    )


class StatutDeLArticle(BaseModel):
    """Le statut d'un article : celui de son texte, et les changements qui le touchent."""

    releve_le: date
    etat_du_texte: Literal["aucune_modification_connue", "modifie", "abroge", "remplace"]
    version_en_base: Optional[VersionEnBase] = None
    changements: List[Changement] = Field(
        default_factory=list,
        description="Changements qui touchent cet article ou qu'il porte, et ceux qui portent "
        "sur tout le texte ; historique compris (depasse, integre_au_texte)",
    )
