"""
Reperes d'un texte dans une liste dense : objet en clair, numero, date de
l'acte, annee et statut prouve en bref.

Une ligne de liste se lit ainsi : « Décret n° 2018/763 · 10 déc. 2018 ·
Fonction publique », puis l'objet, « Révocation d'un magistrat », puis le
statut s'il est prouve. Ces champs sont CALCULES a la serialisation, depuis
le titre, la reference et la date de la ligne : rien n'est stocke, et une
reponse relue dans un cache ecrit avant leur ajout les porte aussi.

Partages par les resultats de POST /search et les lignes de GET /laws : les
deux listes disent la meme chose du meme texte.
"""

from datetime import date
from typing import Optional

from pydantic import BaseModel, computed_field

from app.services.reperes_textes import dates_de_l_acte
from app.services.statut_en_bref import StatutEnBref, statut_en_bref
from app.utils.titre_officiel import numero_du_titre, objet_du_titre


class ReperesDuTexte(BaseModel):
    """
    A heriter par un modele qui porte `title`, `reference` et
    `publication_date` : il y gagne objet, numero, date_signature, annee,
    date_publication_jo et statut.
    """

    @computed_field(  # type: ignore[prop-decorator]
        description=(
            "Objet du texte tire de son intitule officiel (« Révocation d'un "
            "magistrat ») ; null quand la decoupe n'est pas sure : afficher alors `title`"
        )
    )
    @property
    def objet(self) -> Optional[str]:
        return objet_du_titre(getattr(self, "title", None))

    @computed_field(  # type: ignore[prop-decorator]
        description="Numero de l'acte tel que l'intitule l'ecrit (« 2018/763 », « 0342/CAB/PR »)"
    )
    @property
    def numero(self) -> Optional[str]:
        return numero_du_titre(getattr(self, "title", None))

    # Les trois dates suivent dates_de_l_acte, comme la page du texte
    # (LawDetailResponse) et les textes nommes par un statut (TexteLie) : une
    # liste et la page ne peuvent pas dater differemment le meme texte. Les deux
    # Actes uniformes OHADA dont l'intitule ne date que la publication au
    # Journal officiel affichaient ici « 15 nov. 2023 » comme date de l'acte, et
    # null sur leur page.
    @computed_field(  # type: ignore[prop-decorator]
        description=(
            "Date de l'acte lue dans son intitule ; null quand l'intitule ne donne "
            "que l'annee (voir `annee`) ou ne date que sa publication au Journal "
            "officiel (voir `date_publication_jo`)"
        )
    )
    @property
    def date_signature(self) -> Optional[date]:
        return self._dates().date_signature

    @computed_field(description="Annee de l'acte")  # type: ignore[prop-decorator]
    @property
    def annee(self) -> Optional[int]:
        return self._dates().annee

    @computed_field(  # type: ignore[prop-decorator]
        description=(
            "Date de publication au Journal officiel, seulement quand l'intitule ne "
            "donne que celle-la (« publié au Journal officiel de l'OHADA du … ») ; null sinon"
        )
    )
    @property
    def date_publication_jo(self) -> Optional[date]:
        return self._dates().date_publication_jo

    def _dates(self):
        return dates_de_l_acte(
            getattr(self, "publication_date", None), getattr(self, "title", None)
        )

    @computed_field(  # type: ignore[prop-decorator]
        description=(
            "Resume des liens PROUVES avec des textes plus recents ; null : aucun "
            "lien connu, ce qui ne veut pas dire « en vigueur »"
        )
    )
    @property
    def statut(self) -> Optional[StatutEnBref]:
        return statut_en_bref(getattr(self, "reference", None))
