"""
Le limiteur de debit partage par les clients Groq et Mistral.

Tout tourne sur une horloge factice : dormir fait avancer l'horloge, aucun
test n'attend pour de vrai. Ces tests tournent sans base et sans reseau.

Usage:
    pytest tests/test_core/test_limiteur.py -v
"""

import asyncio

import pytest

from app.core.limiteur import JOUR, AttenteTropLongue, Limiteur, lire_duree


class Horloge:
    """Horloge factice : `dormir` avance le temps au lieu d'attendre."""

    def __init__(self):
        self.t = 1000.0
        self.sommeils = []

    def __call__(self):
        return self.t

    def dormir(self, secondes):
        self.sommeils.append(secondes)
        self.t += secondes

    async def dormir_async(self, secondes):
        self.dormir(secondes)


@pytest.fixture
def horloge():
    return Horloge()


def limiteur(horloge, **reglages):
    return Limiteur(
        "test", horloge=horloge, dormir=horloge.dormir,
        dormir_async=horloge.dormir_async, **reglages,
    )


class TestDebit:
    def test_espace_les_departs(self, horloge):
        lim = limiteur(horloge, requetes_par_minute=30)

        attentes = [lim.reserver().attente for _ in range(3)]

        assert attentes == [0.0, 2.0, 4.0]

    def test_rafale(self, horloge):
        lim = limiteur(horloge, requetes_par_minute=30, rafale=3)

        attentes = [lim.reserver().attente for _ in range(4)]

        assert attentes == [0.0, 0.0, 0.0, 2.0]

    def test_le_seau_se_remplit_avec_le_temps(self, horloge):
        lim = limiteur(horloge, requetes_par_minute=30, rafale=2)
        lim.reserver()
        lim.reserver()

        horloge.t += 4.0

        assert [lim.reserver().attente for _ in range(2)] == [0.0, 0.0]

    def test_sans_plafond_rien_n_attend(self, horloge):
        lim = limiteur(horloge)

        assert [lim.reserver(jetons=10_000).attente for _ in range(5)] == [0.0] * 5


class TestAttenteMax:
    def test_leve_sans_rien_consommer(self, horloge):
        lim = limiteur(horloge, requetes_par_minute=30)
        lim.reserver()

        with pytest.raises(AttenteTropLongue) as erreur:
            lim.reserver(attente_max=1.0)

        assert erreur.value.attente == pytest.approx(2.0)
        assert erreur.value.raison == "debit"
        # Le refus n'a rien reserve : le suivant attend 2 s, pas 4.
        assert lim.reserver().attente == pytest.approx(2.0)

    def test_attente_egale_au_maximum_acceptee(self, horloge):
        lim = limiteur(horloge, requetes_par_minute=30)
        lim.reserver()

        assert lim.reserver(attente_max=2.0).attente == pytest.approx(2.0)


class TestJetonsParMinute:
    def test_attend_que_la_fenetre_se_vide(self, horloge):
        lim = limiteur(horloge, jetons_par_minute=1000)

        assert lim.reserver(jetons=600).attente == 0.0
        attente = lim.reserver(jetons=600).attente

        assert attente == pytest.approx(60.0)

    def test_correction_par_la_consommation_reelle(self, horloge):
        """L'estimation (prompt + plafond de sortie) est remplacee par l'usage."""
        lim = limiteur(horloge, jetons_par_minute=1000)
        reservation = lim.reserver(jetons=900)

        reservation.corriger(200)

        assert lim.reserver(jetons=700).attente == 0.0

    def test_raison_nommee(self, horloge):
        lim = limiteur(horloge, jetons_par_minute=1000)
        lim.reserver(jetons=1000)

        with pytest.raises(AttenteTropLongue) as erreur:
            lim.reserver(jetons=10, attente_max=5)

        assert erreur.value.raison == "jetons/minute"

    def test_jetons_d_entree_plafonnes_a_part(self, horloge):
        """Groq plafonne l'entree (7 000/min) en plus du total (8 000/min)."""
        lim = limiteur(horloge, jetons_par_minute=8000, jetons_entree_par_minute=7000)
        lim.reserver(jetons=6500, jetons_entree=6000)

        with pytest.raises(AttenteTropLongue) as erreur:
            lim.reserver(jetons=1200, jetons_entree=1100, attente_max=5)

        assert erreur.value.raison == "jetons-entree/minute"
        # Une entree courte passe : 6 500 + 1 200 tient sous 8 000 au total, et
        # 6 000 + 100 sous 7 000 en entree.
        assert lim.reserver(jetons=1200, jetons_entree=100).attente == 0.0

    def test_correction_de_l_entree(self, horloge):
        lim = limiteur(horloge, jetons_entree_par_minute=1000)
        reservation = lim.reserver(jetons=950, jetons_entree=900)

        reservation.corriger(400, 300)

        assert lim.reserver(jetons=700, jetons_entree=700).attente == 0.0

    def test_demande_plus_grosse_que_le_plafond(self, horloge):
        """Elle part seule, fenetre vide, au lieu d'etre bloquee pour toujours."""
        lim = limiteur(horloge, jetons_par_minute=1000)
        lim.reserver(jetons=10)

        assert lim.reserver(jetons=5000).attente == pytest.approx(60.0)


class TestBudgetJournalier:
    def test_requetes_par_jour(self, horloge):
        lim = limiteur(horloge, requetes_par_jour=2)
        lim.reserver()
        lim.reserver()

        with pytest.raises(AttenteTropLongue) as erreur:
            lim.reserver(attente_max=60)

        assert erreur.value.raison == "requetes/jour"
        assert erreur.value.attente == pytest.approx(JOUR)

    def test_jetons_par_jour(self, horloge):
        lim = limiteur(horloge, jetons_par_jour=1000)
        lim.reserver(jetons=800)

        with pytest.raises(AttenteTropLongue) as erreur:
            lim.reserver(jetons=300, attente_max=60)

        assert erreur.value.raison == "jetons/jour"


class TestRepousser:
    def test_apres_un_429(self, horloge):
        lim = limiteur(horloge, requetes_par_minute=30)

        lim.repousser(30, "429 Retry-After")

        with pytest.raises(AttenteTropLongue) as erreur:
            lim.reserver(attente_max=5)
        assert erreur.value.raison == "429 Retry-After"
        assert lim.reserver().attente == pytest.approx(30.0)

    def test_une_echeance_plus_proche_ne_raccourcit_pas(self, horloge):
        lim = limiteur(horloge)
        lim.repousser(60)
        lim.repousser(5)

        assert lim.reserver().attente == pytest.approx(60.0)


class TestRecaler:
    def test_plus_aucune_requete_du_jour(self, horloge):
        lim = limiteur(horloge, requetes_par_jour=1000)

        lim.recaler(requetes_restantes_jour=0, reprise_requetes_s=120)

        assert lim.reserver().attente == pytest.approx(120.0)

    def test_les_appels_des_autres_processus_comptent(self, horloge):
        """Le fournisseur annonce 100 jetons restants : les autres ont consomme le reste."""
        lim = limiteur(horloge, jetons_par_minute=1000)

        lim.recaler(jetons_restants_minute=100)

        assert lim.reserver(jetons=50).attente == 0.0
        assert lim.reserver(jetons=200).attente == pytest.approx(60.0)

    def test_l_ecart_s_inscrit_a_sa_place_dans_le_temps(self, horloge):
        """
        Le recalage tombe a « maintenant », AVANT un depart deja reserve pour
        plus tard. Ajoute en queue, il faisait attendre 88 s au lieu de 59.
        """
        lim = limiteur(horloge, jetons_par_minute=8000)
        horloge.t = 99.0
        lim.reserver(jetons=1000)
        lim.repousser(31)
        assert lim.reserver(jetons=3000).depart == pytest.approx(130.0)
        horloge.t = 101.0
        lim.recaler(jetons_restants_minute=2000)  # les autres ont pris 2 000

        horloge.t = 102.0
        reservation = lim.reserver(jetons=4000)

        # A 161, l'ecart (inscrit a 101) est sorti de la fenetre : 3 000 + 4 000.
        assert reservation.depart == pytest.approx(161.0)

    def test_requetes_du_jour_recalees(self, horloge):
        lim = limiteur(horloge, requetes_par_jour=10)

        lim.recaler(requetes_restantes_jour=1)

        assert lim.reserver().attente == 0.0
        with pytest.raises(AttenteTropLongue):
            lim.reserver(attente_max=60)


class TestAttendre:
    def test_un_429_pendant_le_sommeil_retarde_le_depart(self, horloge):
        """
        Un autre appel recoit un 429 pendant que celui-ci dort : partir a
        l'heure reservee, c'etait partir en pleine penalite.
        """
        lim = limiteur(horloge, requetes_par_minute=30)
        lim.reserver()
        dormir = horloge.dormir

        def dormir_et_subir_un_429(secondes):
            dormir(secondes)
            if len(horloge.sommeils) == 1:
                lim.repousser(10, "429")

        lim._dormir = dormir_et_subir_un_429

        lim.attendre_sync()

        assert horloge.sommeils == [2.0, 10.0]
        assert horloge.t == pytest.approx(1012.0)

    def test_le_supplement_respecte_attente_max(self, horloge):
        lim = limiteur(horloge, requetes_par_minute=30)
        lim.reserver()
        dormir = horloge.dormir

        def dormir_et_subir_un_429(secondes):
            dormir(secondes)
            lim.repousser(60, "429")

        lim._dormir = dormir_et_subir_un_429

        with pytest.raises(AttenteTropLongue):
            lim.attendre_sync(attente_max=5)

    def test_synchrone_dort_l_attente(self, horloge):
        lim = limiteur(horloge, requetes_par_minute=30)

        lim.attendre_sync()
        lim.attendre_sync()

        assert horloge.sommeils == [2.0]
        assert horloge.t == pytest.approx(1002.0)

    async def test_asynchrone_dort_l_attente(self, horloge):
        lim = limiteur(horloge, requetes_par_minute=30)

        await lim.attendre()
        await lim.attendre()

        assert horloge.sommeils == [2.0]


class TestConcurrence:
    async def test_un_seul_appel_a_la_fois(self, horloge):
        lim = limiteur(horloge, concurrence=1)
        dedans = []
        relache = asyncio.Event()

        async def appel(nom):
            async with lim.creneau():
                dedans.append(nom)
                await relache.wait()

        premiere = asyncio.create_task(appel("a"))
        seconde = asyncio.create_task(appel("b"))
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert dedans == ["a"]
        relache.set()
        await asyncio.gather(premiere, seconde)
        assert dedans == ["a", "b"]

    def test_un_semaphore_par_boucle(self, horloge):
        """
        Partage entre deux boucles, un asyncio.Semaphore leverait RuntimeError.
        Il ne se lie a sa boucle que lorsqu'un appel doit ATTENDRE : d'ou deux
        appels concurrents pour une seule place, dans chaque boucle.
        """
        lim = limiteur(horloge, concurrence=1)

        async def deux_appels():
            async def appel():
                async with lim.creneau():
                    await asyncio.sleep(0)

            await asyncio.gather(appel(), appel())

        asyncio.run(deux_appels())
        asyncio.run(deux_appels())

    async def test_l_attente_d_une_place_est_bornee(self, horloge):
        """
        Toutes les places prises par des generations longues : l'appel
        attendait sans limite, sans secours ni 503.
        """
        lim = limiteur(horloge, concurrence=1)
        liberer = asyncio.Event()

        async def occuper():
            async with lim.creneau():
                await liberer.wait()

        occupant = asyncio.create_task(occuper())
        await asyncio.sleep(0)

        with pytest.raises(AttenteTropLongue) as erreur:
            async with lim.creneau(attente_max=0.05):
                pass

        assert erreur.value.raison == "concurrence"
        liberer.set()
        await occupant

    async def test_le_refus_rend_la_place(self, horloge):
        lim = limiteur(horloge, requetes_par_minute=30, concurrence=1)
        async with lim.creneau():
            pass

        with pytest.raises(AttenteTropLongue):
            async with lim.creneau(attente_max=0.5):
                pass

        horloge.t += 2.0
        async with lim.creneau(attente_max=0.5):
            pass


@pytest.mark.parametrize(
    "texte, secondes",
    [
        ("7", 7.0),
        ("7.66s", 7.66),
        ("2m59.56s", 179.56),
        ("1h2m3s", 3723.0),
        ("120ms", 0.12),
        ("", None),
        (None, None),
        ("bientot", None),
        ("3s plus tard", None),
    ],
)
def test_lire_duree(texte, secondes):
    resultat = lire_duree(texte)
    if secondes is None:
        assert resultat is None
    else:
        assert resultat == pytest.approx(secondes)
