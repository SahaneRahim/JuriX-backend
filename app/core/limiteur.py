"""
Limiteur de debit des appels aux API externes (Groq, Mistral).

POURQUOI. Les paliers gratuits sont etroits, et chaque depassement coutait
cher :
- Groq : 30 requetes par minute, 8 000 jetons par minute, 1 000 requetes et
  200 000 jetons par jour, PAR MODELE. Le reclassement des lois envoyait ses
  requetes sans attendre : 89 sur 122 sont revenues en 429.
- Mistral : 30 requetes par minute pour ministral-14b. Un 429 etait rendu tel
  quel a l'utilisateur, comme un quota du jour epuise.

Un 429 se paie deux fois : la requete est perdue, et le fournisseur impose une
attente plus longue que celle qu'on se serait imposee. Le limiteur fait donc
attendre AVANT d'envoyer.

LA MECANIQUE. Chaque appel RESERVE son depart, puis dort jusqu'a lui :
- debit : seau a jetons (algorithme GCRA), `rafale` departs d'affilee au plus ;
- jetons par minute (dont jetons d'entree), requetes et jetons par jour :
  fenetres glissantes ;
- `repousser()` : plus aucun depart avant une echeance (apres un 429) ;
- `recaler()` : l'etat suit les en-tetes `x-ratelimit-*` du fournisseur, qui
  voient aussi les appels des AUTRES processus (API et script de reclassement
  partagent la meme cle).

Les departs sont servis dans l'ordre des reservations. Une reservation dont
l'attente depasserait `attente_max` leve `AttenteTropLongue` SANS RIEN
CONSOMMER : l'appelant choisit son repli (le classement d'intention rend
« juridique » plutot que de faire attendre l'utilisateur).

L'horloge et les fonctions de sommeil sont injectables : les tests tournent
sur une horloge factice, sans aucun vrai `sleep`.

Author: JuriX Team
"""

import asyncio
import logging
import re
import threading
import time
import weakref
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator, Awaitable, Callable, Deque, Optional

logger = logging.getLogger(__name__)

MINUTE = 60.0
JOUR = 86_400.0


class AttenteTropLongue(Exception):
    """
    Le prochain depart possible est trop loin. Rien n'a ete reserve.

    `raison` nomme la contrainte qui bloque : « debit », « jetons/minute »,
    « jetons-entree/minute », « requetes/jour », « jetons/jour », ou celle
    donnee a `repousser` (apres un 429).
    """

    def __init__(self, nom: str, attente: float, raison: str):
        self.attente = attente
        self.raison = raison
        super().__init__(
            f"{nom} : prochain depart possible dans {attente:.1f} s ({raison})"
        )


@dataclass
class Reservation:
    """Un depart reserve. `corriger` remplace l'estimation des jetons par le reel."""

    depart: float
    attente: float
    jetons: int
    jetons_entree: int = 0
    _entrees: tuple = ()
    _entrees_entree: tuple = ()

    def corriger(self, jetons_reels: int, jetons_entree_reels: Optional[int] = None) -> None:
        self.jetons = max(0, int(jetons_reels))
        for entree in self._entrees:
            entree[1] = self.jetons
        if jetons_entree_reels is not None:
            self.jetons_entree = max(0, int(jetons_entree_reels))
            for entree in self._entrees_entree:
                entree[1] = self.jetons_entree


class _Fenetre:
    """
    Somme glissante sur `duree` secondes, plafonnee a `plafond`.

    Les entrees sont des listes [instant, quantite], rangees par instant : les
    departs sont servis dans l'ordre, donc chaque nouvelle entree est la plus
    recente. La fenetre d'un instant `t` couvre ]t - duree, t].
    """

    def __init__(self, duree: float, plafond: Optional[int]):
        self.duree = duree
        self.plafond = plafond
        self.entrees: Deque[list] = deque()

    def purger(self, maintenant: float) -> None:
        while self.entrees and self.entrees[0][0] <= maintenant - self.duree:
            self.entrees.popleft()

    def premier_instant(self, debut: float, quantite: int) -> float:
        """Le premier instant >= debut ou `quantite` tient dans la fenetre."""
        if self.plafond is None:
            return debut
        # Une demande plus grosse que le plafond attend une fenetre vide :
        # mieux vaut la laisser partir seule que la bloquer pour toujours.
        quantite = min(quantite, self.plafond)
        dans_la_fenetre = [e for e in self.entrees if e[0] > debut - self.duree]
        total = sum(e[1] for e in dans_la_fenetre)
        if total + quantite <= self.plafond:
            return debut
        for entree in dans_la_fenetre:
            total -= entree[1]
            if total + quantite <= self.plafond:
                return entree[0] + self.duree
        return debut  # inatteignable : la fenetre vide accepte toujours

    def ajouter(self, instant: float, quantite: int) -> Optional[list]:
        """
        Inscrit une entree A SA PLACE dans le temps. Les departs reserves
        arrivent dans l'ordre, mais un recalage (consommation des autres
        processus, inscrite a « maintenant ») peut tomber avant des departs
        deja reserves pour plus tard : ajoutee en queue, elle faussait
        `premier_instant`, qui suppose l'ordre.
        """
        if self.plafond is None:
            return None
        entree = [instant, quantite]
        position = len(self.entrees)
        while position > 0 and self.entrees[position - 1][0] > instant:
            position -= 1
        self.entrees.insert(position, entree)
        return entree

    def total(self, instant: float) -> int:
        return sum(e[1] for e in self.entrees if e[0] > instant - self.duree)


_UNITES = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}
_MORCEAU_DE_DUREE = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")


def lire_duree(texte: Optional[str]) -> Optional[float]:
    """
    Duree d'un en-tete de fournisseur, en secondes.

    `Retry-After` porte des secondes (« 7 ») ; Groq ecrit ses
    `x-ratelimit-reset-*` en « 2m59.56s », « 7.66s » ou « 120ms ».
    Rend None pour une valeur absente ou illisible.
    """
    if texte is None:
        return None
    texte = str(texte).strip().lower()
    if not texte:
        return None
    try:
        return max(0.0, float(texte))
    except ValueError:
        pass
    morceaux = _MORCEAU_DE_DUREE.findall(texte)
    if not morceaux or "".join(n + u for n, u in morceaux) != texte:
        return None
    return sum(float(n) * _UNITES[u] for n, u in morceaux)


class Limiteur:
    """
    Debit, jetons et budget journalier d'UN modele chez UN fournisseur.

    Partage par toutes les taches et tous les fils du processus : la
    reservation se fait sous verrou, sans jamais attendre sous ce verrou.
    """

    def __init__(
        self,
        nom: str,
        *,
        requetes_par_minute: Optional[float] = None,
        rafale: int = 1,
        jetons_par_minute: Optional[int] = None,
        jetons_entree_par_minute: Optional[int] = None,
        requetes_par_jour: Optional[int] = None,
        jetons_par_jour: Optional[int] = None,
        concurrence: Optional[int] = None,
        horloge: Callable[[], float] = time.monotonic,
        dormir: Callable[[float], None] = time.sleep,
        dormir_async: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        if rafale < 1:
            raise ValueError("rafale doit valoir au moins 1")
        self.nom = nom
        self.intervalle = MINUTE / requetes_par_minute if requetes_par_minute else 0.0
        self.tolerance = (rafale - 1) * self.intervalle
        self.concurrence = concurrence
        self._jetons_minute = _Fenetre(MINUTE, jetons_par_minute)
        self._jetons_entree_minute = _Fenetre(MINUTE, jetons_entree_par_minute)
        self._requetes_jour = _Fenetre(JOUR, requetes_par_jour)
        self._jetons_jour = _Fenetre(JOUR, jetons_par_jour)
        self._horloge = horloge
        self._dormir = dormir
        self._dormir_async = dormir_async
        self._verrou = threading.Lock()
        # Instant theorique du prochain depart conforme (GCRA).
        self._tat = float("-inf")
        self._dernier_depart = float("-inf")
        self._pas_avant = float("-inf")
        self._raison_pas_avant = "repousse"
        # Un asyncio.Semaphore appartient a la boucle qui l'a vu en premier :
        # partage entre deux boucles (une par test, ou un script), il leve
        # RuntimeError. Un semaphore par boucle, oublie avec elle.
        self._semaphores: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()

    # ------------------------------------------------------------------ etat

    def maintenant(self) -> float:
        return self._horloge()

    def bloque_jusqu_a(self) -> float:
        """Echeance posee par `repousser` (ou -inf)."""
        return self._pas_avant

    def repousser(self, secondes: float, raison: str = "repousse") -> None:
        """Plus aucun depart avant `secondes` : le fournisseur a repondu 429."""
        with self._verrou:
            echeance = self._horloge() + max(0.0, secondes)
            if echeance > self._pas_avant:
                self._pas_avant = echeance
                self._raison_pas_avant = raison
        logger.warning("⏳ %s : departs suspendus %.1f s (%s)", self.nom, secondes, raison)

    def recaler(
        self,
        *,
        requetes_restantes_jour: Optional[int] = None,
        reprise_requetes_s: Optional[float] = None,
        jetons_restants_minute: Optional[int] = None,
    ) -> None:
        """
        Aligne l'etat sur ce que le fournisseur annonce.

        Le fournisseur voit TOUS les appels faits avec la cle ; ce processus ne
        voit que les siens. La difference est inscrite comme une consommation
        anonyme, qui sort des fenetres a leur rythme normal.
        """
        with self._verrou:
            maintenant = self._horloge()
            if requetes_restantes_jour is not None and self._requetes_jour.plafond:
                if requetes_restantes_jour <= 0:
                    echeance = maintenant + (reprise_requetes_s or JOUR)
                    if echeance > self._pas_avant:
                        self._pas_avant = echeance
                        self._raison_pas_avant = "requetes/jour"
                else:
                    self._ajouter_l_ecart(
                        self._requetes_jour, maintenant, requetes_restantes_jour
                    )
            if jetons_restants_minute is not None and self._jetons_minute.plafond:
                self._ajouter_l_ecart(self._jetons_minute, maintenant, jetons_restants_minute)

    @staticmethod
    def _ajouter_l_ecart(fenetre: _Fenetre, maintenant: float, restant: int) -> None:
        fenetre.purger(maintenant)
        ecart = (fenetre.plafond - max(0, restant)) - fenetre.total(maintenant)
        if ecart > 0:
            fenetre.ajouter(maintenant, ecart)

    # ----------------------------------------------------------- reservation

    def reserver(
        self,
        jetons: int = 0,
        attente_max: Optional[float] = None,
        *,
        jetons_entree: Optional[int] = None,
    ) -> Reservation:
        """
        Reserve le prochain depart possible et rend l'attente a observer.

        `jetons` estime le total de l'appel (entree et sortie) ; `jetons_entree`
        sa seule entree, quand le fournisseur la plafonne a part (Groq).

        Leve AttenteTropLongue si l'attente depasse `attente_max` : dans ce cas
        RIEN n'est reserve, et l'appelant peut se replier sans penaliser les
        suivants.
        """
        jetons = max(0, int(jetons))
        entree = max(0, int(jetons if jetons_entree is None else jetons_entree))
        with self._verrou:
            maintenant = self._horloge()
            for fenetre in (
                self._jetons_minute, self._jetons_entree_minute,
                self._requetes_jour, self._jetons_jour,
            ):
                fenetre.purger(maintenant)

            depart, raison = maintenant, "debit"
            for candidat, cause in (
                (self._tat - self.tolerance, "debit"),
                (self._dernier_depart, "debit"),
                (self._pas_avant, self._raison_pas_avant),
            ):
                if candidat > depart:
                    depart, raison = candidat, cause

            # Les fenetres repoussent le depart ; chacune peut en appeler une
            # autre, d'ou la boucle jusqu'a stabilite (deux tours en pratique).
            while True:
                precedent = depart
                for fenetre, quantite, cause in (
                    (self._jetons_minute, jetons, "jetons/minute"),
                    (self._jetons_entree_minute, entree, "jetons-entree/minute"),
                    (self._requetes_jour, 1, "requetes/jour"),
                    (self._jetons_jour, jetons, "jetons/jour"),
                ):
                    suivant = fenetre.premier_instant(depart, quantite)
                    if suivant > depart:
                        depart, raison = suivant, cause
                if depart == precedent:
                    break

            attente = depart - maintenant
            if attente_max is not None and attente > attente_max:
                raise AttenteTropLongue(self.nom, attente, raison)

            if self.intervalle:
                self._tat = max(self._tat, depart) + self.intervalle
            self._dernier_depart = depart
            entrees = tuple(
                entree
                for entree in (
                    self._jetons_minute.ajouter(depart, jetons),
                    self._jetons_jour.ajouter(depart, jetons),
                )
                if entree is not None
            )
            entree_minute = self._jetons_entree_minute.ajouter(depart, entree)
            self._requetes_jour.ajouter(depart, 1)

        return Reservation(
            depart=depart,
            attente=max(0.0, attente),
            jetons=jetons,
            jetons_entree=entree,
            _entrees=entrees,
            _entrees_entree=(entree_minute,) if entree_minute is not None else (),
        )

    async def attendre(
        self,
        jetons: int = 0,
        attente_max: Optional[float] = None,
        *,
        jetons_entree: Optional[int] = None,
    ) -> Reservation:
        """Reserve, puis dort jusqu'au depart (sans bloquer la boucle)."""
        reservation = self.reserver(jetons, attente_max, jetons_entree=jetons_entree)
        debut = self._horloge()
        if reservation.attente > 0:
            await self._dormir_async(reservation.attente)
        supplement = self._supplement(debut, attente_max)
        while supplement > 0:
            await self._dormir_async(supplement)
            supplement = self._supplement(debut, attente_max)
        return reservation

    def attendre_sync(
        self,
        jetons: int = 0,
        attente_max: Optional[float] = None,
        *,
        jetons_entree: Optional[int] = None,
    ) -> Reservation:
        """Reserve, puis dort jusqu'au depart : pour le pipeline et les scripts."""
        reservation = self.reserver(jetons, attente_max, jetons_entree=jetons_entree)
        debut = self._horloge()
        if reservation.attente > 0:
            self._dormir(reservation.attente)
        supplement = self._supplement(debut, attente_max)
        while supplement > 0:
            self._dormir(supplement)
            supplement = self._supplement(debut, attente_max)
        return reservation

    def _supplement(self, debut: float, attente_max: Optional[float]) -> float:
        """
        L'attente qu'impose un `repousser` survenu PENDANT le sommeil.

        Un appelant deja endormi partait a l'heure reservee, en plein dans la
        penalite qu'un 429 venait d'imposer : chaque depart reserve recoltait
        son propre 429. Au-dela de `attente_max` (compte depuis la
        reservation), AttenteTropLongue ; la reservation reste comptee, ce qui
        est prudent.
        """
        with self._verrou:
            maintenant = self._horloge()
            supplement = self._pas_avant - maintenant
            raison = self._raison_pas_avant
        if supplement <= 0:
            return 0.0
        if attente_max is not None and maintenant + supplement - debut > attente_max:
            raise AttenteTropLongue(self.nom, maintenant + supplement - debut, raison)
        return supplement

    @asynccontextmanager
    async def creneau(
        self,
        jetons: int = 0,
        attente_max: Optional[float] = None,
        *,
        jetons_entree: Optional[int] = None,
    ) -> AsyncIterator[Reservation]:
        """
        Un appel en cours : place parmi les `concurrence` autorises, puis depart.

        La place est prise AVANT la reservation : reserver d'abord ferait
        partir la requete plus tard que prevu, et les fenetres compteraient
        un depart qui n'a pas eu lieu a l'instant inscrit.

        `attente_max` borne AUSSI l'attente d'une place : sans cela, quand
        toutes les places sont prises par des generations longues, l'appel
        attendait sans limite, et ni le modele de secours ni le 503 promis
        n'arrivaient. Le temps passe a attendre la place est deduit du budget
        de la reservation.
        """
        semaphore = self._semaphore()
        restant = attente_max
        if semaphore is not None:
            if attente_max is None:
                await semaphore.acquire()
            else:
                debut = time.monotonic()
                try:
                    await asyncio.wait_for(semaphore.acquire(), timeout=attente_max)
                except asyncio.TimeoutError:
                    raise AttenteTropLongue(self.nom, attente_max, "concurrence") from None
                restant = max(0.0, attente_max - (time.monotonic() - debut))
        try:
            yield await self.attendre(jetons, restant, jetons_entree=jetons_entree)
        finally:
            if semaphore is not None:
                semaphore.release()

    def _semaphore(self) -> Optional[asyncio.Semaphore]:
        if not self.concurrence:
            return None
        boucle = asyncio.get_running_loop()
        semaphore = self._semaphores.get(boucle)
        if semaphore is None:
            semaphore = asyncio.Semaphore(self.concurrence)
            self._semaphores[boucle] = semaphore
        return semaphore
