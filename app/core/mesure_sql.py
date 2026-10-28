"""
Compteur des instructions SQL de la requete HTTP en cours.

POURQUOI. Le nombre d'allers-retours vers la base a ete la premiere cause de
lenteur mesuree — 38 par recherche a 83 ms chacun, en octobre 2026 — et il ne
se comptait qu'en local, derriere un proxy TCP. Compte ici pour chaque
requete, il part dans l'en-tete Server-Timing (app/main.py, DureeServeur), a
cote de la duree totale :

    Server-Timing: app;dur=1082.4, db;desc="n=5 tx=2";dur=310.2

- n : instructions envoyees a la base, reussies ou non ;
- tx : transactions ouvertes ;
- cx : connexions ouvertes pendant la requete, quand il y en a (TLS compris,
  une vingtaine d'allers-retours chacune : c'est le prix d'un reveil) ;
- ping : connexions inactives sondees a leur sortie du pool, quand il y en a
  (database._verifier_a_la_sortie) ;
- dur : somme des durees des instructions, en ms. Les volets paralleles d'une
  recherche hybride s'y additionnent : elle peut depasser `app`.

Lire les allers-retours : les caches d'instructions etant coupes (pooler en
mode transaction), chaque instruction en coute DEUX (preparation, execution),
et chaque transaction deux de plus (BEGIN, puis COMMIT ou ROLLBACK). Soit
environ 2 × n + 2 × tx, plus un par sondage et l'ouverture des connexions.

Hors du compte : ce qui suit l'envoi des en-tetes — la fermeture de la session
de la requete (son ROLLBACK), les ecritures differees des recherches.

Bon marche : par instruction, une variable de contexte lue, deux lectures
d'horloge et deux additions. Aucun texte SQL ni parametre n'est garde : une
question d'utilisateur n'a rien a faire dans une mesure.
"""

import contextvars
import time
from dataclasses import dataclass
from typing import Optional, Tuple

from sqlalchemy import event


@dataclass
class MesureSQL:
    """Ce qu'une requete a demande a la base. Des nombres, rien d'autre."""

    instructions: int = 0
    transactions: int = 0
    connexions: int = 0
    sondages: int = 0
    duree_s: float = 0.0

    def en_tete(self) -> str:
        """La mesure en metrique Server-Timing (voir le module)."""
        desc = f"n={self.instructions} tx={self.transactions}"
        if self.connexions:
            desc += f" cx={self.connexions}"
        if self.sondages:
            desc += f" ping={self.sondages}"
        return f'db;desc="{desc}";dur={self.duree_s * 1000:.1f}'


# Un OBJET partage et non un compteur par contexte : une tache lancee par la
# requete (le volet semantique d'une recherche, sur sa propre connexion) recoit
# une copie du contexte, qui designe le meme objet. Ses instructions comptent
# donc pour la requete, comme il se doit.
_MESURE: "contextvars.ContextVar[Optional[MesureSQL]]" = contextvars.ContextVar(
    "jurix_mesure_sql", default=None
)

# Debut de l'instruction en cours, dans le `info` de la connexion : une
# connexion n'execute qu'une instruction a la fois.
_DEBUT = "jurix_debut_instruction"


def commencer() -> Tuple[MesureSQL, contextvars.Token]:
    """Ouvre la mesure du contexte courant ; terminer() la referme."""
    mesure = MesureSQL()
    return mesure, _MESURE.set(mesure)


def terminer(jeton: contextvars.Token) -> None:
    _MESURE.reset(jeton)


def courante() -> Optional[MesureSQL]:
    """La mesure du contexte courant, None hors d'une requete."""
    return _MESURE.get()


def ne_plus_compter() -> None:
    """
    Pour une tache lancee par la requete mais qui lui survit (ecritures
    differees) : ses instructions viennent apres l'en-tete, et ne disent rien
    du temps de reponse. Sans effet sur la requete, qui garde son contexte.
    """
    _MESURE.set(None)


def noter_un_sondage() -> None:
    mesure = _MESURE.get()
    if mesure is not None:
        mesure.sondages += 1


def _avant_instruction(conn, cursor, statement, parameters, context, executemany) -> None:
    mesure = _MESURE.get()
    if mesure is None:
        return
    mesure.instructions += 1
    conn.info[_DEBUT] = time.perf_counter()


def _fin_d_instruction(conn) -> None:
    debut = conn.info.pop(_DEBUT, None)
    mesure = _MESURE.get()
    if debut is not None and mesure is not None:
        mesure.duree_s += time.perf_counter() - debut


def _apres_instruction(conn, cursor, statement, parameters, context, executemany) -> None:
    _fin_d_instruction(conn)


def _instruction_en_echec(contexte) -> None:
    # Une instruction refusee a coute ses allers-retours elle aussi.
    if contexte.connection is not None:
        _fin_d_instruction(contexte.connection)


def _debut_de_transaction(conn) -> None:
    mesure = _MESURE.get()
    if mesure is not None:
        mesure.transactions += 1


def _connexion_ouverte(dbapi_connection, connection_record) -> None:
    mesure = _MESURE.get()
    if mesure is not None:
        mesure.connexions += 1


_ECOUTEURS = (
    ("before_cursor_execute", _avant_instruction),
    ("after_cursor_execute", _apres_instruction),
    ("handle_error", _instruction_en_echec),
    ("begin", _debut_de_transaction),
    ("connect", _connexion_ouverte),
)


def brancher(moteur_synchrone) -> None:
    """Pose le compteur sur un moteur (engine.sync_engine pour un moteur asyncpg)."""
    for nom, ecouteur in _ECOUTEURS:
        if not event.contains(moteur_synchrone, nom, ecouteur):
            event.listen(moteur_synchrone, nom, ecouteur)


def debrancher(moteur_synchrone) -> None:
    for nom, ecouteur in _ECOUTEURS:
        if event.contains(moteur_synchrone, nom, ecouteur):
            event.remove(moteur_synchrone, nom, ecouteur)
