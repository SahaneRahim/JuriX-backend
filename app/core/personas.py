"""
Les profils (personas) de JuriX : la seule liste, et sa normalisation.

POURQUOI CE MODULE. La liste vivait en trois copies, qui pouvaient diverger :
le validateur de app/schemas/rag.py, PersonaService.VALID_PERSONAS et les cles
de prompts.SYSTEM_PROMPTS. Une valeur inconnue y etait traitee de trois facons
differentes : 422, 400, ou repli SILENCIEUX sur le prompt citoyen —
get_system_prompt('etudiant', 'fr') rendait le prompt citoyen, sans un mot.

Ce module n'importe rien de l'application : schemas, services, scripts et
tests peuvent s'en servir sans cycle d'import.

« etudiant » sans accent, « Étudiant », la forme decomposee (NFD, celle d'un
copier-coller depuis macOS) designent tous le meme profil. La forme canonique
est 'étudiant' en NFC (U+00E9), celle que fige la contrainte de la migration
015 sur users.persona.
"""

import unicodedata
from typing import Dict, Tuple

# Les profils proposes a l'utilisateur. L'accent est ecrit en echappement : la
# forme NFC ne depend ainsi d'aucun editeur ni d'aucun copier-coller.
PERSONAS: Tuple[str, ...] = ("citoyen", "avocat", "entrepreneur", "\u00e9tudiant")

# La valeur enregistree quand l'utilisateur n'a pas repondu (« Ne pas
# repondre »). Sans accent, expres : pas de piege NFC/NFD. Ce n'est pas un
# profil proposable, ni une valeur qu'un compte peut garder.
PROFIL_GENERAL = "general"

# Ce qu'une conversation peut porter.
PROFILS: Tuple[str, ...] = PERSONAS + (PROFIL_GENERAL,)


def _replier(valeur: str) -> str:
    """Sans accents, sans espaces au bord, sans casse : la cle de comparaison."""
    decompose = unicodedata.normalize("NFKD", valeur)
    sans_marques = "".join(c for c in decompose if not unicodedata.combining(c))
    return sans_marques.strip().casefold()


_PAR_CLE: Dict[str, str] = {_replier(profil): profil for profil in PROFILS}


def _refus(valeur: object, admises: Tuple[str, ...]) -> ValueError:
    return ValueError(f"Profil inconnu : {valeur!r}. Valeurs admises : {', '.join(admises)}")


def normaliser_profil(valeur: object) -> str:
    """
    Le profil canonique d'une requete de chat ou d'une conversation.

    None, '' ou une chaine blanche donnent PROFIL_GENERAL : l'utilisateur n'a
    pas repondu. Sinon la valeur doit designer l'un des PROFILS, a la casse et
    aux accents pres.

    Raises:
        ValueError: valeur non textuelle ou inconnue.
    """
    if valeur is None:
        return PROFIL_GENERAL
    if not isinstance(valeur, str):
        raise _refus(valeur, PROFILS)
    if not valeur.strip():
        return PROFIL_GENERAL
    canonique = _PAR_CLE.get(_replier(valeur))
    if canonique is None:
        raise _refus(valeur, PROFILS)
    return canonique


def normaliser_persona(valeur: object) -> str:
    """
    L'un des PERSONAS, et lui seul : ce qu'un compte peut enregistrer.

    Un refus n'est jamais stocke sur un compte, qui garde alors NULL : None,
    '' et PROFIL_GENERAL sont donc refuses ici.

    Raises:
        ValueError: valeur vide, PROFIL_GENERAL, non textuelle ou inconnue.
    """
    if valeur is None or (isinstance(valeur, str) and not valeur.strip()):
        raise _refus(valeur, PERSONAS)
    profil = normaliser_profil(valeur)
    if profil == PROFIL_GENERAL:
        raise _refus(valeur, PERSONAS)
    return profil
