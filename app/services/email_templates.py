"""
Gabarits des courriels transactionnels, en francais et en anglais.

DEUX PARTIES, TOUJOURS. Chaque message existe en texte brut ET en HTML. Un
message dont le lien n'existerait que dans la partie HTML devient illisible chez
qui n'affiche pas le HTML — et le lien est justement la seule chose qui compte.

L'URL APPARAIT EN CLAIR dans les deux parties. Sans nom de domaine, l'expediteur
affiche est reecrit en `@brevosend.com`, que le destinataire ne reconnait pas :
un lien masque derriere un libelle a exactement l'allure d'un hameconnage. Voir
l'URL entiere, meme longue, aide a decider de cliquer.

AUCUN HTML VENANT DE L'UTILISATEUR. Le nom affiche est echappe : il vient d'un
champ libre a l'inscription.
"""

from html import escape

# Le nom du produit, repete plutot que centralise ailleurs : c'est un texte de
# message, pas une configuration.
PRODUIT = "JuriX"


def _enveloppe(titre: str, corps_html: str) -> str:
    """Mise en page minimale : les clients de messagerie ignorent la moitie du CSS."""
    return (
        '<div style="font-family:system-ui,-apple-system,Segoe UI,sans-serif;'
        'max-width:520px;margin:0 auto;padding:24px;color:#1a1a1a">'
        f'<h1 style="font-size:20px;margin:0 0 16px">{escape(titre)}</h1>'
        f"{corps_html}"
        '<hr style="border:none;border-top:1px solid #e5e5e5;margin:24px 0">'
        '<p style="font-size:12px;color:#666;margin:0">'
        f"{escape(PRODUIT)} — plateforme juridique camerounaise."
        "</p></div>"
    )


def _bouton(url: str, libelle: str) -> str:
    return (
        f'<p style="margin:0 0 16px"><a href="{escape(url)}" '
        'style="display:inline-block;background:#1a4d8f;color:#fff;padding:12px 20px;'
        f'border-radius:6px;text-decoration:none">{escape(libelle)}</a></p>'
        # L'URL en clair EN PLUS du bouton : voir le docstring du module.
        f'<p style="margin:0 0 16px;font-size:13px;color:#555;word-break:break-all">'
        f"{escape(url)}</p>"
    )


def verification_adresse(nom: str, url: str, langue: str = "fr") -> tuple[str, str, str]:
    """Rend (sujet, texte, html) du message de verification d'adresse."""
    if langue == "en":
        sujet = f"Confirm your {PRODUIT} email address"
        texte = (
            f"Hello {nom},\n\n"
            f"Confirm your email address to finish setting up your {PRODUIT} account:\n\n"
            f"{url}\n\n"
            "This link expires in 24 hours. If you did not create an account, "
            "you can ignore this message.\n"
        )
        html = _enveloppe(
            "Confirm your email address",
            f"<p>Hello {escape(nom)},</p>"
            f"<p>Confirm your email address to finish setting up your {PRODUIT} account.</p>"
            + _bouton(url, "Confirm my address")
            + "<p style='font-size:13px;color:#555'>This link expires in 24 hours. "
            "If you did not create an account, you can ignore this message.</p>",
        )
        return sujet, texte, html

    sujet = f"Confirmez votre adresse {PRODUIT}"
    texte = (
        f"Bonjour {nom},\n\n"
        f"Confirmez votre adresse e-mail pour terminer la creation de votre compte {PRODUIT} :\n\n"
        f"{url}\n\n"
        "Ce lien expire dans 24 heures. Si vous n'avez pas cree de compte, "
        "vous pouvez ignorer ce message.\n"
    )
    html = _enveloppe(
        "Confirmez votre adresse",
        f"<p>Bonjour {escape(nom)},</p>"
        f"<p>Confirmez votre adresse e-mail pour terminer la creation de votre compte {PRODUIT}.</p>"
        + _bouton(url, "Confirmer mon adresse")
        + "<p style='font-size:13px;color:#555'>Ce lien expire dans 24 heures. "
        "Si vous n'avez pas cree de compte, vous pouvez ignorer ce message.</p>",
    )
    return sujet, texte, html


def reinitialisation(nom: str, url: str, langue: str = "fr") -> tuple[str, str, str]:
    """Rend (sujet, texte, html) du message de reinitialisation de mot de passe."""
    if langue == "en":
        sujet = f"Reset your {PRODUIT} password"
        texte = (
            f"Hello {nom},\n\n"
            "Someone asked to reset the password for this address. "
            "If it was you, use this link:\n\n"
            f"{url}\n\n"
            "This link expires in 30 minutes and can be used once. "
            "If you did not ask for this, ignore this message: "
            "your password stays unchanged.\n"
        )
        html = _enveloppe(
            "Reset your password",
            f"<p>Hello {escape(nom)},</p>"
            "<p>Someone asked to reset the password for this address.</p>"
            + _bouton(url, "Choose a new password")
            + "<p style='font-size:13px;color:#555'>This link expires in 30 minutes "
            "and can be used once. If you did not ask for this, ignore this message: "
            "your password stays unchanged.</p>",
        )
        return sujet, texte, html

    sujet = f"Reinitialisez votre mot de passe {PRODUIT}"
    texte = (
        f"Bonjour {nom},\n\n"
        "Quelqu'un a demande la reinitialisation du mot de passe de cette adresse. "
        "Si c'est bien vous, utilisez ce lien :\n\n"
        f"{url}\n\n"
        "Ce lien expire dans 30 minutes et ne sert qu'une fois. "
        "Si vous n'avez rien demande, ignorez ce message : "
        "votre mot de passe reste inchange.\n"
    )
    html = _enveloppe(
        "Reinitialisez votre mot de passe",
        f"<p>Bonjour {escape(nom)},</p>"
        "<p>Quelqu'un a demande la reinitialisation du mot de passe de cette adresse.</p>"
        + _bouton(url, "Choisir un nouveau mot de passe")
        + "<p style='font-size:13px;color:#555'>Ce lien expire dans 30 minutes "
        "et ne sert qu'une fois. Si vous n'avez rien demande, ignorez ce message : "
        "votre mot de passe reste inchange.</p>",
    )
    return sujet, texte, html


def compte_google(nom: str, url_connexion: str, langue: str = "fr") -> tuple[str, str, str]:
    """
    Message envoye quand l'adresse correspond a un compte SANS mot de passe.

    Il n'y a rien a reinitialiser sur un tel compte, et lui greffer un mot de
    passe lui ajouterait une porte d'entree dont son proprietaire n'a jamais
    voulu. Le message honnete est : reconnectez-vous avec Google.

    Il est envoye quand meme, et c'est deliberate : rester muet ferait de
    l'absence de courriel un signal exploitable pour distinguer un compte Google
    d'une adresse inconnue.
    """
    if langue == "en":
        sujet = f"Signing in to {PRODUIT}"
        texte = (
            f"Hello {nom},\n\n"
            "You asked to reset your password, but this account signs in with Google "
            "and has no password to reset.\n\n"
            f"Use the \"Sign in with Google\" button here:\n\n{url_connexion}\n"
        )
        html = _enveloppe(
            "Signing in",
            f"<p>Hello {escape(nom)},</p>"
            "<p>You asked to reset your password, but this account signs in with Google "
            "and has no password to reset.</p>" + _bouton(url_connexion, "Go to sign-in"),
        )
        return sujet, texte, html

    sujet = f"Connexion a {PRODUIT}"
    texte = (
        f"Bonjour {nom},\n\n"
        "Vous avez demande a reinitialiser votre mot de passe, mais ce compte se "
        "connecte avec Google et n'a pas de mot de passe a reinitialiser.\n\n"
        f"Utilisez le bouton « Se connecter avec Google » ici :\n\n{url_connexion}\n"
    )
    html = _enveloppe(
        "Connexion",
        f"<p>Bonjour {escape(nom)},</p>"
        "<p>Vous avez demande a reinitialiser votre mot de passe, mais ce compte se "
        "connecte avec Google et n'a pas de mot de passe a reinitialiser.</p>"
        + _bouton(url_connexion, "Aller a la connexion"),
    )
    return sujet, texte, html
