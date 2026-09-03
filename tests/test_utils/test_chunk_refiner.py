"""
Tests du raffineur de chunks (règles R2 à R7).

Le module n'avait aucun test. Ces règles déterminent directement ce que le
modèle reçoit en contexte : une erreur ici se traduit par des réponses sans
fondement, sans qu'aucune exception ne soit levée.
"""

import pytest

from app.utils.chunk_refiner import (
    DocumentContext,
    normalize_for_chunking,
    refine,
)


@pytest.fixture
def contexte():
    return DocumentContext(
        reference="Décret n° 2024/191",
        title="portant ratification de la Convention de Crédit-Acheteur",
        doc_type="decret",
        date="4 juin 2024",
        category="Finances publiques",
        language="fr",
    )


def chunk(number, content, **kw):
    """Construit un chunk au format produit par text_chunker."""
    base = {
        "number": number,
        "title": None,
        "content": content,
        "position": 0,
        "parent_id": None,
        "section": None,
        "word_count": len(content.split()),
        "char_count": len(content),
        "page_number": 1,
    }
    base.update(kw)
    return base


class TestNormalisation:
    """Prépare le markdown OCR pour text_chunker."""

    def test_gras_markdown_retire(self):
        """**ARTICLE 1ER** empêche la détection : le motif attend « Article »
        en début de ligne, pas « **Article »."""
        assert "ARTICLE 1ER" in normalize_for_chunking("**ARTICLE 1ER**: Contenu")
        assert "**" not in normalize_for_chunking("**ARTICLE 1ER**: Contenu")

    def test_titre_markdown_retire(self):
        assert normalize_for_chunking("# Titre").strip() == "Titre"

    def test_lettres_espacees_recollees(self):
        """L'OCR restitue l'interlettrage des titres : « A R R Ê T E »."""
        assert "ARRÊTE" in normalize_for_chunking("A R R Ê T E :")

    def test_marqueurs_de_page_preserves(self):
        assert "<<PAGE:3>>" in normalize_for_chunking("<<PAGE:3>>\nTexte")

    def test_tableaux_preserves(self):
        out = normalize_for_chunking("<table><tr><td>a</td></tr></table>")
        assert "<table>" in out


class TestR2Contextualisation:
    def test_embed_text_porte_le_contexte(self, contexte):
        r = refine([chunk("3", "La dépense résultant des présentes dispositions." * 5)], contexte)
        embeddable = r.embeddable
        assert embeddable, "le chunk aurait dû être vectorisable"
        texte = embeddable[0]["embed_text"]
        # Sans en-tête, « La dépense résultant des présentes dispositions » est
        # un chunk orphelin : ni le lecteur ni l'embedding ne savent de quoi il parle.
        assert "Décret n° 2024/191" in texte
        assert "Finances publiques" in texte
        assert "Article 3" in texte

    def test_pas_de_libelle_article_pour_un_repli_paragraphe(self, contexte):
        """PARA_n n'est pas un vrai numéro : l'annoncer ferait citer au chatbot
        des références inexistantes."""
        r = refine([chunk("PARA_2", "Contenu quelconque suffisamment long." * 4)], contexte)
        assert "Article PARA_2" not in r.embeddable[0]["embed_text"]

    def test_contenu_non_modifie(self, contexte):
        """Le contexte va dans embed_text, jamais dans content (affichage/citation)."""
        contenu = "Article utile avec un contenu suffisamment long pour être conservé."
        r = refine([chunk("1", contenu)], contexte)
        assert r.chunks[0]["content"] == contenu


class TestR3Visas:
    def test_visas_hors_index_et_cites(self, contexte):
        visas = (
            "Vu la Constitution ;\n"
            "Vu la loi n° 2007-006 du 26 décembre 2007 portant régime financier ;\n"
            "Vu le décret n° 2011/412 du 09 décembre 2011 portant réorganisation ;"
        )
        r = refine([chunk("LEGAL_BASIS", visas)], contexte)
        assert r.legal_basis is not None
        assert not r.embeddable, "les visas ne doivent pas être vectorisés"
        # Présents dans chaque document, ils domineraient le FTS et écraseraient
        # la similarité cosinus.
        assert any("2007-006" in c for c in r.citations)
        assert any("2011/412" in c for c in r.citations)

    def test_suite_de_page_reste_indexee(self, contexte):
        """
        text_chunker étiquette LEGAL_BASIS tout le texte précédant le premier
        article. Sur une page qui ne commence pas par un article, c'est du
        contenu juridique réel : l'exclure le ferait disparaître de la recherche.
        """
        suite = "la délivrance des titres de passeports ; l'obtention d'une carte grise." * 3
        r = refine([chunk("LEGAL_BASIS", suite)], contexte)
        assert r.legal_basis is None
        assert r.chunks[0]["kind"] == "continuation"
        assert r.chunks[0]["embed"] is True


class TestR4ListesNominatives:
    TABLE = "<table><tbody>" + "".join(
        f"<tr><td>{i}.</td><td>NOM PRENOM{i:02d}</td><td>765 6{i:02d}-Y</td></tr>"
        for i in range(1, 21)
    ) + "</tbody></table>"

    def test_liste_effondree_en_un_chunk(self, contexte):
        contenu = (
            "Les anciens Gardiens de la Paix dont les noms suivent sont nommés "
            "Élèves-Inspecteurs de Police, indice 230.\n" + self.TABLE
        )
        r = refine([chunk("1", contenu)], contexte)
        assert len(r.roster) == 20
        assert len(r.embeddable) == 1, "20 personnes ne doivent pas donner 20 vecteurs"
        # L'enveloppe juridique est conservée, la liste ne l'est pas
        assert "Élèves-Inspecteurs" in r.chunks[0]["content"]
        assert "NOM PRENOM05" not in r.chunks[0]["content"]
        assert "20 personnes concernées" in r.chunks[0]["content"]

    def test_appariement_nom_matricule(self, contexte):
        contenu = "Sont nommés :\n" + self.TABLE
        r = refine([chunk("1", contenu)], contexte)
        entree = r.roster[4]
        assert entree.name == "NOM PRENOM05"
        assert entree.identifier == "765 605-Y"

    def test_entete_de_tableau_ignoree(self, contexte):
        table = (
            "<table><thead><tr><th>N°</th><th>Nom</th><th>Indice</th></tr></thead>"
            "<tbody>" + "".join(
                f"<tr><td>{i}.</td><td>NOM PRENOM{i:02d}</td><td>765 6{i:02d}-Y</td></tr>"
                for i in range(1, 8)
            ) + "</tbody></table>"
        )
        r = refine([chunk("1", "Sont nommés :\n" + table)], contexte)
        assert all(e.name != "Nom" for e in r.roster), "« Nom » compté comme personne"
        assert len(r.roster) == 7

    def test_article_normatif_non_traite_comme_liste(self, contexte):
        contenu = (
            "L'administration fiscale met en œuvre l'assistance internationale "
            "en matière de recouvrement des créances fiscales, qu'elle soit "
            "sollicitée par une autorité étrangère ou qu'elle en fasse la demande."
        )
        r = refine([chunk("L 94 septies", contenu)], contexte)
        assert not r.roster
        assert r.chunks[0]["kind"] == "article"


class TestR5R6TaillesEtTableaux:
    def test_tableau_jamais_coupe(self, contexte):
        table = "<table>" + "".join(
            f"<tr><td>ligne {i}</td><td>{i * 1000}</td></tr>" for i in range(200)
        ) + "</table>"
        r = refine([chunk("86", "Les charges sont évaluées ainsi :\n" + table)], contexte)
        # Une ligne isolée ne répond à aucune question : le tableau reste entier
        assert len(r.chunks) == 1
        assert r.chunks[0]["oversized"] is True

    def test_article_long_coupe_aux_alineas(self, contexte):
        contenu = "Dispositions générales.\n" + "\n".join(
            f"({i}) " + "Texte de l'alinéa suffisamment long pour compter. " * 12
            for i in range(1, 6)
        )
        r = refine([chunk("94", contenu)], contexte, target_max_chars=800)
        assert len(r.chunks) > 1
        assert all(c["parent_id"] == "94" for c in r.chunks[1:])

    def test_article_d_execution_hors_index(self, contexte):
        boilerplate = (
            "Le présent décret sera enregistré, publié selon la procédure "
            "d'urgence, puis inséré au Journal Officiel en français et en anglais."
        )
        r = refine([chunk("5", boilerplate)], contexte)
        # Présent dans presque chaque texte : indexé, il produirait des milliers
        # de quasi-doublons. Conservé en base, retiré du vectoriel.
        assert r.chunks[0]["kind"] == "boilerplate"
        assert r.chunks[0]["embed"] is False
        assert r.chunks[0]["content"] == boilerplate

    def test_fragment_trop_court_hors_index(self, contexte):
        r = refine([chunk("PARA_9", "45")], contexte)
        assert r.chunks[0]["embed"] is False
        assert r.chunks[0]["kind"] == "fragment"


class TestR7Deduplication:
    def test_chunks_identiques_fusionnes(self, contexte):
        contenu = "Le présent texte entre en application immédiate sur tout le territoire."
        r = refine([chunk("1", contenu), chunk("2", contenu)], contexte)
        assert r.stats["duplicates_removed"] == 1
        assert len(r.chunks) == 1


class TestRienNEstPerdu:
    def test_aucun_chunk_supprime(self, contexte):
        """
        Principe directeur du module : les chunks écartés du vectoriel gardent
        embed=False mais restent en base et cherchables en FTS.
        """
        chunks = [
            chunk("LEGAL_BASIS", "Vu la Constitution ;"),
            chunk("1", "Contenu normatif suffisamment long pour être conservé." * 3),
            chunk("5", "Le présent décret sera enregistré et publié au Journal Officiel."),
            chunk("PARA_9", "12"),
        ]
        r = refine(chunks, contexte)
        assert len(r.chunks) == len(chunks)
        assert r.stats["embeddable"] < len(chunks), "tout ne doit pas être vectorisé"
