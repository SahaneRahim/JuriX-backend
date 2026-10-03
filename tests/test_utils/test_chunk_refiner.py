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
        # Le contenu garde la liste : c'est lui qu'on affiche, qu'on cite et
        # que le plein texte indexe. La liste n'etait enregistree nulle part
        # ailleurs, et les noms disparaissaient de l'application.
        assert "Élèves-Inspecteurs" in r.chunks[0]["content"]
        assert "NOM PRENOM05" in r.chunks[0]["content"]
        # Seul le vecteur prend la forme resumee
        assert "NOM PRENOM05" not in r.chunks[0]["embed_text"]
        assert "Élèves-Inspecteurs" in r.chunks[0]["embed_text"]
        assert "20 personnes" in r.chunks[0]["embed_text"]

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


class TestAlineas:
    """
    Le motif des alineas portait un groupe capturant : re.split insere chaque
    groupe capture dans son resultat. On obtenait en alternance des fragments
    « Chapeau.\\n1 » et des alineas, et l'alinea (2) devenait « Article 94.4 ».
    """

    CONTENU = "Dispositions générales.\n" + "\n".join(
        f"({i}) " + "Texte de l'alinéa suffisamment long pour compter. " * 12
        for i in range(1, 6)
    )

    def test_un_morceau_par_alinea_numerotes_dans_l_ordre(self, contexte):
        r = refine([chunk("94", self.CONTENU)], contexte, target_max_chars=800)
        assert [c["number"] for c in r.chunks] == [f"94.{i}" for i in range(1, 6)]
        for i, c in enumerate(r.chunks, start=1):
            assert f"({i})" in c["content"]

    def test_aucun_fragment_reduit_a_un_numero(self, contexte):
        r = refine([chunk("94", self.CONTENU)], contexte, target_max_chars=800)
        assert all(c["kind"] != "fragment" for c in r.chunks)

    def test_alineas_rendus_en_liste_par_docling(self, contexte):
        contenu = self.CONTENU.replace("\n(", "\n- (")
        r = refine([chunk("94", contenu)], contexte, target_max_chars=800)
        assert [c["number"] for c in r.chunks] == [f"94.{i}" for i in range(1, 6)]


class TestTableauxMarkdown:
    """Docling et la consigne Gemini rendent les tableaux en markdown a barres."""

    LISTE = "| N° | Nom | Matricule |\n| --- | --- | --- |\n" + "".join(
        f"| {i} | NOM PRENOM{i:02d} | 765 6{i:02d}-Y |\n" for i in range(1, 8)
    )

    def test_liste_nominative_en_tableau_markdown(self, contexte):
        chapeau = (
            "Les anciens Gardiens de la Paix dont les noms suivent sont nommés "
            "Élèves-Inspecteurs de Police, indice 230, pour compter de la date de "
            "signature du présent arrêté :\n"
        )
        r = refine([chunk("1", chapeau + self.LISTE)], contexte)
        assert len(r.roster) == 7
        assert all(e.name != "Nom" for e in r.roster), "en-tête compté comme personne"
        assert r.roster[4].identifier == "765 605-Y"
        assert "NOM PRENOM05" in r.chunks[0]["content"]
        assert "NOM PRENOM05" not in r.chunks[0]["embed_text"]
        assert r.chunks[0]["kind"] == "roster"

    def test_grand_tableau_coupe_entre_ses_lignes(self, contexte):
        """
        Entier, un tableau de loi de finances faisait un chunk de 79 000
        caracteres dont l'embedding ne lisait que les 10 000 premiers.
        """
        table = "| ligne | montant |\n| --- | --- |\n" + "".join(
            f"| ligne {i} | {i * 1000} |\n" for i in range(200)
        )
        r = refine([chunk("86", "Les charges sont évaluées ainsi :\n" + table)], contexte)

        assert len(r.chunks) > 1
        assert all(c["kind"] == "table" and c["embed"] for c in r.chunks)
        assert all(len(c["content"]) <= 3000 for c in r.chunks)
        assert [c["number"] for c in r.chunks] == [f"86.{i}" for i in range(1, len(r.chunks) + 1)]
        # Chaque morceau dit de quel tableau il s'agit et nomme ses colonnes
        assert all(c["content"].startswith("Les charges sont évaluées ainsi :") for c in r.chunks)
        assert all("| ligne | montant |" in c["content"] for c in r.chunks)
        # Aucune ligne perdue, aucune dupliquee, aucune coupee
        lignes = [ln for c in r.chunks for ln in c["content"].splitlines() if ln.startswith("| ligne ") and ln[8].isdigit()]
        assert lignes == [f"| ligne {i} | {i * 1000} |" for i in range(200)]

    def test_premiere_ligne_de_donnees_non_repetee(self, contexte):
        """Sans vrai en-tete, Docling met la premiere ligne de donnees en tete."""
        table = "| 31 | EDUCATION PRESCOLAIRE | 31 915 303 |\n|---|---|---|\n" + "".join(
            f"| {32 + i} | PROGRAMME {i} | 1 000 {i:03d} |\n" for i in range(200)
        )
        r = refine([chunk("12", "Les crédits sont répartis comme suit :\n" + table)], contexte)

        assert len(r.chunks) > 1
        assert sum(c["content"].count("EDUCATION PRESCOLAIRE") for c in r.chunks) == 1

    def test_texte_apres_le_tableau_conserve(self, contexte):
        table = "| poste | montant |\n| --- | --- |\n" + "".join(f"| poste {i} | {i} |\n" for i in range(300))
        contenu = "Répartition :\n" + table + "\nLe reliquat est reporté sur l'exercice suivant."
        r = refine([chunk("7", contenu)], contexte)

        assert "Le reliquat est reporté" in r.chunks[-1]["content"]

    def test_tableau_html_trop_long_prend_la_nature_table(self, contexte):
        table = "<table>" + "".join(
            f"<tr><td>ligne {i}</td><td>{i * 1000}</td></tr>" for i in range(200)
        ) + "</table>"
        r = refine([chunk("86", "Les charges sont évaluées ainsi :\n" + table)], contexte)
        assert r.chunks[0]["kind"] == "table"

    def test_embed_text_sans_ligne_de_separation(self, contexte):
        table = "| poste | montant |\n| --- | --- |\n| Fonctionnement | 1 200 000 |\n"
        contenu = "Les crédits ouverts au titre de l'exercice sont répartis comme suit :\n" + table
        r = refine([chunk("12", contenu)], contexte)
        assert "---" not in r.chunks[0]["embed_text"]
        assert "Fonctionnement | 1 200 000" in r.chunks[0]["embed_text"]


class TestListesEnTexteBrut:
    """
    Formes reelles du corpus que l'ancien motif ratait : sur 13 listes de 5
    personnes ou plus d'un echantillon, 2 seulement etaient reconnues.
    """

    def test_rang_sans_ponctuation(self, contexte):
        lignes = "\n".join(f"{2440 + i} DAWE KOLWE RICHARD{i} 598 33{i}-A" for i in range(6))
        r = refine([chunk("1", "Sont admis au concours :\n" + lignes)], contexte)
        assert len(r.roster) == 6
        assert r.roster[0].rank == "2440"

    def test_texte_apres_le_matricule(self, contexte):
        lignes = "\n".join(f"{i}. NGO BIYONG Marie{i} 501 30{i}-Z anc. cons." for i in range(1, 7))
        r = refine([chunk("1", "Sont promus :\n" + lignes)], contexte)
        assert len(r.roster) == 6
        assert r.roster[2].identifier == "501 303-Z"

    def test_liste_sans_matricule_dans_un_texte_de_nomination(self, contexte):
        prenoms = ["Paul", "Marie", "Jean", "Claire", "Luc", "Anne", "Marc", "Rose"]
        lignes = "\n".join(f"{i}. ABENA ESSOMBA {p}" for i, p in enumerate(prenoms, start=1))
        contenu = "Sont nommés élèves commissaires de police les candidats dont les noms suivent :\n" + lignes
        r = refine([chunk("1", contenu)], contexte)
        assert len(r.roster) == 8
        assert r.chunks[0]["kind"] == "roster"

    def test_enumeration_normative_en_capitales_conservee(self, contexte):
        """Sans contexte de nomination, une enumeration reste du contenu."""
        lignes = "\n".join(
            f"{i}. MINISTERE DES {nom}" for i, nom in enumerate(
                ["FINANCES", "TRANSPORTS", "MINES", "FORETS", "SPORTS", "ARTS"], start=1
            )
        )
        contenu = "Le comité interministériel comprend les représentants des départements suivants :\n" + lignes
        r = refine([chunk("3", contenu)], contexte)
        assert not r.roster
        assert "MINISTERE DES MINES" in r.chunks[0]["content"]

    def test_quelques_noms_dans_un_long_article_ne_font_pas_une_liste(self, contexte):
        noms = "\n".join(f"{i}. ABENA Paul{i}" for i in range(1, 6))
        corps = "\n".join(f"Disposition normative numéro {i} applicable aux agents nommés." for i in range(12))
        r = refine([chunk("4", corps + "\n" + noms)], contexte)
        assert not r.roster


class TestNormalisationDocling:

    def test_commentaires_html_retires(self):
        out = normalize_for_chunking("Article 1.- Texte.\n<!-- image -->\nSuite.")
        assert "<!--" not in out

    def test_echappements_html_decodes(self):
        out = normalize_for_chunking("Ministère de l&#x27;Économie &amp; des Finances")
        assert out == "Ministère de l'Économie & des Finances"

    def test_alinea_en_liste_redevient_alinea(self):
        out = normalize_for_chunking("Article 2.- Le conseil :\n- (1) délibère ;\n- (2) vote.")
        assert "\n(1) délibère" in out and "\n(2) vote" in out

    def test_marqueurs_de_page_toujours_preserves(self):
        assert "<<PAGE:3>>" in normalize_for_chunking("<<PAGE:3>>\nArticle 1.- Texte.")


# ==================== SORTIES DOCLING REELLES ====================
#
# Extraits copies des sorties Docling du corpus (banc OCR). Chaque cas a
# produit un defaut constate : chunk perdu, liste prise pour du texte, nom
# efface, signature vectorisee.


def decoupe(pages, contexte):
    """Chaine de lecture reelle : nettoyage, normalisation, decoupe, raffinage."""
    from app.utils.markdown_cleanup import strip_stamp_blocks
    from app.utils.text_chunker import extract_articles

    texte = "\n\n".join(
        f"<<PAGE:{i}>>\n{strip_stamp_blocks(p)}" for i, p in enumerate(pages, 1) if p.strip()
    )
    bruts = extract_articles(
        normalize_for_chunking(texte), strict=False, min_article_length=1, language="fr"
    )
    return bruts, refine(bruts, contexte)


class TestSignature:
    """Decret d'une page (3997) : Docling lit le titre de l'acte APRES la signature."""

    PAGE = """Article 1.- Le Ministre de 1'Economie est habilité à signer avec la Belfius Banque Belgique une Convention de crédit acheteur d'un montant de 45 millions d'euros.

Article 2.- Le présent décret sera enregistré, publié selon la procédure d'urgence, puis inséré au Journal Officiel en français et en anglais./-

15 SEPT 2015 YAOUNDE, le

LEPRESIDENTDELA REPUBLIQUE,

PAUL BIYA

2015/39

DECRETN°

habilitant le Ministre de l'Economie, de la Planification et de l'Aménagement du Territoire, à signer avec la Belfius Banque Belgique, une Convention de crédit acheteur d'un montant de 45 millions d'euros, soit environ 29,52 milliards de F CFA, pour le financement du projet d'approvisionnement en eau potable.

## LE PRESIDENTDE LA REPUBLIQUE,"""

    def test_formule_d_execution_hors_index(self, contexte):
        _, r = decoupe([self.PAGE], contexte)
        par_numero = {c["number"]: c for c in r.chunks}
        # Collee a la signature et au titre, elle depassait 600 caracteres et
        # partait dans l'index vectoriel.
        assert par_numero["2"]["kind"] == "boilerplate"
        assert par_numero["2"]["embed"] is False

    def test_signature_conservee_hors_vectoriel(self, contexte):
        _, r = decoupe([self.PAGE], contexte)
        signature = [c for c in r.chunks if c["number"] == "SIGNATURE"]
        assert len(signature) == 1
        assert signature[0]["embed"] is False
        assert "PAUL BIYA" in signature[0]["content"], "rien n'est supprime"
        assert "PAUL BIYA" not in next(c for c in r.chunks if c["number"] == "2")["content"]

    def test_annexe_apres_la_signature_reste_vectorisee(self, contexte):
        annexe = "\n\nANNEXE I : BAREME DES REDEVANCES\n\n" + (
            "La redevance annuelle est fixée à cinq cents francs par hectare exploité. " * 4
        )
        _, r = decoupe([self.PAGE + annexe], contexte)
        annexes = [c for c in r.chunks if c["number"] == "ANNEXE"]
        assert len(annexes) == 1
        assert annexes[0]["embed"] is True
        assert "BAREME" in annexes[0]["content"]

    def test_article_sans_signature_intact(self, contexte):
        contenu = "Le Président de la République fixe par décret les modalités d'application de la présente loi."
        r = refine([chunk("12", contenu)], contexte)
        assert [c["number"] for c in r.chunks] == ["12"]
        assert r.chunks[0]["content"] == contenu


class TestListesDocling:
    def test_premiere_ligne_du_tableau_n_est_pas_un_en_tete(self, contexte):
        """9073 : Docling met en en-tete la premiere ligne de donnees."""
        page = """ARTICLE 1er.- Les Inspecteurs de Police Principaux dont les noms suivent, sont inscrits sur la liste d'aptitude.

| 1. NDOUNGA EVINA JEAN CHRISTIAN   | 607 027-V   |
|-----------------------------------|-------------|
| 2. ELOUNDOU ESSOMBA ALBERT        | 583397-B    |
| 3. AVA GABRIEL                    | 605904-J    |
| 4. WAKAM ANDRE                    | 607 246-M   |
| 5. NJAL GWEM MOISE BENOIT         | 606 036-V   |
| 6. MBITA SERGE IRENE              | 606385-U    |

ARTICLE 2.- La dépense est imputée sur le budget de l'Etat, Chapitre 12 - Article 390000 - Paragraphe 6220."""
        _, r = decoupe([page], contexte)
        noms = [e.name for e in r.roster]
        assert "NDOUNGA EVINA JEAN CHRISTIAN" in noms
        assert len(noms) == 6
        # Rang et nom partagent la cellule : ils sont separes
        assert r.roster[1].rank == "2" and r.roster[1].identifier == "583397-B"

    def test_matricule_au_format_actuel(self, contexte):
        """10532 : « 0570984M », sans espace ni tiret."""
        lignes = "\n".join(
            f"| {i}. | NOM{i} PRENOM{i} | 05{i:05d}M | Anc. Cons. 04 ms 18 jrs |" for i in range(1, 8)
        )
        table = lignes.replace("\n", "\n|---|---|---|---|\n", 1)
        r = refine([chunk("2", "Sont promus les fonctionnaires dont les noms suivent :\n" + table)], contexte)
        assert len(r.roster) == 7
        assert r.roster[0].identifier == "0500001M"

    def test_nom_colle_par_l_ocr(self, contexte):
        """4753 : « MBINCHOBLAISEAMBE », un nom en un seul mot."""
        table = """| 6. | MBANGWANAAUGUSTINEMBOLE | Mie 133 849-Y | anc. cons. 04 ans. |
|------|---------------------------|-----------------|-----------------------------------|
| 7. | MBINCHOBLAISEAMBE | Mie 145 033-A | anc. cons. 04 ans. |
| 8. | NTOCKSAMUEL | Mie 162 028-W | anc. cons. 04 ans. |
| 9. | TAPOYAYEBIA | Mie 142 530-S | anc. cons. 04 ans. |
| 10. | ASANGHANWASAMUEL | Mie 130 287-C | anc, cons. 02 ans 04 mois 14 jrs. |
| 11. | BAYIHA NICODEME | Mie 520 493-R | anc. cons. 02 ans 04 mois 14 jrs. |"""
        r = refine([chunk("3", "Sont avancés les agents dont les noms suivent :\n" + table)], contexte)
        assert [e.name for e in r.roster][:2] == ["MBANGWANAAUGUSTINEMBOLE", "MBINCHOBLAISEAMBE"]

    def test_liste_a_puces_sans_rang(self, contexte):
        """6982 : « - TEMGOUA NOUMBO HONORINE »."""
        noms = [
            "TEMGOUA NOUMBO HONORINE", "DJOPMO DOPGWA MARIE NOÉLLE", "BATOUANEN ALAIN CLOVIS",
            "BELLA ALPHONSE", "METEH AYISSI ELISE", "MUSTAPHA MACINA",
            "ASSAMBA ASSAMBA PROSPÈRE", "MFOUDOU NGAPOUT PAUL FLORIBERT",
        ]
        page = (
            "Article 1er: Les Adjudants-Chefs dont les noms suivent, admis au corps des Officiers "
            "d'Active, sont promus aux grades supérieurs ainsi qu'il suit :\n\n## GENDARMERIE NATIONALE\n\n"
            + "\n".join(f"- {n}" for n in noms)
            + "\n\nArticle 2: Le Ministre est chargé de l'application du présent Décret qui sera "
            "enregistré, puis publié au Journal Officiel./-"
        )
        _, r = decoupe([page], contexte)
        assert len(r.roster) == 8
        premier = next(c for c in r.chunks if c["number"] == "1")
        assert premier["kind"] == "roster"
        assert "TEMGOUA NOUMBO HONORINE" in premier["content"]
        assert "TEMGOUA" not in premier["embed_text"]

    def test_tableau_budgetaire_n_est_pas_une_liste(self, contexte):
        """Des intitules en capitales ne sont pas des personnes."""
        table = "| Code | Programme | AE | CP |\n|---|---|---|---|\n" + "".join(
            f"| {i} | DEVELOPPEMENT DU {nom} | 12 512 {i}63 | 11 900 {i}00 |\n"
            for i, nom in enumerate(["PRESCOLAIRE", "PRIMAIRE", "SECONDAIRE", "SUPERIEUR",
                                     "SPORT SCOLAIRE", "NUMERIQUE"], start=1)
        )
        r = refine([chunk("8", "Les crédits sont répartis par programme ainsi qu'il suit :\n" + table)], contexte)
        assert not r.roster
        assert r.chunks[0]["kind"] != "roster"
        assert "12 512 163" in r.chunks[0]["embed_text"], "les montants doivent rester vectorises"

    def test_longue_liste_coupee_entre_ses_lignes(self, contexte):
        """
        Entiere, une longue liste depasse ce que le contexte du chat lit d'un
        chunk : un nom de la fin, trouve par le plein texte, n'arrivait jamais
        au modele.
        """
        lignes = "\n".join(f"{i}. NOM{i:03d} PRENOM {700 + i % 100} {i % 10}{i % 10}0-A" for i in range(1, 301))
        chapeau = "Sont admis au concours d'entrée à l'Ecole Nationale d'Administration les candidats suivants :"
        r = refine([chunk("1", chapeau + "\n" + lignes)], contexte, target_max_chars=3000)
        morceaux = [c for c in r.chunks if c.get("parent_id") == "1"]
        assert len(morceaux) > 1
        assert sum(1 for c in morceaux if c["embed"]) == 1, "une liste ne fait qu'un vecteur"
        assert all(len(c["content"]) <= 3000 + 300 for c in morceaux)
        assert all(chapeau[:60] in c["content"] for c in morceaux), "chaque morceau dit de quelle liste il s'agit"
        tout = "\n".join(c["content"] for c in morceaux)
        assert all(f"NOM{i:03d} PRENOM" in tout for i in range(1, 301)), "aucun nom perdu"


class TestVisasDocling:
    def test_visas_en_liste(self, contexte):
        """5342 : Docling rend les visas en items de liste."""
        page = """LEPRESIDENT DE LA REPUBLIQUE,

- Vu la Constitution ;
- Vu le décret n° 2005/104 du 13 avril 2005 portant organisation du Ministère de l'Administration Territoriale et de la Décentralisation ;
- Vu le décret nº 2011/408 du 09 décembre 2011 portant organisation du Gouvernement,

## DECRETE :

Article 1er.- Sont, à compter de la date de signature du présent décret, nommés aux postes ci-après."""
        _, r = decoupe([page], contexte)
        assert r.chunks[0]["kind"] == "legal_basis"
        assert r.chunks[0]["embed"] is False
        assert r.citations == ["décret n° 2005/104", "décret n° 2011/408"]

    def test_plusieurs_visas_sur_une_ligne(self, contexte):
        visas = "Vu la loi n° 2019/024 du 24 décembre 2019 ; Vu le décret n° 2020/111 du 3 mars 2020 ;"
        r = refine([chunk("LEGAL_BASIS", visas)], contexte)
        assert r.citations == ["loi n° 2019/024", "décret n° 2020/111"]


class TestTaillesDocling:
    def test_article_court_mais_normatif_vectorise(self, contexte):
        r = refine([chunk("4", "Le décret n° 2001/041 du 19 février 2001 est abrogé.")], contexte)
        assert r.chunks[0]["embed"] is True

    def test_repli_court_reste_hors_index(self, contexte):
        r = refine([chunk("PARA_3", "Le décret n° 2001/041 est abrogé.")], contexte)
        assert r.chunks[0]["embed"] is False

    def test_long_article_sans_alineas_coupe_aux_paragraphes(self, contexte):
        """
        Un article de definitions de 18 000 caracteres restait d'un bloc :
        tronque a l'embedding, sa fin n'etait jamais vectorisee.
        """
        definitions = "\n\n".join(
            f"- terme{i} : " + "définition suffisamment longue pour peser dans la taille. " * 6
            for i in range(40)
        )
        r = refine([chunk("2", "Au sens du présent code, on entend par :\n\n" + definitions)], contexte)
        assert len(r.chunks) > 1
        assert all(len(c["content"]) <= 3000 for c in r.chunks)
        tout = "\n".join(c["content"] for c in r.chunks)
        assert all(f"terme{i} :" in tout for i in range(40))

    def test_alineas_courts_regroupes(self, contexte):
        contenu = "Le conseil :\n" + "\n".join(f"({i}) délibère sur le point {i} ;" for i in range(1, 9))
        contenu += "\n" + "Texte complémentaire. " * 150
        r = refine([chunk("7", contenu)], contexte)
        # Huit alineas d'une ligne ne font pas huit fragments
        assert all(c["kind"] != "fragment" for c in r.chunks)
        assert len(r.chunks) < 8


class TestPseudoNumeros:
    @pytest.mark.parametrize("numero", [
        "PREAMBULE", "LEGAL_BASIS", "SIGNATURE", "ANNEXE", "ANNEXE.2", "SECTION_1", "PARA_12", "para_3.1",
    ])
    def test_pseudo_numeros_reconnus(self, numero):
        from app.utils.chunk_refiner import est_pseudo_numero

        assert est_pseudo_numero(numero)

    @pytest.mark.parametrize("numero", ["1", "1er", "12.3", "L 94 septies", "QUATRE-VINGT-SIXIEME", "", None])
    def test_vrais_numeros_non_pris_pour_des_pseudo(self, numero):
        from app.utils.chunk_refiner import est_pseudo_numero

        assert not est_pseudo_numero(numero)


class TestTitresEtPromulgation:
    def test_chapeau_fait_de_titres_hors_index(self, contexte):
        titres = (
            "CHAPITRE II DES ORGANES DE PASSATION DES MARCHES PUBLICS\n\n"
            "SECTION I DES MAITRES D'OUVRAGE ET DES MAITRES D'OUVRAGE DELEGUES"
        )
        r = refine([chunk("SECTION_1", titres)], contexte)
        assert r.chunks[0]["embed"] is False
        assert r.chunks[0]["content"] == titres, "rien n'est supprime"

    def test_chapeau_avec_du_texte_reste_vectorise(self, contexte):
        texte = (
            "SECTION I DE L'ANALYSE DES RISQUES\n\nfilières, les espèces et les produits "
            "réglementés, afin de déterminer les mesures de prévention et de lutte appropriées."
        )
        r = refine([chunk("SECTION_1", texte)], contexte)
        assert r.chunks[0]["embed"] is True

    def test_formule_de_promulgation_hors_index(self, contexte):
        debut = (
            "2023/014\n\nLOI N°\n\nPORTANT CODE MINIER\n\n"
            "Le Parlement a délibéré et adopté, le Président de la République promulgue "
            "la loi dont la teneur suit :"
        )
        r = refine([chunk("LEGAL_BASIS", debut)], contexte)
        assert r.chunks[0]["kind"] == "boilerplate"
        assert r.chunks[0]["embed"] is False

    def test_titre_seul_en_tete_hors_index(self, contexte):
        r = refine([chunk("LEGAL_BASIS", "LOI N° 2025/006 REGISSANT LA BIOSECURITE AU CAMEROUN " * 3)], contexte)
        assert r.chunks[0]["embed"] is False


class TestArticlePremierIllisible:
    """
    Le marqueur « Article 1er » recouvert par le cachet : 53 documents du
    corpus n'avaient plus AUCUN vecteur.
    """

    def test_dispositif_detache_des_visas(self, contexte):
        visas = (
            "LE PRESIDENT DE LA REPUBLIQUE,\n\n- Vu la Constitution ;\n"
            "- Vu le décret n° 2011/408 du 09 décembre 2011 portant organisation du Gouvernement ;\n\n"
            "DECRETE:\n\n"
            "ARm o r a oun la journée du lundi 16 août 2021 est déclarée fériée sur toute l'étendue du territoire."
        )
        r = refine([chunk("LEGAL_BASIS", visas)], contexte)
        par_numero = {c["number"]: c for c in r.chunks}
        assert par_numero["LEGAL_BASIS"]["kind"] == "legal_basis"
        assert "fériée" not in par_numero["LEGAL_BASIS"]["content"]
        assert par_numero["DISPOSITIF"]["embed"] is True
        assert "fériée" in par_numero["DISPOSITIF"]["content"]
        assert r.citations == ["décret n° 2011/408"]

    def test_promulgation_d_une_loi(self, contexte):
        debut = (
            "Loi n°2014/008 du 18 juillet 2014 autorisant le Président de la République à ratifier l'accord.\n\n"
            "Le parlement a délibéré et adopté, le Président de la République promulgue la loi dont la teneur suit :\n\n"
            "Areee r e rs e n sur l'encouragement et la protection réciproques des investissements, signé le 24 janvier 2007."
        )
        r = refine([chunk("LEGAL_BASIS", debut)], contexte)
        par_numero = {c["number"]: c for c in r.chunks}
        assert par_numero["LEGAL_BASIS"]["kind"] == "boilerplate"
        assert par_numero["DISPOSITIF"]["embed"] is True

    def test_mot_decrete_ampute(self, contexte):
        visas = "- Vu la Constitution ;\n\nCRETE:\n\n" + "La convention de crédit conclue avec la banque est ratifiée. " * 2
        r = refine([chunk("LEGAL_BASIS", visas)], contexte)
        assert [c["number"] for c in r.chunks] == ["LEGAL_BASIS", "DISPOSITIF"]

    def test_sans_decrete_apres_le_dernier_visa(self, contexte):
        visas = (
            "- Vu la Constitution ;\n- Vu le décret n° 2019/043 du 05 février 2019 accordant délégation ;\n\n"
            "A d d d , les anciens Élèves-Inspecteurs de Police, dont la moyenne générale de notes obtenue "
            "à l'examen de fin de formation est inférieure à douze, sont nommés Inspecteurs de Police stagiaires."
        )
        r = refine([chunk("LEGAL_BASIS", visas)], contexte)
        assert [c["number"] for c in r.chunks] == ["LEGAL_BASIS", "DISPOSITIF"]

    def test_suite_de_visa_coupee_reste_dans_les_visas(self, contexte):
        visas = "- Vu la Constitution ;\n- Vu le décret n° 2011/408 du 09 décembre 2011\n\nportant organisation du Gouvernement ;"
        r = refine([chunk("LEGAL_BASIS", visas)], contexte)
        assert [c["number"] for c in r.chunks] == ["LEGAL_BASIS"]

    def test_visas_suivis_d_un_vrai_article(self, contexte):
        r = refine([chunk("LEGAL_BASIS", "- Vu la Constitution ;\n\nDECRETE :"), chunk("1", "Texte normatif de l'article premier.")], contexte)
        assert [c["number"] for c in r.chunks] == ["LEGAL_BASIS", "1"]

    def test_formule_collee_a_l_article_1(self, contexte):
        """Marqueur de l'article 2 illisible : la formule se colle a l'article 1."""
        contenu = (
            "La journée du lundi 16 août 2021 est déclarée fériée sur toute l'étendue du territoire "
            "de la République du Cameroun.\n\n"
            "AR e e e dre d'urgence, puis inséré au Journal Officiel en français et en anglais./-"
        )
        r = refine([chunk("1", contenu)], contexte)
        assert r.chunks[0]["kind"] == "article"
        assert r.chunks[0]["embed"] is True

    def test_formule_seule_reste_hors_index(self, contexte):
        contenu = (
            "Le présent décret sera enregistré, publié suivant la procédure d'urgence, puis inséré "
            "au Journal Officiel en français et en anglais./-\n\nCERTIFITDTRUECOPY"
        )
        r = refine([chunk("2", contenu)], contexte)
        assert r.chunks[0]["kind"] == "boilerplate"

    def test_dispositif_est_un_pseudo_numero(self):
        from app.utils.chunk_refiner import est_pseudo_numero

        assert est_pseudo_numero("DISPOSITIF") and est_pseudo_numero("DISPOSITIF.2")


@pytest.mark.parametrize("reste", [
    "Vu le décret n° 2011/045 du 08 mars 2011 portant organisation de l'Université de Bamenda,",
    "0 7 JUIN 2017\n\nPaix -Travail – Patrie",
])
def test_ce_qui_suit_decrete_sans_dispositif_reste_dans_les_visas(contexte, reste):
    r = refine([chunk("LEGAL_BASIS", "- Vu la Constitution ;\n\nDECRETE :\n\n" + reste)], contexte)
    assert [c["number"] for c in r.chunks] == ["LEGAL_BASIS"]


class TestListesEtTableauxBudgetaires:
    def test_intitule_de_programme_n_est_pas_un_texte_de_nomination(self, contexte):
        """« PROMOTION DE LA FEMME » dans une cellule ne fait pas une liste."""
        table = "| Programme | Responsable |\n|---|---|\n" + "".join(
            f"| PROMOTION DE LA FEMME {i} | DIRECTION GENERALE {i} |\n" for i in range(8)
        )
        r = refine([chunk("4", "Les programmes se déclinent comme suit :\n" + table)], contexte)
        assert not r.roster
        assert r.chunks[0]["kind"] != "roster"

    def test_tous_les_tableaux_d_une_liste_sortent_du_vecteur(self, contexte):
        """Un tableau par page ; la detection en manque un : il part quand meme."""
        page1 = "| 1. | NOM PRENOM01 | 701 895-X |\n|---|---|---|\n" + "".join(
            f"| {i}. | NOM PRENOM{i:02d} | 70{i} 89{i}-X |\n" for i in range(2, 9)
        )
        illisible = "| | AKONO AKONO MARTIN | J-057 035 |\n|---|---|---|\n| | ABAH OMGBA | J-057 036 |\n"
        contenu = (
            "Les Gardiens de la Paix dont les noms suivent sont inscrits au tableau d'avancement :\n"
            + page1 + "\n" + illisible
        )
        r = refine([chunk("1", contenu)], contexte)
        assert r.chunks[0]["kind"] == "roster"
        assert "AKONO" in r.chunks[0]["content"], "le contenu garde toute la liste"
        assert "AKONO" not in r.chunks[0]["embed_text"]
        assert "PRENOM05" not in r.chunks[0]["embed_text"]
