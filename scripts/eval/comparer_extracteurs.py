"""
Banc d'essai : Gemini contre DeepSeek sur l'extraction paginee d'un PDF.

POURQUOI CE SCRIPT EXISTE. Les billets qui vantent DeepSeek pour l'OCR mesurent
la densite et la justesse d'un tableau. JuriX ne depend pas de ca. Il depend du
CONTRAT DE PAGINATION decrit dans app/services/pdf_extraction_service.py :
`<<PAGE:n>>` par page, numerotation sur la page PHYSIQUE du PDF. C'est pour lui
que LlamaParse a ete abandonne, pas pour le prix — mesure a l'epoque : 271
pages de PDF pour 27 pages extraites, et 710 articles portant tous
`page_number = 1`. Aucun test publie sur DeepSeek ne mesure cela.

Le document de reference est le Code Minier (PRC-9866, 69 pages), celui-la meme
que cite la docstring de l'extracteur, avec ses chiffres : densite
2352 caracteres/page, un marqueur par page, lot 21-32 numerote exactement.

TROIS MESURES, dans l'ordre d'importance :

1. PAGINATION — chaque page demandee apparait une fois et une seule, dans
   l'ordre, numerotee sur la page physique. Eliminatoire : sans elle, les liens
   vers les articles pointent au hasard.
2. DENSITE — caracteres par page, contre la reference Gemini. Une densite
   effondree signale des pages avalees.
3. FIDELITE — la mesure qui compte pour un corpus juridique. Le PDF porte une
   couche texte, donc une verite terrain. On verifie que les numeros d'article
   rendus EXISTENT dans la page d'origine, et quelle proportion des mots longs
   produits s'y retrouve. C'est le defaut observe sur DeepSeek-Vision en
   conditions reelles : un mot insere qui n'etait pas dans l'image source. Sur
   une loi, un mot invente est pire qu'un mot manquant — rien ne le signale.

Usage :
    .venv/bin/python scripts/eval/comparer_extracteurs.py --moteur gemini
    .venv/bin/python scripts/eval/comparer_extracteurs.py --moteur deepseek
    .venv/bin/python scripts/eval/comparer_extracteurs.py --moteur tous

DeepSeek exige DEEPSEEK_API_KEY dans l'environnement ou le .env.
"""

import argparse
import asyncio
import base64
import io
import json
import os
import re
import sys
import time
import unicodedata
import warnings
from pathlib import Path
from typing import Dict, List

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PDF_PAR_DEFAUT = "/home/rahim/jurix/documents francais/9866_loi-n-2023-014-du-19-decembre-2023-portant-code-minier.pdf"

# Le lot que la docstring de l'extracteur donne comme « numerote 21..32
# exactement ». Le reprendre rend la comparaison directement lisible contre un
# chiffre deja mesure, au lieu d'en inventer un nouveau.
PREMIERE_PAGE = 21
DERNIERE_PAGE = 32

# Densite Gemini mesuree sur ce document, inscrite dans la docstring.
DENSITE_REFERENCE = 2352

MARQUEUR = re.compile(r"<<PAGE:(\d+)>>")
NUM_ARTICLE = re.compile(r"ARTICLE\s+(\d+)", re.IGNORECASE)

# Le contrat, mot pour mot, impose aux deux moteurs. DeepSeek ne peut etre juge
# sur la pagination que s'il recoit la meme consigne que Gemini.
CONSIGNE = """Transcris cette page de loi camerounaise en markdown, fidelement.

Regles :
- Rends UNIQUEMENT le contenu de la page, sans commentaire ni preambule.
- N'invente rien. Si un passage est illisible, ecris [illisible].
- Conserve les numeros d'article exactement tels qu'ils apparaissent.
- Conserve les tableaux sous forme de tableaux markdown.
"""


def normaliser(texte: str) -> str:
    """Minuscules sans accents : la couche texte du PDF accentue mal."""
    sans = unicodedata.normalize("NFKD", texte.lower())
    return "".join(c for c in sans if not unicodedata.combining(c))


def mots_longs(texte: str) -> set:
    """Mots de plus de 4 lettres : les courts sont du bruit de comparaison."""
    return {m for m in re.findall(r"[a-z]{5,}", normaliser(texte))}


def couche_texte(chemin: Path, premiere: int, derniere: int) -> Dict[int, str]:
    """Verite terrain : le texte deja porte par le PDF, page par page."""
    import pypdf

    lecteur = pypdf.PdfReader(str(chemin))
    return {
        n: (lecteur.pages[n - 1].extract_text() or "")
        for n in range(premiere, derniere + 1)
    }


# ==================== MOTEURS ====================

async def extraire_gemini(chemin: Path, premiere: int, derniere: int) -> Dict[int, str]:
    """Le chemin de PRODUCTION, sans doublure : c'est la reference."""
    from app.services.pdf_extraction_service import GeminiPdfExtractor

    pages = await GeminiPdfExtractor().extract_pages(chemin)
    return {n: pages[n - 1] for n in range(premiere, derniere + 1) if n <= len(pages)}


async def extraire_docling(chemin: Path, premiere: int, derniere: int) -> Dict[int, str]:
    """
    Docling, pipeline local. Ni cle, ni GPU, ni quota.

    DEUX REGLAGES NON NEGOCIABLES, tous deux contre-intuitifs :

    1. `traverse_pictures=True`. Le defaut est False, et sur un document SCANNE
       — un tiers du corpus JuriX — il rend un document ENTIEREMENT VIDE. Le
       ticket #2047 du depot le documente, rapporte par un utilisateur
       francophone. Sans ce drapeau, la mesure dirait « Docling ne lit rien »
       alors qu'on se serait tire une balle dans le pied.

    2. Le filtrage des pages blanches porte sur le MARKDOWN, jamais sur la
       chaine deja prefixee. `f"<<PAGE:{n}>>\n{md}".strip()` est toujours vrai,
       puisque le prefixe n'est jamais vide : le filtre ne filtrerait rien et
       une page blanche consommerait quand meme sa place.
    """
    from docling.document_converter import DocumentConverter

    def _convertir() -> Dict[int, str]:
        document = DocumentConverter().convert(str(chemin)).document
        pages: Dict[int, str] = {}
        for numero in range(premiere, derniere + 1):
            md = document.export_to_markdown(page_no=numero, traverse_pictures=True)
            if md and md.strip():
                pages[numero] = md
        return pages

    # Le pipeline est synchrone et tient le processeur : sans ce deport, il
    # gelerait la boucle d'evenements pendant toute la conversion.
    return await asyncio.to_thread(_convertir)


async def extraire_datalab(chemin: Path, premiere: int, derniere: int) -> Dict[int, str]:
    """
    Chandra 2 via l'API hebergee de Datalab.

    POURQUOI L'API ET NON LES POIDS. Chandra 2 fait 5,3 milliards de parametres,
    soit ~10,6 Go en BF16 : il ne rentre pas sur cette machine (11,8 Go de RAM,
    pas de GPU). L'API a de surcroit l'avantage d'echapper a la clause
    Share-a-Like du MODEL_LICENSE, qui pretend etendre la licence du modele
    « to the Output and any derivatives » — donc potentiellement au corpus
    extrait lui-meme.

    TROIS REGLAGES QUI DECIDENT DE LA VALIDITE DE LA MESURE :

    1. `page_range` est indexe a partir de ZERO. Verifie, pas suppose : une
       requete sur la page 20 rend « les modalites d'application des regimes
       juridique, fiscal, douanier », qui est le debut de la page PHYSIQUE 21.
       Un decalage d'une page ici fausserait tout le test de pagination.

    2. `keep_pageheader_in_output` et `keep_pagefooter_in_output`. Par defaut,
       Datalab JETTE ces blocs. Sur les decrets camerounais ils portent la
       reference au Journal Officiel et le folio imprime : sans ces drapeaux on
       mesurerait une omission qu'on se serait infligee soi-meme.

    3. `disable_image_captions` et `disable_image_extraction`. Les legendes sont
       du texte GENERE, absent de la page : elles tireraient l'ancrage lexical
       vers le bas et ressembleraient a de l'hallucination sans en etre. Et les
       chemins d'image injectent des mots (« Picture », « jpeg ») qui ne sont
       pas dans la source.

    `merge_cross_page` reste a False : il fusionne le contenu par-dessus les
    frontieres de page, c'est-a-dire qu'il detruit exactement ce qu'on mesure.
    """
    import httpx

    cle = os.environ.get("DATALAB_API_KEY", "")
    if not cle:
        raise SystemExit(
            "DATALAB_API_KEY absente. Poser la cle dans l'environnement :\n"
            "    export DATALAB_API_KEY=..."
        )

    config = {"keep_pageheader_in_output": True, "keep_pagefooter_in_output": True}
    async with httpx.AsyncClient(timeout=300) as client:
        with open(chemin, "rb") as fh:
            depot = await client.post(
                "https://www.datalab.to/api/v1/convert",
                headers={"X-API-Key": cle},
                files={"file": (chemin.name, fh.read(), "application/pdf")},
                data={
                    "mode": "balanced",
                    "output_format": "markdown",
                    # Indexation a partir de zero, d'ou le -1.
                    "page_range": f"{premiere - 1}-{derniere - 1}",
                    "paginate": "true",
                    "merge_cross_page": "false",
                    "disable_image_extraction": "true",
                    "disable_image_captions": "true",
                    "additional_config": json.dumps(config),
                },
            )
        depot.raise_for_status()
        depose = depot.json()
        if not depose.get("success"):
            raise SystemExit(f"Datalab a refuse le depot : {depose.get('error')}")

        url = depose["request_check_url"]
        for _ in range(100):
            await asyncio.sleep(3)
            etat = (await client.get(url, headers={"X-API-Key": cle})).json()
            if etat.get("status") == "complete":
                break
        else:
            raise SystemExit("Datalab : delai depasse")

    if not etat.get("success"):
        raise SystemExit(f"Datalab a echoue : {etat.get('error')}")

    cout = etat.get("total_cost")
    if cout is not None:
        print(f"  cout facture : {cout}", flush=True)

    # Le marqueur est `{n}------...`, avec n indexe a partir de zero. La
    # documentation le decrit comme un simple filet horizontal : c'est faux, il
    # porte le numero, et c'est lui qui rend la mesure possible.
    morceaux = re.split(r"\{(\d+)\}-{10,}", etat.get("markdown") or "")
    pages: Dict[int, str] = {}
    for i in range(1, len(morceaux) - 1, 2):
        physique = int(morceaux[i]) + 1
        contenu = morceaux[i + 1]
        if contenu.strip():
            pages[physique] = contenu
    return pages


def _page_en_png(chemin: Path, numero: int, dpi: int = 200) -> bytes:
    from pdf2image import convert_from_path

    image = convert_from_path(str(chemin), dpi=dpi, first_page=numero, last_page=numero)[0]
    tampon = io.BytesIO()
    image.save(tampon, format="PNG")
    return tampon.getvalue()


async def extraire_deepseek(chemin: Path, premiere: int, derniere: int) -> Dict[int, str]:
    """
    DeepSeek ne lit pas un PDF : il lit des IMAGES.

    Difference qui n'est pas un detail. Gemini recoit le PDF natif, avec sa
    couche texte ; DeepSeek recoit une rasterisation, et plafonne chaque image
    a 1024 jetons. Sur une page de loi dense — 2000 a 2400 caracteres — c'est
    la premiere chose a surveiller.
    """
    import httpx

    cle = os.environ.get("DEEPSEEK_API_KEY", "")
    if not cle:
        raise SystemExit(
            "DEEPSEEK_API_KEY absente. Poser la cle dans l'environnement :\n"
            "    export DEEPSEEK_API_KEY=sk-...\n"
            "Le compte se cree sur https://platform.deepseek.com/"
        )

    pages: Dict[int, str] = {}
    async with httpx.AsyncClient(timeout=180) as client:
        for numero in range(premiere, derniere + 1):
            png = await asyncio.to_thread(_page_en_png, chemin, numero)
            b64 = base64.b64encode(png).decode()
            reponse = await client.post(
                "https://api.deepseek.com/chat/completions",
                headers={"Authorization": f"Bearer {cle}"},
                json={
                    "model": "deepseek-flash",
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": CONSIGNE},
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        ],
                    }],
                    "temperature": 0.0,
                    "max_tokens": 4096,
                },
            )
            if reponse.status_code >= 400:
                raise SystemExit(
                    f"DeepSeek a refuse la page {numero} ({reponse.status_code}) : "
                    f"{reponse.text[:200]}"
                )
            pages[numero] = reponse.json()["choices"][0]["message"]["content"]
            print(f"  page {numero} extraite", flush=True)
    return pages


# ==================== MESURES ====================

def mesurer(nom: str, pages: Dict[int, str], verite: Dict[int, str], duree: float) -> dict:
    attendues = list(range(PREMIERE_PAGE, DERNIERE_PAGE + 1))
    rendues = sorted(pages)

    # 1. Pagination — le critere eliminatoire.
    pagination_exacte = rendues == attendues

    # 2. Densite.
    total = sum(len(p) for p in pages.values())
    densite = total // max(len(pages), 1)

    # 3. Fidelite.
    articles_inventes: List[str] = []
    taux_par_page: List[float] = []
    for numero, rendu in pages.items():
        source = verite.get(numero, "")
        source_norm = normaliser(source)

        for num in NUM_ARTICLE.findall(rendu):
            if f"article {num}" not in source_norm and f"article  {num}" not in source_norm:
                articles_inventes.append(f"p.{numero} ART.{num}")

        mots_source = mots_longs(source)
        mots_rendus = mots_longs(rendu)
        if mots_rendus and mots_source:
            taux_par_page.append(len(mots_rendus & mots_source) / len(mots_rendus))

    ancrage = sum(taux_par_page) / len(taux_par_page) if taux_par_page else 0.0

    return {
        "moteur": nom,
        "pages_attendues": len(attendues),
        "pages_rendues": len(rendues),
        "pagination_exacte": pagination_exacte,
        "densite_car_par_page": densite,
        "densite_reference": DENSITE_REFERENCE,
        "articles_inventes": articles_inventes,
        "ancrage_lexical": round(ancrage, 4),
        "duree_s": round(duree, 1),
    }


def afficher(m: dict) -> None:
    print(f"\n=== {m['moteur']} ===")
    ok = "OK" if m["pagination_exacte"] else "ECHEC"
    print(f"  pagination {PREMIERE_PAGE}-{DERNIERE_PAGE} : {ok} "
          f"({m['pages_rendues']}/{m['pages_attendues']} pages)")
    ecart = m["densite_car_par_page"] - m["densite_reference"]
    print(f"  densite               : {m['densite_car_par_page']} car./page "
          f"(reference {m['densite_reference']}, ecart {ecart:+d})")
    print(f"  ancrage lexical       : {m['ancrage_lexical']:.1%} des mots longs "
          f"retrouves dans la couche texte du PDF")
    if m["articles_inventes"]:
        print(f"  ARTICLES INVENTES     : {len(m['articles_inventes'])} -> "
              f"{', '.join(m['articles_inventes'][:8])}")
    else:
        print("  articles inventes     : aucun")
    print(f"  duree                 : {m['duree_s']} s")


async def main() -> None:
    parseur = argparse.ArgumentParser(description=__doc__)
    parseur.add_argument(
        "--moteur",
        choices=["gemini", "deepseek", "docling", "datalab", "tous"],
        default="tous",
    )
    parseur.add_argument("--pdf", default=PDF_PAR_DEFAUT)
    parseur.add_argument("--sortie", default="/tmp/comparaison_extracteurs.json")
    args = parseur.parse_args()

    chemin = Path(args.pdf)
    if not chemin.exists():
        raise SystemExit(f"PDF introuvable : {chemin}")

    print(f"Document : {chemin.name}")
    print(f"Pages    : {PREMIERE_PAGE} a {DERNIERE_PAGE}")
    verite = couche_texte(chemin, PREMIERE_PAGE, DERNIERE_PAGE)
    print(f"Verite terrain : couche texte du PDF, "
          f"{sum(len(t) for t in verite.values()) // len(verite)} car./page en moyenne")

    moteurs = (
        ["gemini", "docling", "datalab"] if args.moteur == "tous" else [args.moteur]
    )
    mesures = []
    for nom in moteurs:
        print(f"\nExtraction {nom}...", flush=True)
        debut = time.time()
        try:
            fn = {
                "gemini": extraire_gemini,
                "docling": extraire_docling,
                "deepseek": extraire_deepseek,
                "datalab": extraire_datalab,
            }[nom]
            pages = await fn(chemin, PREMIERE_PAGE, DERNIERE_PAGE)
        except SystemExit as e:
            print(f"  ignore : {e}")
            continue
        mesure = mesurer(nom, pages, verite, time.time() - debut)
        mesures.append(mesure)
        afficher(mesure)

    Path(args.sortie).write_text(json.dumps(mesures, indent=2, ensure_ascii=False))
    print(f"\nMesures ecrites dans {args.sortie}")


if __name__ == "__main__":
    asyncio.run(main())
