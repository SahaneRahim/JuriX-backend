"""
Texte natif de la loi 92/007 (Code du travail), lu dans la couche Word du PDF.

Le texte en base etait un OCR bruite (« verser a lu se e e o n u ne out,l
nature ») ; le PDF porte une vraie couche de texte. Usage :

    python extraire_loi_1992_007.py loi.pdf loi-1992-007-texte-natif.txt
"""

import re
import subprocess
import sys

DEBUT_DE_PARAGRAPHE = re.compile(
    r"^(?:[-•]|\(\d{1,2}\)|Article\s+\d|Art\.|TITRE\b|CHAPITRE\b|Section\s+[IVX\d]|[a-z]\)\s)"
)


def page(pdf: str, numero: int) -> str:
    brut = subprocess.run(
        ["pdftotext", "-f", str(numero), "-l", str(numero), pdf, "-"],
        capture_output=True, text=True, check=True,
    ).stdout.replace("\f", "")
    paragraphes = []
    for bloc in re.split(r"\n\s*\n", brut):
        courant = ""
        for ligne in (re.sub(r"\s+", " ", x).strip() for x in bloc.split("\n")):
            if not ligne:
                continue
            ligne = re.sub(r"^•\s*", "- ", ligne)
            if courant and DEBUT_DE_PARAGRAPHE.match(ligne):
                paragraphes.append(courant)
                courant = ligne
            else:
                courant = (courant + " " + ligne).strip()
        if courant:
            paragraphes.append(courant)
    return f"<<PAGE:{numero}>>\n" + "\n\n".join(paragraphes)


def main(pdf: str, sortie: str, pages: int = 43) -> None:
    texte = "\n\n".join(page(pdf, n) for n in range(1, pages + 1)) + "\n"
    texte = re.sub(r"(?m)^-[ \t]*\n+(?=\S)", "- ", texte)
    with open(sortie, "w", encoding="utf-8") as f:
        f.write(texte)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
