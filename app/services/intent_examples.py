"""
Banque d'exemples du classement local de l'intention (intent_local.py).

Chaque message de l'utilisateur est rapproche de ces exemples par
EmbeddingGemma ; la categorie dont les exemples les plus proches l'emportent
donne le verdict. Leur qualite fait celle du classement :

- les questions de droit sont surtout en MOTS DE TOUS LES JOURS (« mon patron
  ne me paie plus »), pas seulement en jargon : c'est ainsi qu'on les pose ;
- le corpus est fait pour moitie de nominations et d'avancements : un nom de
  personne, un concours, une promotion sont des questions de droit ;
- la politesse d'ouverture ne fait pas une salutation : « bonjour, puis-je
  divorcer sans avocat ? » est une question de droit.

NE PAS REUTILISER CES PHRASES DANS LE JEU D'EVALUATION
(tests/fixtures/intent_eval.json) : on mesurerait la memoire de la banque, pas
le classement.
"""

from typing import Dict, Tuple

EXEMPLES: Dict[str, Tuple[str, ...]] = {
    "juridique": (
        # Travail
        "mon patron ne me paie plus depuis trois mois, que faire ?",
        "combien de jours de congé payé ai-je droit par an ?",
        "peut-on me licencier pendant mon congé maternité ?",
        "quel est le salaire minimum au Cameroun ?",
        "mon employeur refuse de me déclarer à la CNPS",
        "quelle indemnité en cas de licenciement abusif ?",
        # Famille, état civil, successions
        "comment divorcer quand mon mari refuse ?",
        "qui garde les enfants après une séparation ?",
        "comment obtenir un acte de naissance quand on n'a pas été déclaré ?",
        "mon père est décédé sans testament, comment partager ses biens ?",
        "à quel âge peut-on se marier ?",
        "comment obtenir la nationalité camerounaise ?",
        # Foncier, logement
        "on veut m'expulser de la maison que je loue, en ai-je le droit ?",
        "comment obtenir un titre foncier ?",
        "mon voisin construit sur mon terrain",
        "le propriétaire peut-il augmenter le loyer comme il veut ?",
        # Pénal
        "quelle peine pour un vol simple ?",
        "que risque-t-on pour diffamation sur les réseaux sociaux ?",
        "combien de temps peut durer une garde à vue ?",
        "le harcèlement au travail est-il puni ?",
        "porter plainte contre un policier qui m'a frappé",
        # Entreprises, commerce, impôts
        "comment créer une SARL ?",
        "quels impôts paie une petite entreprise ?",
        "quel est le taux de la TVA ?",
        "comment contester un redressement fiscal ?",
        "quelles sont les obligations du gérant d'une société ?",
        # Administration, marchés, secteurs
        "comment soumissionner à un marché public ?",
        "quelles conditions pour obtenir un permis de recherche minière ?",
        "qui délivre le permis de conduire et à quelles conditions ?",
        "que prévoit la loi sur la protection de l'environnement ?",
        "comment fonctionnent les communes et les régions ?",
        "que prévoit la loi de finances pour cette année ?",
        # Permis, interdits : « a-t-on le droit de... » est une question de droit
        "est-il interdit de fumer dans un lieu public ?",
        "a-t-on le droit de vendre de l'alcool à un mineur ?",
        "est-ce autorisé de construire sans permis de bâtir ?",
        "ai-je le droit d'enregistrer une conversation à l'insu de quelqu'un ?",
        "la pêche est-elle permise dans les aires protégées ?",
        "un étranger peut-il acheter un terrain ?",
        # Textes, articles, références
        "que dit l'article 33 du code minier ?",
        "explique-moi le décret 2015/394",
        "quelles lois parlent de la cybercriminalité ?",
        "y a-t-il un texte sur les associations religieuses ?",
        "quand ce décret entre-t-il en vigueur ?",
        # Nominations, concours, carrières (moitié du corpus)
        "qui a été nommé directeur général de la CAMTEL ?",
        "NGONO Marie fait-elle partie des inspecteurs promus ?",
        "liste des admis au concours de l'école de police",
        "avancement de grade des gardiens de la paix en 2023",
        "qui est le préfet du Mfoundi ?",
        "ABENA ESSOMBA Paul a-t-il été nommé ?",
        # Politesse d'ouverture + vraie question
        "bonjour, puis-je divorcer sans avocat ?",
        "salut, je voudrais savoir comment créer une entreprise",
        "merci, et pour un mineur c'est quoi la peine ?",
        # Suites de questions
        "et pour une SARL ?",
        "et l'article suivant ?",
        "c'est valable combien de temps ?",
        # Anglais
        "can my employer fire me without notice?",
        "how do I register a company in Cameroon?",
        "what is the penalty for corruption?",
        "who was appointed minister of finance?",
        "what does section 12 of the penal code say?",
    ),
    "smalltalk": (
        "bonjour, comment ça va ?",
        "salut, tu vas bien aujourd'hui ?",
        "coucou",
        "bonsoir à toi",
        "merci pour ton aide",
        "merci beaucoup, c'est très clair",
        "super, merci !",
        "ok d'accord, j'ai compris",
        "tu es génial",
        "bonne nuit",
        "à demain",
        "je reviens plus tard",
        "ça marche, merci",
        "bonne journée à toi aussi",
        "comment tu te sens ?",
        "tu as passé une bonne journée ?",
        "hello, how are you?",
        "thanks a lot",
        "good morning",
        "see you later",
    ),
    "meta": (
        "qui es-tu ?",
        "tu es une intelligence artificielle ?",
        "que sais-tu faire ?",
        "d'où viennent tes informations ?",
        "quelles lois connais-tu ?",
        "est-ce que tes réponses sont fiables ?",
        "es-tu un avocat ?",
        "comment fonctionnes-tu ?",
        "tes textes sont-ils à jour ?",
        "qui t'a créé ?",
        "est-ce que c'est gratuit ?",
        "peux-tu remplacer un avocat ?",
        "tu parles anglais ?",
        "comment utiliser cette application ?",
        "est-ce que mes questions sont enregistrées ?",
        "what can you do?",
        "who made you?",
        "where does your information come from?",
    ),
    "hors_sujet": (
        "combien font 15 fois 12 ?",
        "écris-moi un poème sur la pluie",
        "quel temps fera-t-il demain à Douala ?",
        "donne-moi la recette du ndolé",
        "qui a gagné la coupe d'Afrique des nations ?",
        "raconte-moi une blague",
        "comment soigner le paludisme ?",
        "quelle est la capitale du Canada ?",
        "aide-moi à écrire un programme en Python",
        "traduis cette phrase en espagnol",
        "quel est le meilleur téléphone à acheter ?",
        "explique-moi la photosynthèse",
        "qui est le meilleur footballeur du monde ?",
        "comment perdre du poids rapidement ?",
        "résume-moi le dernier film de Marvel",
        "what is the weather like today?",
        "write me a love song",
        "how far is the moon?",
    ),
}
