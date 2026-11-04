"""
Questions en langage courant, relances et questions qui nomment un texte,
pour scripts/eval/evaluer_contexte_chat.py.

D'ou elles viennent. Les questions francaises de tous les jours, les relances
et les questions nommees sont celles de la revue du 29/10/2026 (revue de la
recuperation, tour 2 : data/fiabilite/tmp/revue_recuperation/questions_*.json),
reprises telles quelles, fautes et pidgin compris : c'est la forme des
questions d'un vrai public, celle que le jeu relu (des questions formelles)
ne contient pas. Les questions anglaises de tous les jours (20) ont ete
ecrites ici, sur les memes sujets, pour que le seuil anglais soit calibre sur
autre chose que des traductions formelles.

Aucune n'a de texte attendu, sauf les questions nommees et les relances en
chaine : elles servent a compter les refus de la barriere (une question
legitime refusee est une erreur), les tableaux de programmes lus et les
articles nommes ajoutes. Les montants « prévus » et les questions sur le
budget (revue du 29/10/2026, tour 3) mesurent le detecteur des questions
budgetaires : aucun tableau AE/CP pour les premiers, les tableaux pour les
secondes.

Revue du tour 4 (30/10/2026) : les changements de sujet courts (« Et la
nationalité ? », avec le texte du nouveau sujet attendu), les secondes
chances (« la dot », refusee seule : refus ou contexte qui porte ses mots),
les articles nommes avec un renvoi au texte de la conversation (« l'article
40 du même code », dans les relances en chaine) et les budgets prives avec une
annee (dans les montants prevus).
"""

from typing import Dict, List, Optional, Tuple

# ==================== QUESTIONS DE TOUS LES JOURS ====================

REELLES: Dict[str, List[str]] = {
    "fr": [
        "Mon patron m'a renvoyé sans raison après 5 ans de travail, est-ce qu'il doit me payer "
        "quelque chose ?",
        "j'ai été licencié sans préavis que faire",
        "mon propriétaire veut me chasser de la maison sans me prévenir, il a le droit ?",
        "La police m'a gardé 3 jours sans voir un juge, c'est légal ?",
        "Mon mari est décédé, ses frères veulent prendre la maison, qu'est-ce que je peux faire ?",
        "Combien coûte la création d'une SARL au Cameroun ?",
        "Est-ce que je peux divorcer si mon mari me frappe ?",
        "kel est la peine pour vol de téléphone",
        "Quelles sont les conditions pour obtenir la nationalité camerounaise ?",
        "Un enfant né hors mariage peut-il hériter de son père ?",
        "Mon employeur ne me déclare pas à la CNPS, que faire ?",
        "Je veux créer mon entreprise, quelles sont les étapes ?",
        "Quel est l'âge légal pour se marier au Cameroun ?",
        "Peut-on être emprisonné pour une dette ?",
        "Le chef de quartier peut-il vendre un terrain ?",
        "Comment obtenir un titre foncier ?",
        "Que risque quelqu'un qui conduit sans permis ?",
        "Mon voisin a construit sur mon terrain, quels recours ?",
        "Est-ce que l'avortement est autorisé au Cameroun ?",
        "Quelle est la durée du congé de maternité ?",
        "Combien d'heures de travail par semaine maximum ?",
        "Le SMIG au Cameroun c'est combien ?",
        "Un fonctionnaire peut-il faire du commerce ?",
        "Quelles sont les sanctions pour corruption ?",
        "Comment porter plainte contre un policier ?",
        "La polygamie est-elle légale au Cameroun ?",
        "Quels documents pour un acte de naissance tardif ?",
        "J'ai acheté un terrain sans papier, est-ce valable ?",
        "Quel est le délai de prescription pour un vol ?",
        "Mon fils mineur a été arrêté, que dit la loi ?",
        "Que dit la loi sur l'homosexualité au Cameroun ?",
        "Combien de temps pour contester un licenciement ?",
        "Peut-on licencier une femme enceinte ?",
        "c koi la diffamation",
        "Les droits du locataire au Cameroun",
        "Comment se passe une succession sans testament ?",
        "Quelle taxe pour importer une voiture d'occasion ?",
        "Mon banquier a bloqué mon compte sans explication",
        "Quelles sont les règles pour ouvrir un débit de boissons ?",
        "quelles indemnites en cas de licenciement abusif",
        "duree maximale de la garde a vue",
        "que risque un voleur de telephone",
        "indemnite licenciement abusif",
        "mon employeur ne paie pas mon salaire depuis 3 mois",
        "Un policier peut-il fouiller mon téléphone ?",
        "Est-ce qu'un étranger peut acheter un terrain au Cameroun ?",
        "Que faire si on m'a volé ma moto ?",
        "Comment récupérer une dette d'un ami qui refuse de payer ?",
        "Quels sont les droits d'un enfant au Cameroun ?",
        "Je suis enceinte et mon copain refuse de reconnaître l'enfant",
        "Mon chef de village me demande de l'argent pour régler un litige, c'est normal ?",
        "Un chef traditionnel peut-il juger une affaire de vol ?",
        "Est-ce que la dot est obligatoire ?",
        "Un pasteur peut-il célébrer un mariage civil ?",
        "La sorcellerie est-elle punie par la loi ?",
        "Puis-je enregistrer ma tontine ?",
        "Mon frère a vendu le champ familial sans mon accord",
        "Le bailleur garde ma caution",
        "Les moto-taxis doivent-ils avoir une assurance ?",
        "Je travaille comme ménagère sans contrat, quels sont mes droits ?",
        "Est-ce que je peux enregistrer mon enfant sans le père ?",
        "Le maire refuse de célébrer notre mariage",
        "Le commissaire garde ma moto depuis 2 mois",
        "Ma femme est partie avec les enfants au village",
        "Le propriétaire a coupé l'électricité",
        "On m'accuse de sorcellerie au village",
        "J'ai reçu une convocation à la gendarmerie, dois-je y aller ?",
        "Un huissier est venu saisir mes meubles",
        "Mon voisin fait du bruit toute la nuit",
        "Ma belle-famille veut récupérer la maison après le décès de mon mari",
        "Le gérant de la boutique est parti avec la caisse",
        "Mon associé a vidé le compte de la société",
        "Le chauffeur de bus m'a laissé en route",
        "On m'a vendu une voiture volée",
        "Je n'ai pas reçu mon salaire de fonctionnaire depuis mon intégration",
        "Mon enfant a été renvoyé de l'école pour non paiement",
        "Le médecin a refusé de soigner ma mère aux urgences",
        "Je veux changer mon nom de famille",
        "Mon acte de naissance comporte une erreur",
        "Un ami refuse de me rendre mon argent prêté sans reçu",
        "Mon mari a pris une deuxième femme sans me demander",
        "Ma fille de 15 ans est enceinte d'un adulte, que faire ?",
        "Mes enfants ne veulent plus s'occuper de moi",
        "Le père de mon enfant ne donne rien pour lui",
        "Ma tante veut vendre le terrain de mon grand-père",
        "On a détruit ma maison pour construire une route",
        "Je veux louer ma chambre à un étudiant",
        "Mon patron me fait travailler le dimanche sans payer plus",
        "J'ai été blessé sur un chantier",
        "On me demande de payer pour avoir un emploi à la mairie",
        "Le policier m'a demandé de l'argent au contrôle",
        "Mon téléphone a été volé dans le taxi",
        "Quelqu'un a publié mes photos sur Facebook",
        "On m'insulte sur WhatsApp",
        "Mon client refuse de payer la marchandise livrée",
        "Je veux ouvrir une boutique de vente de médicaments",
        "Le fisc me réclame des impôts que j'ai déjà payés",
        "Je n'ai pas de carte d'identité, comment voter ?",
        "Mon fils a été enrôlé de force par des séparatistes",
        "Mon voisin a coupé mes arbres",
        "La mairie a fermé mon commerce",
        "Mon camion a été saisi à la douane",
        "Je suis réfugié centrafricain, quels sont mes droits ?",
        "Mon enfant est né à la maison, comment le déclarer ?",
        "Je veux adopter l'enfant de ma sœur",
        "Ma pension de veuve n'est pas payée",
        "Mon employeur m'a muté loin de ma famille",
        "Un ami a utilisé mon nom pour emprunter de l'argent",
        "Mon frère m'a frappé et blessé",
        "Le chef de famille refuse que les filles héritent",
    ],
    "en": [
        # Revue du 29/10/2026, pidgin compris.
        "My boss fired me without notice, what can I do?",
        "How long can police keep me in custody?",
        "Can my landlord evict me without notice?",
        "Is polygamy legal in Cameroon?",
        "What is the minimum wage in Cameroon?",
        "How do I get a land title?",
        "Ma boss don sack me without reason, wetin I fit do?",
        "Police don catch my broda since three days, dem fit hold am so?",
        "Can a foreigner buy land in Cameroon?",
        "What is the punishment for stealing a phone?",
        # Ecrites ici, sur les memes sujets.
        "My employer has not paid my salary for three months, what can I do?",
        "Can my husband take a second wife without my consent?",
        "My neighbour built a house on my land, what are my options?",
        "The police kept me for three days without seeing a judge, is that legal?",
        "How do I register my company in Cameroon?",
        "What happens if I drive without a licence?",
        "Can a pregnant woman be dismissed from her job?",
        "How long is maternity leave in Cameroon?",
        "Is abortion legal in Cameroon?",
        "How do I get a divorce if my husband beats me?",
        "Who inherits when someone dies without a will?",
        "Can I go to prison for not paying a debt?",
        "What are the penalties for corruption?",
        "How do I file a complaint against a police officer?",
        "Someone posted my photos on Facebook without my permission",
        "My landlord refuses to give back my deposit",
        "My child was born at home, how do I register the birth?",
        "Can a traditional chief sell land?",
        "What is the legal age of marriage in Cameroon?",
        "Is homosexuality punished in Cameroon?",
    ],
}

# Questions sur des montants, ou les tableaux de programmes des lois de
# finances sortaient en tete (revue du 29/10/2026), et deux vraies questions
# sur le budget, ou ils doivent rester.
MONTANTS: Dict[str, List[str]] = {
    "fr": [
        "Quel est le salaire minimum au Cameroun ?",
        "Combien gagne un fonctionnaire au Cameroun ?",
        "Quel est le montant des allocations familiales ?",
        "Quel est le taux de cotisation à la CNPS ?",
        "Quel est l'âge de départ à la retraite ?",
        "Quel est le montant minimum de la pension de retraite ?",
        "Combien de jours de congés payés par an ?",
        "Combien coûte un passeport au Cameroun ?",
        "Combien coûte la carte nationale d'identité ?",
        "Quel est le taux de TVA au Cameroun ?",
        "Quel est le salaire d'un enseignant du primaire ?",
        "Quelle est la prime de transport d'un salarié ?",
        "Combien coûte le permis de conduire ?",
        "Quel est le montant de la bourse d'un étudiant ?",
        "Quelles indemnités pour un accident de travail ?",
    ],
    "en": [
        "How much does a passport cost in Cameroon?",
        "What is the VAT rate in Cameroon?",
        "How much is the family allowance?",
    ],
}
# Les vraies questions sur le budget, avec le vocabulaire du budget
# (search_service.question_budgetaire) : les tableaux de programmes doivent
# rester.
BUDGETAIRES: Dict[str, List[str]] = {
    "fr": [
        "Quel est le budget du ministère de la santé dans la loi de finances 2023 ?",
        "Quels crédits de paiement la loi de finances 2024 ouvre-t-elle pour la prévention de "
        "la maladie ?",
        "Quelles autorisations d'engagement la loi de finances 2025 prévoit-elle pour le "
        "ministère du travail ?",
        "Quel est le budget de l'État pour 2025 ?",
        "Combien l'État camerounais dépense-t-il en 2025 pour aider les PME ?",
        "Quelle dotation budgétaire pour la protection du travail en 2023 ?",
    ],
    "en": [
        "What budget does the 2024 finance law allocate to disease prevention?",
        "What payment appropriations does the 2025 finance law grant for labour protection?",
        "What is the state budget for 2025?",
    ],
}

# Des questions de DROIT sur un montant « prévu », « alloué », un « budget »
# ou des « crédits » prives, une « dotation » comptable : la revue du
# 29/10/2026 (tour 3) les a trouvees prises pour budgetaires, et le modele
# lisait des tableaux AE/CP (8 sur 8 pour le SMIG). Aucune ne doit en lire.
MONTANTS_PREVUS: Dict[str, List[str]] = {
    "fr": [
        "Quel est le montant de l'amende prévue pour le vol ?",
        "Le montant prévu de la pension de veuve",
        "Quel est le montant de la caution prévu par le bail ?",
        "Mon budget est serré, puis-je payer mon loyer en retard ?",
        "Les crédits ouverts par ma banque",
        "Comment calculer la dotation aux amortissements selon l'OHADA ?",
        "Quel est le montant prévu par la loi pour le SMIG ?",
        "Quel est le montant prévu du SMIG au Cameroun ?",
        "L'État a-t-il prévu une augmentation du SMIG ?",
        "Quel est le montant prévu pour l'indemnité de licenciement ?",
        "Quelle somme est prévue en cas de licenciement abusif ?",
        "Quel montant est prévu en cas de licenciement abusif ?",
        "Quel montant est prévu pour les allocations familiales ?",
        "Qu'est-ce que l'État a prévu pour les travailleurs licenciés ?",
        "Qu'est-ce que l'État a prévu pour les victimes d'accidents du travail ?",
        "L'État est-il condamné aux dépens ?",
        # Un budget PRIVE avec une annee (revue du tour 4, sonde_budget_annee) :
        # la regle « budget + une annee » du tour 4 les prenait pour le budget
        # de l'Etat, et quatre lisaient un tableau AE/CP (« Quel budget pour
        # construire une maison en 2025 ? » : PRC-10557 art. 95.73, au rang 3).
        # Regle retiree : aucun tableau.
        "Quel budget prévoir pour créer une SARL en 2025 ?",
        "J'ai un budget de 2000 euros pour acheter un terrain, quelles démarches ?",
        "Quel budget pour un mariage en 2024 ?",
        "Comment gérer le budget familial en 2024 ?",
        "Le budget de l'association pour 2024 doit-il être voté par l'assemblée générale ?",
        "Quel budget pour un avocat en 2025 ?",
        "Quel budget pour construire une maison en 2025 ?",
        "Quel budget pour ouvrir un restaurant à Douala en 2025 ?",
        "Quel budget faut-il pour immatriculer une entreprise en 2024 ?",
        "budget prévisionnel 2025 d'une SARL",
    ],
    "en": [
        "How much compensation is allocated for unfair dismissal?",
        "What amount does the state allocate to the unemployed?",
        "How much is set aside for severance pay?",
        "Is the appropriation of a lost object theft?",
        "What budget do I need to start a business in 2025?",
    ],
}

# ==================== RELANCES ====================

# (question precedente, relance) : la relance est d'abord jugee seule, puis,
# refusee, jointe a la question precedente (RAGService._chercher_la_relance).
_PRECEDENTE_FR = "Quelles indemnités en cas de licenciement abusif ?"
_PRECEDENTE_EN = "What compensation is due for unfair dismissal?"
RELANCES_AVEC_HISTORIQUE: Dict[str, List[Tuple[str, str]]] = {
    "fr": [(_PRECEDENTE_FR, r) for r in (
        "Peux-tu détailler ?",
        "Explique plus simplement s'il te plaît",
        "Donne-moi un exemple concret",
        "Et si l'employeur refuse ?",
        "Combien de temps ?",
        "C'est quoi la différence ?",
        "Que dois-je faire maintenant ?",
        "Quel article exactement ?",
        "Peux-tu citer l'article ?",
        "Résume ta réponse",
        "Et dans mon cas ?",
        "Est-ce que c'est toujours en vigueur ?",
        "Quelle est la procédure à suivre ?",
        "Et la sanction ?",
        "Et si je n'ai pas de contrat écrit ?",
        "Peux-tu reformuler ?",
        "Qui dois-je contacter ?",
        "Combien ça coûte ?",
        "Que dit l'article suivant ?",
        "Et l'article 38 ?",
        "Et l'article 12 ?",
        "Et les indemnités ?",
        "Et le préavis ?",
        "Et la pension ?",
        "Et les impôts ?",
        "Et le salaire ?",
        "Et la prime ?",
        "Et les frais ?",
        "Et la retraite ?",
        "Et pour les dommages ?",
        "Et le montant ?",
    )],
    "en": [(_PRECEDENTE_EN, r) for r in (
        "Can you explain more?",
        "What should I do now?",
        "Give me an example",
        "Which article exactly?",
        "And what about the notice period?",
        "Is it still in force?",
    )],
}

# Des relances DE SUITE, et des articles nommes sans leur texte (revue du
# 29/10/2026, tour 3) : (questions precedentes de l'utilisateur, relance,
# textes dont au moins un extrait doit etre lu). La seconde relance etait
# jointe a la premiere et refusee ; « Et l'article 38 ? » lisait l'article 38
# de huit textes sans rapport. La reponse precedente nomme son texte, comme
# celles du modele.
_REPONSE_FR = (
    "En cas de licenciement abusif, l'article 39 du Code du travail prévoit des "
    "dommages-intérêts, en plus de l'indemnité de préavis et de l'indemnité de licenciement."
)
_REPONSE_EN = (
    "Under the Labour Code, an unfair dismissal gives a right to damages, in addition to the "
    "notice and severance allowances."
)
_CODE_DU_TRAVAIL = ("LOI-1992-007",)
RELANCES_EN_CHAINE: Dict[str, List[Tuple[Tuple[str, ...], str, Tuple[str, ...]]]] = {
    "fr": [
        ((_PRECEDENTE_FR,), "Peux-tu détailler ?", _CODE_DU_TRAVAIL),
        ((_PRECEDENTE_FR, "Peux-tu détailler ?"), "Donne-moi un exemple concret", _CODE_DU_TRAVAIL),
        ((_PRECEDENTE_FR, "Peux-tu détailler ?"), "Explique plus simplement s'il te plaît",
         _CODE_DU_TRAVAIL),
        ((_PRECEDENTE_FR, "Explique plus simplement s'il te plaît"), "Peux-tu détailler ?",
         _CODE_DU_TRAVAIL),
        ((_PRECEDENTE_FR, "Donne-moi un exemple concret"), "Peux-tu détailler ?", _CODE_DU_TRAVAIL),
        ((_PRECEDENTE_FR, "Peux-tu détailler ?", "Donne-moi un exemple concret"),
         "Et l'article 38 ?", _CODE_DU_TRAVAIL),
        ((_PRECEDENTE_FR, "Et pour un CDD ?"), "Peux-tu détailler ?", _CODE_DU_TRAVAIL),
        ((_PRECEDENTE_FR,), "Et l'article 38 ?", _CODE_DU_TRAVAIL),
        ((_PRECEDENTE_FR,), "Quel article exactement ?", _CODE_DU_TRAVAIL),
        ((_PRECEDENTE_FR,), "Combien ça coûte ?", _CODE_DU_TRAVAIL),
        (("Que dit l'article 39 du Code du travail ?",), "Et l'article 40 ?", _CODE_DU_TRAVAIL),
        (("Que dit l'article 74 du Code pénal ?",), "Et l'article 75 ?", ("PRC-4722",)),
        # Un article nomme avec un RENVOI au texte de la conversation (revue du
        # tour 4, sonde_renvoi_article_flux et sonde_meme_code) : cherchees
        # seules, elles lisaient l'article 38 ou 40 de six a huit textes sans
        # rapport, aucun extrait du Code du travail ; « dudit code » remontait
        # en tete l'article 38 du Code de la nationalite. L'article doit etre
        # lu dans le texte du tour precedent, comme pour « Et l'article 38 ? ».
        ((_PRECEDENTE_FR,), "Que dit l'article 38 de ce code ?", _CODE_DU_TRAVAIL),
        ((_PRECEDENTE_FR,), "Et l'article 38 de cette loi ?", _CODE_DU_TRAVAIL),
        ((_PRECEDENTE_FR,), "Et l'article 40 de ce texte ?", _CODE_DU_TRAVAIL),
        ((_PRECEDENTE_FR,), "Que prévoit l'article 40 du même code ?", _CODE_DU_TRAVAIL),
        ((_PRECEDENTE_FR,), "Que dit l'article 38 dudit code ?", _CODE_DU_TRAVAIL),
        ((_PRECEDENTE_FR,), "Que dit l'article 38 de la loi ?", _CODE_DU_TRAVAIL),
        (("Que dit l'article 39 du Code du travail ?",), "Que dit l'article 40 du même code ?",
         _CODE_DU_TRAVAIL),
        (("Que dit l'article 39 du Code du travail ?",), "Et l'article 40 de ce code ?",
         _CODE_DU_TRAVAIL),
        # La SUITE d'un changement de sujet (sonde du tour 5) : jointe au sujet
        # entier (« … licenciement abusif ? Et la nationalité ? »), « Peux-tu
        # détailler ? » ne lisait aucun extrait du Code de la nationalite. Le
        # texte du nouveau sujet doit etre lu.
        ((_PRECEDENTE_FR, "Et la nationalité ?"), "Peux-tu détailler ?", ("LOI-1968-LF-3",)),
        ((_PRECEDENTE_FR, "Et l'héritage ?"), "Donne-moi un exemple concret", ("CODE-CIVIL",)),
        ((_PRECEDENTE_FR, "Et si mon mari me bat ?"), "Que dois-je faire maintenant ?",
         ("PRC-4722", "CODE-CIVIL", "ORD-1981-002")),
    ],
    "en": [
        ((_PRECEDENTE_EN,), "Can you explain more?", _CODE_DU_TRAVAIL),
        ((_PRECEDENTE_EN, "Can you explain more?"), "Give me an example", _CODE_DU_TRAVAIL),
        ((_PRECEDENTE_EN, "Give me an example"), "What should I do now?", _CODE_DU_TRAVAIL),
        ((_PRECEDENTE_EN,), "What about the notice?", _CODE_DU_TRAVAIL),
        (("What does Article 39 of the Labour Code say?",), "And Article 40?", _CODE_DU_TRAVAIL),
        ((_PRECEDENTE_EN,), "What does Article 40 of this law say?", _CODE_DU_TRAVAIL),
        ((_PRECEDENTE_EN,), "And Article 40 of the same code?", _CODE_DU_TRAVAIL),
        ((_PRECEDENTE_EN, "What about inheritance?"), "Can you explain more?", ("CODE-CIVIL",)),
    ],
}
REPONSE_PRECEDENTE: Dict[str, str] = {"fr": _REPONSE_FR, "en": _REPONSE_EN}

# ==================== CHANGEMENTS DE SUJET ====================

# Une question COURTE ouverte par « Et », « Mais », « What about », « But what
# if », qui apporte ses propres mots : une suite, ou un NOUVEAU sujet (revue du
# tour 4, sonde_nouveau_sujet). Le tour 4 ne la cherchait que jointe au sujet
# precedent : apres une question sur le licenciement, « Et la nationalité ? »
# lisait 0 extrait sur 8 du Code de la nationalite (8 sur 8 cherchee seule),
# « Et l'héritage ? » un seul extrait sur les successions, « Et en cas de
# viol ? » 2 extraits du Code penal sur 8 (7 seule). (questions precedentes,
# question, textes du NOUVEAU sujet dont au moins un extrait doit etre lu.)
# Les violences d'un mari : le Code penal, ou le divorce pour sevices (Code
# civil, ordonnance de 1981 sur l'etat civil), que la question lit seule.
_VIOLENCES_CONJUGALES = ("PRC-4722", "CODE-CIVIL", "ORD-1981-002")
CHANGEMENTS_DE_SUJET: Dict[str, List[Tuple[Tuple[str, ...], str, Tuple[str, ...]]]] = {
    "fr": [
        ((_PRECEDENTE_FR,), "Et si mon mari me bat ?", _VIOLENCES_CONJUGALES),
        ((_PRECEDENTE_FR,), "Et si mon mari me frappe, que faire ?", _VIOLENCES_CONJUGALES),
        ((_PRECEDENTE_FR,), "Et si on me vole mon téléphone ?", ("PRC-4722",)),
        ((_PRECEDENTE_FR,), "Et en cas de viol ?", ("PRC-4722",)),
        ((_PRECEDENTE_FR,), "Mais que dit la loi sur le harcèlement sexuel ?", ("PRC-4722",)),
        ((_PRECEDENTE_FR,), "Mais si la police m'arrête sans mandat ?", ("LOI-2005-007",)),
        ((_PRECEDENTE_FR,), "Et pour le divorce ?", ("CODE-CIVIL", "ORD-1981-002")),
        ((_PRECEDENTE_FR,), "Et l'héritage ?", ("CODE-CIVIL",)),
        ((_PRECEDENTE_FR,), "Et la nationalité ?", ("LOI-1968-LF-3",)),
        ((_PRECEDENTE_FR,), "Et pour un terrain sans titre foncier ?",
         ("ORD-1974-001", "DEC-1976-165")),
        # Les deux autres de la revue (sonde_detecteurs), mesurees au tour 5.
        ((_PRECEDENTE_FR,), "Et la garde des enfants ?", ("CODE-CIVIL", "ORD-1981-002")),
        # Le centre de formalites de creation des entreprises (decret n°
        # 2013/297), l'agence de promotion des PME, ou le droit OHADA.
        ((_PRECEDENTE_FR,), "Et pour créer une entreprise ?",
         ("PRC-547", "PRC-9741", "OHADA-AUSCGIE-2014", "OHADA-AUDCG-2010")),
    ],
    "en": [
        ((_PRECEDENTE_EN,), "What about domestic violence?", _VIOLENCES_CONJUGALES),
        ((_PRECEDENTE_EN,), "But what if my landlord evicts me?",
         ("CODE-CIVIL", "OHADA-AUDCG-2010")),
        ((_PRECEDENTE_EN,), "What about inheritance?", ("CODE-CIVIL",)),
        # Quatre mots propres : pas une relance, cherchee seule (sonde_detecteurs).
        ((_PRECEDENTE_EN,), "And if the police arrest me without a warrant?", ("LOI-2005-007",)),
    ],
}

# Des questions REFUSEES seules, qui ont leurs propres mots, dans une
# conversation sur le licenciement (revue du tour 4, sonde_numero_et_refus) :
# jointes au sujet par la seconde chance, « la dot » (0,235 seule) et « c koi
# la dot » (0,215) passaient (0,492 et 0,461) avec huit extraits du Code du
# travail, du statut des fonctionnaires et du Code civil, aucun sur la dot. Le
# modele aurait repondu sur l'ancien sujet, ou de memoire. Attendu : un refus,
# ou un contexte qui porte leurs mots ; jamais celui de l'ancien sujet seul.
SECONDES_CHANCES: Dict[str, List[Tuple[Tuple[str, ...], str, Tuple[str, ...]]]] = {
    "fr": [
        ((_PRECEDENTE_FR,), "la dot", ()),
        ((_PRECEDENTE_FR,), "c koi la dot", ()),
        ((_PRECEDENTE_FR,), "c koi un CDD", ()),
    ],
}

# ==================== ARTICLES NOMMES ====================

# (question, reference du texte attendu en tete ; None : RIEN ne doit etre
# ajoute). Les questions de la revue du 29/10/2026 qui menaient a un autre
# texte, les textes revises (Constitution), un numero ambigu (Code civil),
# un texte remplace (loi n° 2013/004), des articles absents.
NOMMEES_REVUE: Dict[str, List[Tuple[str, Optional[str]]]] = {
    "fr": [
        ("Que dit l'article 2 de la loi sur le terrorisme ?", None),
        ("Que dit l'article premier de la loi sur le terrorisme ?", None),
        ("Que dit l'article 2 de la loi sur les armes ?", None),
        ("Que dit l'article 1er de la loi sur le droit d'auteur ?", None),
        ("Que dit l'article 2 de la loi sur la sécurité sociale ?", None),
        ("Que dit l'article 1 de la loi foncière ?", None),
        ("Que dit l'article 2 de la loi sur la police ?", None),
        ("Que dit l'article 1 de la loi sur l'Assemblée nationale ?", None),
        ("Que dit l'article 1 de la loi sur la décentralisation ?", None),
        ("Que dit l'article 2 de la loi sur les investissements ?", None),
        ("Que dit l'article 3 du décret de 2018 ?", None),
        ("Que dit l'article 2 de la loi de 2024 ?", None),
        ("Que dit l'article 1 de la loi sur la cybercriminalité ?", None),
        ("Que dit l'article 2 de la loi sur la cybercriminalité ?", None),
        ("Que dit l'article 1er de la loi sur les associations ?", None),
        ("Que dit l'article 30 de la loi de finances ?", None),
        # Revises depuis : rien d'ajoute, la recherche et les lignes de statut.
        ("Que dit l'article 6 de la Constitution ?", None),
        ("Que dit l'article 15 de la Constitution ?", None),
        # Numero porte par trois articles de la compilation.
        ("Que dit l'article 3 du Code civil ?", None),
        # Texte remplace par l'ordonnance de 2025.
        ("Que dit l'article 1 de la loi n° 2013/004 ?", None),
        ("Que dit l'article 2 de la loi n° 2013/004 ?", None),
        # Absents du Code du travail.
        ("Que dit l'article 200 du Code du travail ?", None),
        ("Que dit l'article 250 du Code du travail ?", None),
        # Ajoutes : la table, ou le numero propre.
        ("Que dit l'article 76 du Code du travail ?", "LOI-1992-007"),
        ("Que dit l'article 5 de l'Acte uniforme relatif au droit des sociétés ?",
         "OHADA-AUSCGIE-2014"),
        ("Que dit l'article 1er de la Constitution ?", "PRC-6283"),
        ("Que dit l'article 35 de la loi n° 92/007 ?", "LOI-1992-007"),
        ("Que dit l'article 119 du CPP ?", "LOI-2005-007"),
        ("Que dit l'article 10 du Code minier ?", "PRC-9866"),
    ],
    "en": [
        ("What does Article 6 of the Constitution say?", None),
        ("What does Article 2 of the law on terrorism say?", None),
        ("What does Article 37 of the Labour Code say?", "LOI-1992-007"),
        ("What does Article 2 of Law No. 2005/007 say?", "LOI-2005-007"),
    ],
}
