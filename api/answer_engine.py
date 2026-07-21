"""Turn a classified intent (+ the user's original question + session context)
into a natural-language answer suitable for sending back over WhatsApp.

Pipeline order inside answer():

    1. small-talk short-circuits (greeting / thanks / goodbye)
    2. glossary lookup — "what is X" style definitional questions
    3. out-of-scope gate — off-topic / no-domain-vocab / low-confidence
    4. RAG + generator — retrieve statute chunks, generate a natural-language
       answer with flan-t5. Returns None if either component isn't loaded.
    5. answer_bank fallback — the pre-baked canonical response per intent+lang

Runtime dependencies (all optional except answer_bank.json):

    artifacts/answer_bank.json      — REQUIRED. Baked at training time.
    artifacts/rag_index.npz         — OPTIONAL. Enables layer 4.
    artifacts/rag_chunks.json       — OPTIONAL. Enables layer 4.
    sheriabot_generator/            — OPTIONAL. Enables generative rewrite.

Each optional layer degrades gracefully: missing RAG index → no retrieval;
missing generator model → answer_bank still fires. The bot always ships.

# KNOWN LIMITATION (needs retraining, not a code fix):
#   "Nimepigwa makofi na bosi" (I was slapped by my boss) is currently
#   classified as `workplace_injury` because the training data groups all
#   "physical harm at work" under WCF/injury. Semantically it's assault or
#   physical harassment. Fix: add a `workplace_violence` (or
#   `harassment_physical`) intent to the training set and retrain BERT.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Dict, Any, List, Optional

from config import ARTIFACTS

# Optional layers — safe to import even if their model/index isn't present.
# They handle missing artifacts internally and return None/[] on failure.
import rag
import generator

log = logging.getLogger("sheriabot.answer_engine")

ANSWER_BANK_JSON = ARTIFACTS / "answer_bank.json"

# Distinctive Swahili tokens. Used both as a fast short-text lang detector
# (before langid) and as a last-resort fallback if langid isn't installed.
_SWAHILI_MARKERS = {
    # function words
    "na","ya","wa","ni","kwa","za","la","ku","katika","hii","hiyo","nini",
    "gani","lini","wapi","vipi","je","kwenye","kutoka","kuhusu",
    # verbs / first-person forms
    "nimefutwa","nifanye","nataka","sijui","siwezi","sitaki","nafanya",
    "nimechoka","nimeumia","ninafanya","nikafanya","ninaishi","naishi","naitwa",
    # greetings / social phrases
    "tafadhali","asante","asanteni","shukrani","habari","jambo","mambo",
    "salama","shikamoo","hodi","hongera","karibu","samahani",
    # goodbyes
    "kwaheri","baadaye","tutaonana","usiku","mchana","jioni",
    # pronouns
    "mimi","wewe","yeye","sisi","nyinyi","wao","yangu","yako","yake","yetu",
    # topic keywords (Swahili-only — acronyms like CMA/ELRA/WCF are excluded
    # because they are language-neutral proper nouns that appear in both EN
    # and SW sentences)
    "kazi","mshahara","mkataba","likizo","ugonjwa","mimba","mtoto","mwajiri",
    "mfanyakazi","haki","sheria","kesi","malipo","fidia","kufukuzwa",
    # short affirmatives / negatives
    "ndio","ndiyo","hapana","sawa","hakika","kabisa",
    # family / relations
    "rafiki","dada","kaka","baba","mama","mtoto",
}

# --- Out-of-scope limiting ------------------------------------------------
#
# BERT is over-confident: it happily labels "How do I bake bread?" as
# `cma_filing` @ 0.997. Raw confidence alone is useless as an OOS signal.
# So we combine THREE checks, and any one of them triggers the OOS reply:
#
#   1. Confidence floor — the model itself is unsure.
#   2. Top-1 / top-2 gap — the model is torn between two classes.
#   3. Domain-vocabulary gate — the input contains ZERO employment/labor-law
#      words in either language. If someone asks about weather or football,
#      BERT will still predict *something*, but the domain gate blocks it.
#
# The domain gate uses substring matching on word ROOTS so morphological
# variants (dismissed / dismissal / kufukuzwa / amenifukuza) all count.

CONFIDENCE_FLOOR = 0.60          # below this → out-of-scope
TOP_GAP_FLOOR    = 0.15          # top-1 minus top-2; below → ambiguous

# Bilingual employment-law vocabulary. Substrings are matched against each
# input token (lower-cased). Kept intentionally narrow — every entry is a
# high-signal root for employment / labor / contract / dispute language.
_DOMAIN_ROOTS_EN = {
    # employment relationship
    "employ", "employer", "employee", "employe", "worker", "workforce",
    "workplace", "workplac", "work", "worksite", "job", "jobs", "colleague",
    "boss", "supervisor", "hire", "hiring", "hired", "recruit", "staff",
    # money
    "wage", "wages", "salary", "salaries", "pay", "paid", "unpaid", "underpay",
    "underpaid", "bonus", "commission", "allowance", "overtime", "compensat",
    "compensation", "remunerat", "severance", "gratuity", "pension",
    "insurance", "benefit", "benefits",
    # contract / termination
    "contract", "contracts", "agreement", "clause", "notice", "notic",
    "terminat", "resign", "resigned", "dismiss", "dismissal", "dismissed",
    "fire", "fired", "sack", "sacked", "retrench", "retrenchment", "redundan",
    "probation", "fixed-term", "fixedterm", "permanent", "casual",
    # rights / disputes / process
    "labor", "labour", "law", "legal", "rights", "right", "court", "tribunal",
    "cma", "elra", "wcf", "wca", "nssf", "psssf", "nhif", "osha", "lla", "gn",
    "case", "claim", "complaint", "dispute", "filing", "file",
    "sue", "suing", "advocate", "attorney", "lawyer", "arbitrat", "mediat",
    "hearing", "appeal",
    # workplace situations
    "discriminat", "harass", "bully", "bullying", "unfair", "wrongful",
    "safety", "injur", "injury", "accident", "sick", "illness", "matern",
    "paternity", "leave", "holiday", "vacation", "union", "strike",
    "grievance",
}
_DOMAIN_ROOTS_SW = {
    # employment relationship
    "kazi", "kaz", "ajira", "ajir", "mfanyakazi", "wafanyakazi", "mwajiri",
    "waajiri", "waajir", "ofisi", "mahali",
    # money
    "mshahara", "mishahara", "malipo", "lipa", "kulipwa", "hajanilipa",
    "hakulipwa", "kiinua", "kiinua-mgongo", "posho", "fidia", "bima",
    "pensheni", "faida",
    # contract / termination
    "mkataba", "mikataba", "makubaliano", "notisi", "kufukuzwa", "amenifukuza",
    "wamenifukuza", "wamefukuzwa", "kuachishwa", "kuachwa", "kupunguzwa",
    "kufutwa", "kumaliza",
    # rights / disputes / process
    "sheria", "haki", "kesi", "mahakama", "cma", "elra", "wcf", "wca",
    "nssf", "psssf", "nhif", "osha", "lla", "malalamiko",
    "lalamiko", "kufungua", "kuwasilisha", "wakili", "usuluhishi", "usimamizi",
    # workplace situations
    "ubaguzi", "unyanyasaji", "kudhulumiwa", "kudhulumu", "kudhulum", "dhulum",
    "meneja", "usimamiz", "kingono", "kimapenzi", "jeraha", "kuumia", "ugonjwa",
    "mgonjwa", "mimba", "uzazi", "likizo", "muungano", "mgomo", "usalama",
    "hatari",
}

# --- Glossary of legal acronyms + concepts + Kiswahili terms -------------
# Definitional questions ("what is X?", "X ni nini?", "meaning of X") bypass
# the intent classifier and are answered from this table. BERT was never
# trained on define-questions so without this it would return a procedural
# answer for a definition query.
#
# Keys are lower-case, single tokens. For concepts that have both an English
# and a Swahili name (e.g. "severance"/"kifuta", "notice"/"notisi"), we
# register BOTH keys, each pointing at the same bilingual definition, so the
# glossary works regardless of which language the user asked in.
_GLOSSARY = {
    # ═══════════════════════════════════════════════════════════════════
    # INSTITUTIONAL ACRONYMS
    # ═══════════════════════════════════════════════════════════════════
    "cma": {
        "en": ("CMA stands for the Commission for Mediation and Arbitration — "
               "the government body that resolves labour disputes in Tanzania. "
               "You file complaints there using CMA Form 1 within 60 days of "
               "the dispute."),
        "sw": ("CMA ni Tume ya Upatanishi na Usuluhishi — chombo cha serikali "
               "kinachoshughulikia migogoro ya kazi Tanzania. Unawasilisha "
               "malalamiko kwa Fomu Na. 1 ya CMA ndani ya siku 60 tangu "
               "mgogoro."),
    },
    "elra": {
        "en": ("ELRA is the Employment and Labour Relations Act, 2004 "
               "(Cap. 366) — the main statute governing employer-employee "
               "relations in mainland Tanzania. Amended by the Labour Laws "
               "(Amendments) Act No. 4 of 2025 (LLA 4/2025)."),
        "sw": ("ELRA ni Sheria ya Ajira na Mahusiano Kazini ya mwaka 2004 "
               "(Sura 366) — sheria kuu inayosimamia mahusiano kati ya "
               "mwajiri na mfanyakazi Tanzania. Iliboreshwa na Sheria Na. 4 "
               "ya 2025 (LLA 4/2025)."),
    },
    "lia": {
        "en": ("LIA is the Labour Institutions Act, 2004 — sets up the "
               "labour institutions (Labour Commissioner, CMA, Labour Court, "
               "Labour, Economic and Social Council). Works alongside ELRA."),
        "sw": ("LIA ni Sheria ya Taasisi za Kazi ya 2004 — inaunda taasisi "
               "za kazi (Kamishna wa Kazi, CMA, Mahakama ya Kazi, Baraza la "
               "Kazi, Uchumi na Jamii). Inashirikiana na ELRA."),
    },
    "wcf": {
        "en": ("WCF stands for the Workers' Compensation Fund — the "
               "government fund that compensates workers injured or made "
               "ill at work. Report injuries to WCF within 12 months."),
        "sw": ("WCF ni Mfuko wa Fidia ya Wafanyakazi — mfuko wa serikali "
               "unaolipa fidia kwa wafanyakazi waliopata majeraha au "
               "maradhi kazini. Ripoti majeraha ndani ya miezi 12."),
    },
    "wca": {
        "en": ("WCA is the Workers' Compensation Act, 2008 — the law that "
               "establishes the Workers' Compensation Fund and workplace "
               "injury compensation rules."),
        "sw": ("WCA ni Sheria ya Fidia ya Wafanyakazi ya 2008 — inayoanzisha "
               "Mfuko wa WCF na kanuni za fidia kwa majeraha ya kazini."),
    },
    "nssf": {
        "en": ("NSSF is the National Social Security Fund — the private-"
               "sector pension scheme all employers must register their "
               "employees with. Contribution is 20 % of the wage (10 % "
               "employer, 10 % employee)."),
        "sw": ("NSSF ni Mfuko wa Taifa wa Hifadhi ya Jamii — mfuko wa "
               "pensheni wa sekta binafsi ambao waajiri wote lazima "
               "wawasajili wafanyakazi wao. Mchango ni 20 % ya mshahara "
               "(10 % mwajiri, 10 % mfanyakazi)."),
    },
    "psssf": {
        "en": ("PSSSF is the Public Service Social Security Fund — the "
               "public-sector equivalent of NSSF, for government employees."),
        "sw": ("PSSSF ni Mfuko wa Hifadhi ya Jamii wa Utumishi wa Umma — "
               "sawa na NSSF lakini kwa watumishi wa serikali."),
    },
    "nhif": {
        "en": ("NHIF is the National Health Insurance Fund — the health-"
               "insurance scheme for formal-sector workers."),
        "sw": ("NHIF ni Mfuko wa Taifa wa Bima ya Afya — bima ya afya kwa "
               "wafanyakazi wa sekta rasmi."),
    },
    "osha": {
        "en": ("OSHA is the Occupational Safety and Health Authority — the "
               "regulator that enforces workplace safety standards under "
               "the Occupational Safety and Health Act, 2003."),
        "sw": ("OSHA ni Mamlaka ya Usalama na Afya Mahali pa Kazi — "
               "inayotekeleza viwango vya usalama chini ya Sheria ya "
               "Usalama na Afya Mahali pa Kazi ya 2003."),
    },
    "lla": {
        "en": ("LLA usually refers to the Labour Laws (Amendments) Act "
               "No. 4 of 2025 — a major 2025 amendment package that "
               "updated ELRA, LIA, WCA, and other labour statutes."),
        "sw": ("LLA ni Sheria ya Marekebisho ya Sheria za Kazi Na. 4 ya "
               "2025 — kifurushi cha marekebisho makubwa cha 2025 "
               "kilichoboresha ELRA, LIA, WCA, na sheria nyingine za kazi."),
    },
    "tucta": {
        "en": ("TUCTA is the Trade Union Congress of Tanzania — the "
               "national federation of trade unions. Recognised workers' "
               "unions such as TALGWU, CHODAWU, and TUICO are affiliated "
               "under TUCTA."),
        "sw": ("TUCTA ni Shirikisho la Vyama vya Wafanyakazi Tanzania — "
               "vyama vya wafanyakazi vilivyosajiliwa kama TALGWU, "
               "CHODAWU, na TUICO viko chini ya TUCTA."),
    },
    "paye": {
        "en": ("PAYE is Pay As You Earn — the income tax that the employer "
               "deducts from your monthly salary and remits to TRA. Rates "
               "are progressive (0-30 %) depending on salary band."),
        "sw": ("PAYE ni kodi ya mapato inayokatwa kutoka mshahara wako "
               "kila mwezi na mwajiri, kisha kupelekwa TRA. Viwango ni "
               "vya ngazi (0-30 %) kulingana na kiasi cha mshahara."),
    },
    "sdl": {
        "en": ("SDL is the Skills Development Levy — a 4 % payroll tax the "
               "EMPLOYER pays to TRA on top of gross salaries, funding "
               "national skills training programmes."),
        "sw": ("SDL ni ushuru wa Maendeleo ya Ujuzi — kodi ya 4 % "
               "inayolipwa na MWAJIRI kwa TRA juu ya mshahara ghafi, "
               "kufadhili mafunzo ya ujuzi ya kitaifa."),
    },

    # ═══════════════════════════════════════════════════════════════════
    # LEGAL CONCEPTS — EN keys (also aliased below with SW keys)
    # ═══════════════════════════════════════════════════════════════════
    "severance": {
        "en": ("Severance pay (Kiswahili: kifuta jasho) is the terminal "
               "payment an employer owes when they dismiss you for economic "
               "reasons (retrenchment/redundancy). Formula under ELRA s.42: "
               "7 DAYS of basic wage per COMPLETED YEAR of service, capped "
               "at 10 years. So 5 years' service = 35 days' basic wage. "
               "It's tax-free."),
        "sw": ("Kifuta jasho ni malipo unayostahili mwajiri akikufukuza kwa "
               "sababu za kiuchumi (kupunguzwa wafanyakazi). Fomula ya ELRA "
               "s.42: SIKU 7 za mshahara wa msingi kwa kila MWAKA "
               "uliokamilika wa utumishi, kikomo miaka 10. Yaani miaka 5 "
               "ya utumishi = siku 35 za mshahara wa msingi. Halijaidiwi "
               "kodi."),
    },
    "gratuity": {
        "en": ("Gratuity (Kiswahili: kiinua-mgongo) is a lump-sum payment "
               "on completion of a fixed-term contract or on retirement. "
               "Amount depends on the contract or on employer policy — "
               "there is no statutory formula in ELRA. Common practice is "
               "10-25 % of accumulated salary."),
        "sw": ("Kiinua-mgongo ni malipo ya jumla wakati wa kumaliza "
               "mkataba wa muda au kustaafu. Kiasi hutegemea mkataba au "
               "sera ya mwajiri — ELRA haina fomula rasmi. Kawaida ni "
               "10-25 % ya mshahara uliokusanywa."),
    },
    "retrenchment": {
        "en": ("Retrenchment is dismissal for OPERATIONAL / economic reasons "
               "— redundancy, restructuring, downsizing. ELRA s.38 requires "
               "the employer to consult with you and any union BEFORE, "
               "consider alternatives (transfer, reduced hours), then pay "
               "severance under s.42. Without consultation the retrenchment "
               "is procedurally unfair."),
        "sw": ("Retrenchment ni kufukuzwa kwa sababu za kiuchumi / "
               "kimuundo. ELRA s.38 inamtaka mwajiri akushauriane nawe na "
               "chama chochote KABLA, azingatie mbadala (uhamisho, masaa "
               "yaliyopunguzwa), kisha alipe kifuta jasho chini ya s.42. "
               "Bila mashauriano, retrenchment ni bila haki kwa "
               "utaratibu."),
    },
    "notice": {
        "en": ("Notice period (Kiswahili: notisi) is the advance warning "
               "either party must give before terminating employment. ELRA "
               "s.41: 7 days for daily-paid workers; 4 days for weekly-paid; "
               "28 days for monthly-paid. If the employer dismisses without "
               "notice, they must pay you 'notice pay' equal to the notice "
               "period."),
        "sw": ("Notisi ni tahadhari ya awali ambayo pande zote lazima "
               "watoe kabla ya kusitisha ajira. ELRA s.41: siku 7 kwa "
               "wanaolipwa kila siku; siku 4 kwa wanaolipwa kila wiki; "
               "siku 28 kwa wanaolipwa kila mwezi. Mwajiri akikufukuza "
               "bila notisi, lazima akulipe 'malipo ya notisi' sawa na "
               "muda wa notisi."),
    },
    "probation": {
        "en": ("Probation is a trial period at the start of employment. "
               "Under GN 42/2007 Rule 8, the maximum probation period is "
               "6 months. Even during probation, the employer must follow "
               "a fair procedure before dismissal (warning + hearing). "
               "Probation ≠ 'no rights' — you can still challenge unfair "
               "treatment."),
        "sw": ("Probation ni kipindi cha majaribio mwanzoni mwa ajira. "
               "Chini ya GN 42/2007 Kanuni 8, upeo wa probation ni miezi "
               "6. Hata wakati wa probation, mwajiri lazima afuate "
               "utaratibu wa haki kabla ya kufukuzwa (onyo + kikao). "
               "Probation SI 'bila haki' — bado unaweza kupinga "
               "ubaguzi."),
    },
    "dismissal": {
        "en": ("Dismissal is the employer terminating your employment. It "
               "must have (1) a valid reason (misconduct, incapacity, "
               "operational requirements) AND (2) a fair procedure "
               "(warning, hearing, opportunity to respond). Missing "
               "either = unfair dismissal → CMA claim within 60 days."),
        "sw": ("Kufukuzwa ni mwajiri kusitisha ajira yako. Lazima kuwe na "
               "(1) sababu halali (utovu wa nidhamu, kutokuwa na uwezo, "
               "mahitaji ya kiuchumi) NA (2) utaratibu wa haki (onyo, "
               "kikao, nafasi ya kujibu). Ukikosa mojawapo = kufukuzwa "
               "bila haki → dai CMA ndani ya siku 60."),
    },
    "constructive": {
        "en": ("Constructive dismissal is when the employer's conduct is "
               "so intolerable that you are forced to resign. The law "
               "treats it AS IF they dismissed you — you keep all rights: "
               "severance, notice pay, unfair-dismissal claim. Common "
               "triggers: unilateral pay cut, demotion without cause, "
               "harassment, forced transfer."),
        "sw": ("Constructive dismissal ni pale mwenendo wa mwajiri "
               "unakulazimisha kujiuzulu. Sheria inaichukulia KAMA "
               "walikufukuza — unahifadhi haki zote: kifuta jasho, "
               "malipo ya notisi, dai la kufukuzwa bila haki. Sababu za "
               "kawaida: kupunguza mshahara bila makubaliano, "
               "kushushwa cheo bila sababu, unyanyasaji, uhamisho wa "
               "lazima."),
    },
    "reinstatement": {
        "en": ("Reinstatement is one of the remedies CMA / Labour Court "
               "can order for unfair dismissal — the employer must give "
               "you your job back with all lost wages. The other remedy "
               "is re-engagement (a similar new role) or compensation "
               "(cash instead of returning). You can request any of the "
               "three."),
        "sw": ("Kurudishwa kazini ni mojawapo ya suluhu ambayo CMA / "
               "Mahakama ya Kazi inaweza kuamuru kwa kufukuzwa bila "
               "haki — mwajiri lazima akurudishe kazini pamoja na "
               "mshahara wote uliopotea. Suluhu nyingine ni "
               "re-engagement (nafasi kama hiyo mpya) au fidia "
               "(pesa badala ya kurudi)."),
    },
    "misconduct": {
        "en": ("Misconduct is behaviour that breaches your employment duties. "
               "GROSS misconduct (theft, fraud, violence, drunkenness on "
               "duty) can justify dismissal without notice — but ONLY "
               "after a disciplinary hearing. ORDINARY misconduct "
               "(lateness, minor errors) requires a graduated warning "
               "process first."),
        "sw": ("Utovu wa nidhamu ni tabia inayokiuka wajibu wako wa kazi. "
               "Utovu MKUBWA (wizi, udanganyifu, ghasia, ulevi kazini) "
               "unaweza kuhalalisha kufukuzwa bila notisi — LAKINI TU "
               "baada ya kikao cha nidhamu. Utovu wa kawaida "
               "(kuchelewa, makosa madogo) unahitaji mchakato wa maonyo "
               "ya ngazi kwanza."),
    },
    "overtime": {
        "en": ("Overtime (Kiswahili: muda wa ziada) is work beyond the "
               "normal 45 hours per week / 9 hours per day. ELRA s.19: "
               "must be paid at 1.5× the normal hourly rate on weekdays "
               "and 2× on rest days and public holidays. Overtime cannot "
               "exceed 50 hours in any 4-week period."),
        "sw": ("Muda wa ziada ni kazi zaidi ya masaa 45 ya kawaida kwa "
               "wiki / masaa 9 kwa siku. ELRA s.19: lazima ulipwe MARA "
               "1.5 ya kiwango cha kawaida cha saa siku za kazi na "
               "MARA 2 siku za mapumziko na sikukuu. Muda wa ziada "
               "usizidi masaa 50 katika kipindi cha wiki 4."),
    },
    "grievance": {
        "en": ("A grievance is a formal complaint you lodge with your "
               "employer over a workplace issue (unpaid wages, unfair "
               "treatment, harassment). Under the Code of Good Practice, "
               "the employer must acknowledge within 5 days and resolve "
               "within 14 days. If unresolved, escalate to CMA."),
        "sw": ("Malalamiko rasmi ni malalamiko ya kimaandishi "
               "unayowasilisha kwa mwajiri kuhusu suala la kazi "
               "(mshahara usiolipwa, ubaguzi, unyanyasaji). Chini ya "
               "Kanuni za Mwenendo Bora, mwajiri lazima ajibu ndani ya "
               "siku 5 na atatue ndani ya siku 14. Ikikosekana suluhu, "
               "peleka CMA."),
    },
    "discrimination": {
        "en": ("Discrimination is treating you unfavourably because of "
               "race, sex, religion, HIV status, pregnancy, disability, "
               "age, or union membership. Prohibited by ELRA s.7. Any "
               "employment decision (hiring, pay, promotion, dismissal) "
               "based on these grounds is unlawful."),
        "sw": ("Ubaguzi ni kukutendea vibaya kwa sababu ya kabila, "
               "jinsia, dini, hali ya VVU, mimba, ulemavu, umri, au "
               "uanachama wa chama. Ni marufuku chini ya ELRA s.7. "
               "Uamuzi wowote wa ajira (kuajiri, malipo, kupanda cheo, "
               "kufukuzwa) unaotokana na sababu hizi ni haramu."),
    },
    "harassment": {
        "en": ("Harassment is any unwelcome conduct that creates a hostile "
               "work environment. Includes verbal, physical, and sexual "
               "harassment. Prohibited by ELRA s.7(3). Report internally "
               "in writing first, then file CMA Form 1 within 60 days if "
               "not resolved."),
        "sw": ("Unyanyasaji ni mwenendo wowote usiotakwa unaotengeneza "
               "mazingira ya uadui kazini. Unajumuisha unyanyasaji wa "
               "maneno, kimwili, na kingono. Ni marufuku chini ya ELRA "
               "s.7(3). Ripoti ndani kimaandishi kwanza, kisha jaza "
               "Fomu Na. 1 ya CMA ndani ya siku 60."),
    },

    # ═══════════════════════════════════════════════════════════════════
    # KISWAHILI KEYS — aliases pointing at concepts above so a Swahili-
    # speaking user asking "kifuta jasho ni nini?" gets the same answer.
    # ═══════════════════════════════════════════════════════════════════
}
# Alias table — Swahili terms that point at the same definition as an
# English concept above. Kept as a separate step to make the mapping
# easier to audit and extend.
_GLOSSARY.update({
    "kifuta":       _GLOSSARY["severance"],
    "kiinua":       _GLOSSARY["gratuity"],
    "notisi":       _GLOSSARY["notice"],
    "kurudishwa":   _GLOSSARY["reinstatement"],
    "kupunguzwa":   _GLOSSARY["retrenchment"],
    "probesheni":   _GLOSSARY["probation"],
    "ubaguzi":      _GLOSSARY["discrimination"],
    "unyanyasaji":  _GLOSSARY["harassment"],
})

# Patterns that indicate the user is asking for a DEFINITION rather than
# procedural advice.
_DEFINITION_PATTERNS = (
    # English
    "what is ", "what's ", "whats ", "what does ", "meaning of ",
    "define ", "explain ", "tell me about ",
    # Swahili
    " ni nini", "nini maana ya ", "maana ya ", "eleza ", "elezea ",
    "nini ", "nieleze kuhusu ", "niambie kuhusu ",
)

# A few phrases that are DEFINITELY off-topic; catches "obvious" cases even if
# the domain gate misses. Substring match on lower-cased whole text.
_OFF_TOPIC_PATTERNS = (
    # weather / time / news
    "weather", "temperature", "forecast", "hali ya hewa", "muda", "saa ngapi",
    # cooking / food
    "recipe", "cook", "bake", "food", "meal", "pilau", "ugali", "chakula",
    # sports / entertainment
    "football", "soccer", "movie", "song", "joke", "mchezo", "mpira",
    # math / trivia / general knowledge
    # (removed "what is the " — was matching legit legal questions like
    # "what is the minimum wage..." and "what is the notice period...")
    "president", "capital city", "population", "who is the president",
    "who won ", "rais wa",
    # shopping
    "buy a car", "kununua", "nunua",
    # random personal
    "i love ", "napenda",
)


class AnswerEngine:
    """Loads the answer bank once and answers questions given a classified intent."""

    def __init__(self) -> None:
        self._bank: Dict[str, Dict[str, Dict[str, str]]] = {}
        self._source: str = ""
        self._load_bank()

    def _load_bank(self) -> None:
        if not ANSWER_BANK_JSON.exists():
            raise FileNotFoundError(
                f"{ANSWER_BANK_JSON} not found. This file is a training-time "
                "artifact produced by bake_answers.py (local) or the Colab "
                "training notebook. The runtime bot cannot start without it — "
                "deploy the trained artifacts/ folder alongside the code."
            )
        self._bank = json.loads(ANSWER_BANK_JSON.read_text())
        self._source = "answer_bank.json"

    # ------------------------------------------------------------------
    # language detection — hybrid: langid for longer text, but short messages
    # with strong Swahili markers ("Kwaheri", "Habari", "Nimefutwa") override
    # langid because it's unreliable on 2-3 word inputs. Restricted to {en,sw}.
    # ------------------------------------------------------------------
    @staticmethod
    def detect_lang(text: str) -> str:
        stripped = text.strip()
        if not stripped:
            return "en"
        tokens = re.findall(r"[a-zA-ZÀ-ſ]+", stripped.lower())
        if not tokens:
            return "en"

        sw_hits = sum(1 for t in tokens if t in _SWAHILI_MARKERS)

        # Short text with any Swahili marker → trust the marker over langid.
        if len(tokens) <= 5 and sw_hits >= 1:
            return "sw"

        # Longer text: langid handles it well.
        try:
            import langid
            lang, _score = langid.classify(stripped)
            if lang in ("en", "sw"):
                return lang
        except ImportError:
            pass

        return "sw" if sw_hits >= 1 else "en"

    # ------------------------------------------------------------------
    # small-talk classifiers
    # ------------------------------------------------------------------
    _GREETINGS = {
        "hi", "hello", "hey", "hallo", "yo", "howdy",
        "habari", "mambo", "jambo", "salama", "shikamoo", "vipi", "hodi",
    }
    _THANKS = {
        "thanks", "thank", "ta",
        "asante", "asanteni", "shukrani",
    }
    _AFFIRMATIVE = {"yes", "yeah", "yep", "ok", "okay", "sure", "ndio", "ndiyo", "sawa"}
    _NEGATIVE    = {"no", "nope", "hapana", "la"}
    _GOODBYE     = {"bye", "goodbye", "kwaheri", "baadaye", "tutaonana"}

    @classmethod
    def is_greeting(cls, text: str) -> bool:
        t = text.strip().lower()
        if not t:
            return False
        first = re.split(r"[\s,.!?]", t, maxsplit=1)[0]
        return first in cls._GREETINGS or t in cls._GREETINGS

    @classmethod
    def is_thanks(cls, text: str) -> bool:
        t = text.strip().lower().rstrip("!.")
        return t in cls._THANKS or any(t.startswith(k + " ") for k in cls._THANKS)

    @classmethod
    def is_affirmative(cls, text: str) -> bool:
        return text.strip().lower().rstrip("!.?") in cls._AFFIRMATIVE

    @classmethod
    def is_negative(cls, text: str) -> bool:
        return text.strip().lower().rstrip("!.?") in cls._NEGATIVE

    @classmethod
    def is_goodbye(cls, text: str) -> bool:
        t = text.strip().lower().rstrip("!.")
        return t in cls._GOODBYE

    # ------------------------------------------------------------------
    # Out-of-scope detection
    # ------------------------------------------------------------------
    @staticmethod
    def _has_domain_vocab(text: str) -> bool:
        """True if the input contains at least one employment/labor-law root
        (EN or SW). Uses substring matching so morphological variants count."""
        tokens = re.findall(r"[a-zA-ZÀ-ſ]+", text.lower())
        if not tokens:
            return False
        for tok in tokens:
            for root in _DOMAIN_ROOTS_EN:
                if root in tok:
                    return True
            for root in _DOMAIN_ROOTS_SW:
                if root in tok:
                    return True
        return False

    @staticmethod
    def _matches_off_topic(text: str) -> bool:
        """True if the input contains any known non-employment phrase."""
        low = text.lower()
        return any(pat in low for pat in _OFF_TOPIC_PATTERNS)

    @classmethod
    def glossary_lookup(cls, user_text: str, lang: str) -> Optional[Dict[str, str]]:
        """If the user is asking 'what is X' / 'X ni nini' for a known
        legal acronym, return {'response', 'citation', 'acronym'}; else None.
        Runs BEFORE BERT because the classifier has no 'define' intent."""
        low = user_text.lower()

        # Must contain a definition pattern (e.g. "what is", "ni nini")
        asks_definition = any(pat in low for pat in _DEFINITION_PATTERNS)
        if not asks_definition:
            return None

        # Find which acronym is being asked about.
        tokens = re.findall(r"[a-zA-Z]+", low)
        for tok in tokens:
            if tok in _GLOSSARY:
                entry = _GLOSSARY[tok].get(lang) or _GLOSSARY[tok]["en"]
                return {"response": entry, "citation": "", "acronym": tok.upper()}
        return None

    @classmethod
    def is_out_of_scope(
        cls,
        user_text: str,
        confidence: Optional[float] = None,
        top_3: Optional[list] = None,
    ) -> Optional[str]:
        """Return a short reason string if the input is out-of-scope, else None.

        Layered checks — the first one to trigger wins:
          1. Matches a hard-coded off-topic phrase.
          2. Contains ZERO employment/labor vocabulary.
          3. BERT confidence is below the floor.
          4. Top-1 / top-2 gap is smaller than the floor (model is torn).
        """
        if cls._matches_off_topic(user_text):
            return "off_topic_phrase"
        if not cls._has_domain_vocab(user_text):
            return "no_domain_vocab"
        if confidence is not None and confidence < CONFIDENCE_FLOOR:
            return "low_confidence"
        if top_3 and len(top_3) >= 2:
            gap = float(top_3[0][1]) - float(top_3[1][1])
            if gap < TOP_GAP_FLOOR:
                return "ambiguous_top2"
        return None

    # ------------------------------------------------------------------
    # public entry point
    # ------------------------------------------------------------------
    def answer(
        self,
        user_text: str,
        intent: str,
        confidence: Optional[float] = None,
        session: Optional[Dict[str, Any]] = None,
        top_3: Optional[list] = None,
    ) -> Dict[str, Any]:
        session = session or {}
        # Always re-detect language from the CURRENT message. The session's
        # stored language is only a fallback for empty/whitespace inputs; it
        # must never override an obvious code-switch mid-conversation.
        lang = self.detect_lang(user_text) or session.get("lang", "en")
        name = (session.get("profile") or {}).get("name")

        # ---- small-talk short-circuits (no legal lookup needed) ----
        if self.is_greeting(user_text):
            return self._reply(self._greeting_text(lang, name, session.get("turn_count", 0)),
                               intent="greeting", lang=lang, source="greeting")

        if self.is_thanks(user_text):
            return self._reply(self._thanks_reply(lang, name),
                               intent="thanks", lang=lang, source="thanks")

        if self.is_goodbye(user_text):
            return self._reply(self._goodbye_reply(lang, name),
                               intent="goodbye", lang=lang, source="goodbye")

        # ---- glossary lookup (BEFORE the intent classifier) ----
        # Handles "what is CMA?", "WCF ni nini?", "meaning of ELRA" etc.
        # BERT has no 'define' intent so these would otherwise fall through
        # to procedural answers or refusals.
        gloss = self.glossary_lookup(user_text, lang)
        if gloss is not None:
            text = gloss["response"]
            text = f"{text}\n\n{self._disclaimer(lang)}"
            return self._reply(
                text,
                intent=f"glossary:{gloss['acronym']}",
                lang=lang,
                source="glossary",
                citation=gloss.get("citation", ""),
            )

        # ---- out-of-scope gate ----
        oos_reason = self.is_out_of_scope(user_text, confidence, top_3)
        if oos_reason is not None:
            return self._reply(
                self._out_of_scope_reply(lang, name),
                intent=intent, lang=lang, source=f"out_of_scope:{oos_reason}",
            )

        # ─── LAYER 4: RAG + GENERATOR ────────────────────────────────
        # Try to produce a fresh, natural-language answer from retrieved
        # statute chunks. Returns None if either the RAG index or the
        # fine-tuned generator isn't loaded — in which case we drop through
        # to the answer_bank fallback below.
        gen_reply = self._try_generate(user_text, intent, lang, name)
        if gen_reply is not None:
            return gen_reply

        # ─── LAYER 5: ANSWER-BANK FALLBACK ───────────────────────────
        entry = self._bank.get(intent, {}).get(lang) or self._bank.get(intent, {}).get("en")
        if entry is None:
            return self._reply(self._unknown_intent_reply(lang, name),
                               intent=intent, lang=lang, source="unknown_intent_fallback")

        text = entry["response"]
        citation = entry.get("citation", "")
        if citation and citation.lower() not in text.lower():
            text = f"{text}\n\n_Ref: {citation}_"

        if name and not text.startswith(name):
            prefix = f"{name}, " if lang == "en" else f"{name}, "
            text = prefix + text[0].lower() + text[1:] if text else prefix

        # Disclaimer only on substantive legal answers, not small-talk / fallbacks.
        text = f"{text}\n\n{self._disclaimer(lang)}"

        return self._reply(text, intent=intent, lang=lang, source="answer_bank",
                           citation=citation)

    # ------------------------------------------------------------------
    # Gibberish detector for generator output
    # ------------------------------------------------------------------
    @staticmethod
    def _looks_like_gibberish(text: str, lang: str) -> bool:
        """Heuristics that catch degenerate generator output. Any ONE hit
        is enough to bail — false positives just mean we fall back to
        answer_bank, which is still a good answer.

        Signs of a diverged fine-tune:
          1. A single word repeats > 5 times in a row (e.g. "proiect proiect proiect ...")
          2. The top-5 words together account for > 60 % of all words
          3. The share of ASCII letters is very low for an EN answer
             (real EN answers should be >70 % ASCII letters/spaces)
          4. No word in the retrieved statute's known vocabulary appears
             (severance/ELRA/CMA/kifuta/sheria/... — a real legal answer
             should mention at least one)
        """
        import re
        from collections import Counter

        stripped = text.strip()
        if not stripped:
            return True

        words = re.findall(r"[A-Za-zÀ-ſ]+", stripped)
        if len(words) < 3:
            return True

        # 1. immediate repetition
        for i in range(len(words) - 5):
            if len(set(words[i:i+6])) == 1:
                return True

        # 2. narrow vocabulary (only apply to LONGER outputs — short answers
        #    naturally have concentrated vocabulary and would false-positive)
        if len(words) >= 20:
            counts = Counter(w.lower() for w in words)
            top5_share = sum(c for _, c in counts.most_common(5)) / len(words)
            if top5_share > 0.70:      # was 0.60 — raised so terse legal prose passes
                return True

        # 3. ASCII share (catches the "proiect universitaire Bhutan" gibberish
        #    that pulls non-ASCII multilingual tokens). Only for longer outputs.
        if len(stripped) >= 100:
            ascii_chars = sum(1 for c in stripped if c.isascii())
            if ascii_chars / len(stripped) < 0.60:
                return True

        # 4. no legal vocabulary at all
        LEGAL_MARKERS = {
            # institutions / statutes
            "elra", "cma", "wcf", "wca", "nssf", "nhif", "osha", "lla", "lia",
            "gn",  "tucta", "paye", "sdl", "tra",
            # people
            "labour", "labor", "employ", "employer", "employee", "worker",
            "boss", "supervis", "colleague", "meneja",
            # money
            "wage", "salary", "salaries", "pay", "paid", "unpaid",
            "minim", "tzs", "shilling", "gratu", "bonus",
            "sever", "kifuta", "kiinua",
            # time
            "notice", "notisi", "day", "days", "week", "weeks", "month", "months",
            "siku", "wiki", "mwezi", "miezi",
            # actions
            "dismiss", "termin", "resign", "fire", "retrench", "redundan",
            "kufukuz", "kujiuzul", "kupunguzw",
            # objects
            "contract", "leave", "mkataba", "likizo",
            # remedies / procedures
            "compens", "damage", "penal", "fidia", "malipo", "adhabu",
            "form", "fomu", "file", "filing", "hearing", "mediation",
            # rights language
            "sheria", "haki", "ajira", "kazi", "mfanya", "mwajiri",
            "mshahara", "mahakama",
        }
        low = stripped.lower()
        if not any(marker in low for marker in LEGAL_MARKERS):
            return True

        return False

    # ------------------------------------------------------------------
    # Layer-4 helper: RAG retrieve + flan-t5 generate
    # ------------------------------------------------------------------
    def _try_generate(
        self,
        user_text: str,
        intent: str,
        lang: str,
        name: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Return a fully-formatted reply dict if RAG+generator produced one,
        else None so the caller falls back to answer_bank.

        Contract:
          * RAG index missing → chunks == [] → skip generator, return None
          * generator model missing → generator.generate() returns None → return None
          * both work → format answer with citations + disclaimer, return reply
        """
        try:
            chunks: List[Dict[str, Any]] = rag.retrieve(user_text, k=5, lang=lang)
        except Exception as e:
            log.error("RAG retrieval error: %s", e)
            return None
        if not chunks:
            return None

        try:
            generated = generator.generate(user_text, intent, chunks, lang=lang)
        except Exception as e:
            log.error("Generator error: %s", e)
            return None
        if not generated or len(generated.strip()) < 20:
            # Guard against empty / degenerate generations. Fall back to answer_bank.
            return None

        # ── Gibberish detector ────────────────────────────────────────────
        # A diverged fine-tune (bad LR, fp16 instability) produces output that
        # PASSES the length gate but is trash. Signs: heavy word repetition,
        # tokens outside the expected language alphabet. If we suspect trash,
        # log it and fall back to answer_bank so the user never sees it.
        if self._looks_like_gibberish(generated, lang):
            log.warning(
                "Generator produced gibberish for intent=%s lang=%s. "
                "Falling back to answer_bank. Sample: %r",
                intent, lang, generated[:80],
            )
            return None

        text = generated.strip()

        # Attach citations from top retrieved chunks (deduped, in order).
        seen: set = set()
        cites: List[str] = []
        for c in chunks[:3]:
            cit = c.get("citation", "").strip()
            if cit and cit not in seen:
                cites.append(cit); seen.add(cit)
        citation_line = " · ".join(cites)
        if citation_line and citation_line.lower() not in text.lower():
            text = f"{text}\n\n_Ref: {citation_line}_"

        if name and not text.startswith(name):
            prefix = f"{name}, "
            text = prefix + text[0].lower() + text[1:] if text else prefix

        text = f"{text}\n\n{self._disclaimer(lang)}"

        return self._reply(
            text,
            intent=intent, lang=lang, source="rag+generator",
            citation=citation_line,
            retrieved=[{"id": c.get("id", ""), "score": round(c.get("score", 0.0), 3)}
                        for c in chunks[:3]],
        )

    @staticmethod
    def _disclaimer(lang: str) -> str:
        if lang == "sw":
            return ("_Kanusho: Hiki ni chombo cha taarifa kinachoendeshwa na AI "
                    "na hakiwakilishi ushauri rasmi wa kisheria. Kwa masuala rasmi, "
                    "wasiliana na wakili aliyesajiliwa._")
        return ("_Disclaimer: This is an AI-powered informational tool and does not "
                "constitute formal legal advice. For official matters, consult a "
                "registered advocate._")

    # ------------------------------------------------------------------
    # reply helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _reply(text: str, **extra: Any) -> Dict[str, Any]:
        r = {"text": text}
        r.update(extra)
        return r

    @staticmethod
    def _greeting_text(lang: str, name: Optional[str], turn_count: int) -> str:
        # First contact vs returning user in the same session.
        if turn_count > 0:
            if lang == "sw":
                return (f"Karibu tena{', ' + name if name else ''}. "
                        "Una swali gani la sheria ya ajira?")
            return (f"Welcome back{', ' + name if name else ''}. "
                    "What employment question can I help with?")
        if lang == "sw":
            return ("Habari! Mimi ni Sheria-Bot, msaidizi wako wa sheria ya ajira Tanzania. "
                    "Ninaweza kukusaidia vipi?")
        return ("Hi! I'm Sheria-Bot, your Tanzania employment-law assistant. "
                "What employment question can I help with?")

    @staticmethod
    def _thanks_reply(lang: str, name: Optional[str]) -> str:
        if lang == "sw":
            return f"Karibu{', ' + name if name else ''}. Uko na swali lingine?"
        return f"You're welcome{', ' + name if name else ''}. Any other question?"

    @staticmethod
    def _goodbye_reply(lang: str, name: Optional[str]) -> str:
        if lang == "sw":
            return f"Kwaheri{', ' + name if name else ''}. Kuwa salama."
        return f"Goodbye{', ' + name if name else ''}. Take care."

    @staticmethod
    def _out_of_scope_reply(lang: str, name: Optional[str]) -> str:
        """The bot refuses to answer non-employment questions, rather than
        forcing a wrong legal answer. Fires on ANY out-of-scope trigger:
        off-topic phrase, no domain vocab, low BERT confidence, or torn
        top-1/top-2 predictions."""
        if lang == "sw":
            return (f"{name + ', ' if name else ''}"
                    "samahani, mimi ni msaidizi wa sheria ya ajira ya Tanzania tu. "
                    "Ninaweza kusaidia kuhusu mshahara, mkataba, kufukuzwa kazi, "
                    "likizo, CMA, na masuala mengine ya kazi. Tafadhali uliza swali "
                    "linalohusu ajira au kazi.")
        return (f"{name + ', ' if name else ''}"
                "I'm sorry — I'm a Tanzania employment-law assistant only. "
                "I can help with wages, contracts, dismissal, leave, CMA filings, "
                "and other workplace matters. Please ask a question about "
                "employment or workplace issues.")

    @staticmethod
    def _unknown_intent_reply(lang: str, name: Optional[str]) -> str:
        if lang == "sw":
            return (f"{name + ', ' if name else ''}"
                    "Ninaweza kusaidia kuhusu masuala ya sheria ya ajira Tanzania. "
                    "Tafadhali eleza swali lako kwa undani zaidi.")
        return (f"{name + ', ' if name else ''}"
                "I can help with Tanzania employment-law issues. "
                "Please tell me more about your situation.")


# module-level singleton
_engine: Optional[AnswerEngine] = None


def get_engine() -> AnswerEngine:
    global _engine
    if _engine is None:
        _engine = AnswerEngine()
    return _engine
