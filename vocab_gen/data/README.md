# Vendored lexical data

## `prevalence.csv.gz`

Word prevalence norms — the proportion of people who report knowing a word,
from a crowdsourcing study of over 220,000 participants.

- **Source:** Brysbaert, M., Mandera, P., McCormick, S. F., & Keuleers, E. (2019).
  *Word prevalence norms for 62,000 English lemmas.* Behavior Research Methods,
  51(2), 467–479. https://doi.org/10.3758/s13428-018-1077-9
- **Retrieved:** 2026-09-05 from https://osf.io/nbu9e/download
  (`English_Word_Prevalences.xlsx`, 8.5 MB), located via the dataset's NoRaRe
  entry. The `crr.ugent.be` paths that older references give are dead.
- **Rows:** 61,855 of the source's 61,856 (one row lacks a word or prevalence).

Vendored rather than downloaded at runtime so results are reproducible from a
commit and the tool works offline. Converted from xlsx to gzipped CSV to drop
8.5 MB to 734 KB; `openpyxl` was used once for the conversion and is deliberately
not a project dependency.

Columns, unchanged from the source except for lowercasing the word and rounding:

| column | meaning |
|---|---|
| `word` | the lemma, lowercased |
| `pknown` | proportion of participants who knew it, 0–1 |
| `prevalence` | `pknown` on a probit scale, roughly −2.5 to +2.6 |
| `nobs` | participants who saw this word; low values are noisier estimates |
| `zipf_us` | the source's own SUBTLEX-US (subtitle) Zipf frequency |

`zipf_us` is worth keeping alongside `wordfreq`'s mixed-corpus Zipf: subtitles are
a spoken register, so the difference between the two is a register signal that
costs nothing extra.

**Note that prevalence saturates.** Above roughly Zipf 3.5 nearly every word is
known by nearly everyone (median prevalence 2.17 in the 3–4 band, 2.33 in 4–5),
so the measure carries almost no information there. It discriminates in the 0–3
band, which is where this project's vocabulary sits.


## `wordlists.csv.gz`

How many of sixteen independent GRE-style vocabulary lists contain each word —
`word`, `n_lists` — for 9,552 words. **Derived counts only:** no definitions, no
per-source membership, nothing of any one compilation's selection or ordering.
Individual words are not copyrightable; a curated compilation can carry thin
rights, so only the cross-source agreement is kept.

- **Source:** aggregated from https://github.com/Xatta-Trone/gre-words-collection
  (GregMat, Prepscholar, Magoosh, Powerscore, Barron's, Greenlight, Manhattan
  Prep, Vocabulary.com and others), retrieved 2026-09-05.
- Entries on a single list are kept in the file but ignored by `lexicon.py`
  (`MIN_LISTS = 2`): they are one compiler's taste, and are dominated by proper
  nouns and typos.

Why these rather than an academic list. Measured against the real deck:

| list | size | overlap with deck |
|---|---|---|
| Academic Word List | 3,108 | **0 words (0%)** |
| Academic Vocabulary List | 18,561 | 211 (25%) |
| these GRE lists | 9,474 | **642 (75%)** |

The academic lists are built for students entering university and are pitched far
below this deck — their first suggestions are `typically`, `stairway`, `forehead`.

## Norms that were tried and are deliberately *not* vendored

Age-of-acquisition and concreteness norms were fetched, measured, and dropped.
Recorded here because they are what established the design, and because the AoA
file is hard to find again:

- **AoA:** Kuperman, V., Stadthagen-Gonzalez, H., & Brysbaert, M. (2012).
  *Age-of-acquisition ratings for 30,000 English words.* Behavior Research
  Methods, 44(4), 978-990. The `crr.ugent.be` path every reference gives is dead
  and Springer's supplement 403s. It survives in the OSF repository of a 2025
  paper extending the norms, https://osf.io/ch48r/, in
  `AI Generated Print AoA Estimates for Kuperman et al. (2012).xlsx`, whose
  `Kuperman_et_al_2012_AoA` column is the original human data (28,053 lemmas).
- **Concreteness:** Brysbaert, M., Warriner, A. B., & Kuperman, V. (2014).
  *Concreteness ratings for 40 thousand generally known English word lemmas.*
  Behavior Research Methods, 46(3), 904-911. 39,954 lemmas, 1 = abstract,
  5 = concrete, from the Springer supplement
  `13428_2013_403_MOESM1_ESM.xlsx` under DOI 10.3758/s13428-013-0403-5.

Both discriminate the target category well and **both fail on coverage exactly
where it matters** - their missingness is correlated with rarity:

| Zipf band | concreteness | AoA |
|---|---|---|
| 0-2 | 37% | 14% |
| 2-3 | 65% | 63% |
| 3-4 | 93% | 80% |
| 4-8 | 98% | 82% |

Filtering on either would bias candidates toward commoner words, which is
backwards. An embedding model predicting concreteness from them as training
labels reaches held-out Spearman 0.754 on rare words and gives full coverage -
but ranked inside the deck's difficulty band it returns `abidingness`,
`comprehensibleness`, `disadvantageousness`, because nothing in a distributional
score knows which forms people actually write. The curated lists exclude those by
construction, which is why they are what ships and these are not.
