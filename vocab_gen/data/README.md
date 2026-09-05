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
