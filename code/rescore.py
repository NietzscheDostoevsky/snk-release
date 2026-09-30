"""
Tiered Probe-B scorer (Tier 0 / Tier 1 / Tier 2).

Tier 0  exact word-boundary match against the single canonical synonym.
Tier 1  same criterion under Snowball stemming.
Tier 2  stemmed match against the curated multi-synonym key (the primary metric).

Usage, from the repository root:
    python code/rescore.py [definitions.csv ...]

With no argument it scores the raw definition outputs shipped in
raw/experiment-34/. Each input file needs `word` and `output` columns; a
`model` column is carried through when present. Scored rows are written to
stdout as CSV and per-model tier means are printed to stderr.

The scored outcomes for the full corpus are released as
results/probeB_rescored_long.csv; the raw generations they were computed from
are not part of this package, so this script reproduces the scorer, not the
whole corpus.
"""
import pandas as pd, re, os, sys
from nltk.stem.snowball import SnowballStemmer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multi_synonyms import EXPANSIONS

stem = SnowballStemmer('english').stem
TOK = re.compile(r"[a-z]+(?:'[a-z]+)?")

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rare   = pd.read_csv(os.path.join(HERE, 'data', 'rare_words_429.csv'))
common = pd.read_csv(os.path.join(HERE, 'data', 'common_words_100.csv'))
syn_map = dict(zip(rare.word, rare.common_synonym))
syn_map.update(dict(zip(common.word, common.common_synonym)))

def score_row(word, out):
    out_l = str(out).lower()
    toks = set(TOK.findall(out_l))
    stems = {stem(t) for t in toks}
    orig = str(syn_map[word]).lower()
    t0 = int(re.search(rf'\b{re.escape(orig)}\b', out_l) is not None)
    def match_term(term):
        term = term.lower().strip()
        if ' ' in term or '-' in term:
            # phrase: match all content words' stems appearing as a contiguous-ish substring;
            # use simple substring on normalized text plus stem-phrase fallback
            norm = re.sub(r'[^a-z ]',' ', out_l)
            if term.replace('-',' ') in re.sub(r'\s+',' ',norm): return True
            words = [w for w in re.split(r'[ -]', term) if w]
            stem_phrase = ' '.join(stem(w) for w in words)
            out_stemmed = ' '.join(stem(t) for t in TOK.findall(out_l))
            return stem_phrase in out_stemmed
        return stem(term) in stems
    t1 = int(t0 or match_term(orig))
    expanded = [orig] + [s for s in EXPANSIONS.get(word,'').split('|') if s]
    t2 = int(t1 or any(match_term(s) for s in expanded))
    return t0, t1, t2

DEFAULT = [os.path.join(HERE, 'raw', 'experiment-34', 'definitions.csv')]

def main(paths):
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df[['t0', 't1', 't2']] = pd.DataFrame(
            [score_row(r.word, r.output) for r in df.itertuples()], index=df.index)
        df['source'] = os.path.basename(os.path.dirname(p)) or os.path.basename(p)
        keep = [c for c in ('source', 'model', 'word', 't0', 't1', 't2') if c in df.columns]
        frames.append(df[keep])
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(sys.stdout, index=False)
    by = 'model' if 'model' in out.columns else 'source'
    print(out.groupby(by)[['t0', 't1', 't2']].mean().round(4).to_string(), file=sys.stderr)

if __name__ == '__main__':
    main(sys.argv[1:] or DEFAULT)
