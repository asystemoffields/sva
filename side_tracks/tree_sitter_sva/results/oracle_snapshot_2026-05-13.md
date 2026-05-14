# Tree-Sitter Oracle Snapshot - 2026-05-13

## Run

- Script: `side_tracks/tree_sitter_sva/tree_sitter_oracle_test.py`
- Corpus: first `12` Python files under `experiments/`
- Parsed files: `12`
- Tokens: `35,871`
- Query samples: `3,000`
- Targets: `9,125`
- Target relation: for each identifier occurrence, recover recent previous occurrences of the same identifier
- Tree route units: selected tree-sitter Python syntax nodes up to `256` tokens
- Fixed route units: chunks of `64`, `128`, and `256` tokens

Command:

```powershell
$dep = Join-Path $env:TEMP 'sva_tree_sitter_deps'
New-Item -ItemType Directory -Force -Path $dep | Out-Null
python -m pip install --quiet --target $dep -r side_tracks\tree_sitter_sva\requirements.txt
$env:PYTHONPATH = $dep
python side_tracks\tree_sitter_sva\tree_sitter_oracle_test.py --paths experiments --max-files 12 --max-queries 3000 --budgets 1,2,4,8 --chunk-tokens 64,128,256
```

## Results

| Router | Open units | Recall | Avg opened tokens | P50 opened tokens | P95 opened tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| tree-sitter nodes | 1 | 0.570521 | 45.883 | 16.000 | 179.000 |
| tree-sitter nodes | 2 | 0.791562 | 59.655 | 28.000 | 208.050 |
| tree-sitter nodes | 4 | 0.978192 | 73.630 | 49.000 | 233.050 |
| fixed 64-token chunks | 1 | 0.629041 | 56.883 | 64.000 | 64.000 |
| fixed 64-token chunks | 2 | 0.872438 | 95.389 | 102.000 | 128.000 |
| fixed 64-token chunks | 4 | 1.000000 | 120.221 | 113.000 | 256.000 |
| fixed 128-token chunks | 1 | 0.720000 | 105.383 | 128.000 | 128.000 |
| fixed 128-token chunks | 2 | 0.923397 | 166.248 | 154.000 | 256.000 |
| fixed 128-token chunks | 4 | 1.000000 | 196.072 | 158.000 | 384.000 |
| fixed 256-token chunks | 1 | 0.798356 | 193.810 | 230.000 | 256.000 |
| fixed 256-token chunks | 2 | 0.964493 | 285.120 | 256.000 | 512.000 |
| fixed 256-token chunks | 4 | 1.000000 | 312.768 | 256.000 | 722.150 |

## Interpretation

Tree routing is more token-efficient in this proxy. At four opened units, tree-sitter nodes recover `97.8%` of targets while opening about `74` tokens on average. Fixed 64-token chunks reach perfect oracle recall, but open about `120` tokens on average.

The tradeoff is clear at low budgets. Fixed chunks have higher raw recall because identifiers often sit close together in nearby text, while syntax nodes open smaller spans and miss some cross-statement references. The tree shape starts looking useful when the objective is recall per opened token rather than recall per opened unit.

## Next Test

Use model-derived attention targets instead of same-identifier targets. The clean version is:

1. run SmolLM2 over Python-like source;
2. collect each head's top attention keys;
3. map token positions into tree-sitter syntax nodes;
4. compare tree-node oracle coverage against fixed chunks under the same opened-token budget.
