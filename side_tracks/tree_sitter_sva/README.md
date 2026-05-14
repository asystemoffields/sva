# Tree-Sitter SVA Side Track

This folder is a separate lane for testing syntax-tree routing as a possible hierarchy above token-level SVA.

## Question

Can parse-tree spans act as a cheap structured catalog before token-level lookup?

The first test is an oracle coverage check over Python source. For each identifier occurrence, it treats recent previous occurrences of the same identifier as target keys. It then compares two ideal routers:

- fixed token chunks;
- tree-sitter syntax nodes such as functions, classes, assignments, statements, loops, calls, and parameters.

The oracle greedily opens a small number of route units and measures how many target identifier keys become visible, plus how many source tokens were opened.

## First Result

See `results/oracle_snapshot_2026-05-13.md`.

On the first repo-code proxy, tree-sitter nodes were more token-efficient but had slightly lower raw oracle recall than fixed chunks. At four opened units, tree nodes recovered `0.978192` of targets while opening about `73.630` tokens on average. Fixed 64-token chunks recovered `1.000000` while opening about `120.221` tokens.

## Run

From the repository root:

```powershell
python -m pip install -r side_tracks\tree_sitter_sva\requirements.txt
python side_tracks\tree_sitter_sva\tree_sitter_oracle_test.py --paths experiments --max-files 12 --budgets 1,2,4,8 --chunk-tokens 64,128,256
```

## Output

Rows use CSV-like lines:

```text
tree_sitter_oracle_result,router,budget,queries,targets,recall,avg_opened_tokens,p50_opened_tokens,p95_opened_tokens,avg_units_opened
```

Useful readings:

- Higher recall at the same opened-token count means the routing shape is better.
- Tree nodes winning at low budgets means syntax spans concentrate useful references.
- Fixed chunks winning means syntax spans are too broad, too fragmented, or poorly matched to the target relation.

## Scope

This is a code-structure proxy, not the main SVA result. A positive result here would justify a richer tree router test on model attention targets. A weak result would still be useful because it would bound how much structure tree routing gives before learned routing is added.
