# Hierarchical Tree SVA Side Track

This is a parked side track for the chunk/tree idea, kept separate from the current hard-negative coarse-PQ mainline.

## Idea

Long-context SVA may need a hierarchy rather than one flat catalog. A query would first route to coarse spans, then open only selected spans and run token-level lookup inside them.

The useful analogy is a queryable parse tree:

```text
document
  section / topic nodes
    chunk nodes
      token leaves
```

Each node probably needs multiple slots, not one summary vector. A 250-token chunk can contain unrelated roles: entity mentions, rare terms, local syntax, temporal facts, and discourse relations. One vector will blur those. A multi-slot node can offer several reasons for the query to open that span.

## First Oracle Test

Before training a router, run an oracle coverage test:

1. Use real SmolLM2 attention top-16 keys.
2. Partition the context into chunks such as `128,250,256,512`.
3. For each query/head, ask how many chunks contain the true top-16 keys.
4. Measure the recall upper bound if an ideal router could open `1,2,4,8,16` chunks.

If a small number of chunks covers most top keys, hierarchical routing is promising. If true keys are scattered across many chunks, the tree needs richer multi-slot nodes, overlapping spans, or a different structure.

## Relationship To Mainline

The mainline hard-negative result says flat compressed retrieval can be made much sharper when trained against its own near misses. This side track asks whether a tree can reduce the search space before token-level SVA runs.

The two ideas can combine later: tree routing picks spans, and hard-negative SVA handles token survival inside opened spans.
