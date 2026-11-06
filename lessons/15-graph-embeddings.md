# Lesson 15 — Graph embeddings: who and what belong together?

Lesson 5 embedded *text*: "Eve trains ranking models" lands near "recommendation engine"
because the words are about the same thing. This lesson embeds *structure*. Two people end up
close because they connect to the same places, people and memories, whatever the words say.

The question for the companion is: **"who else is part of this?"** The user mentions Cy, the
route setter. Cy belongs with the climbing crowd, Ana and Ben, even though no memory mentions
Cy together with either of them.

## The life graph as GDS sees it — [`companion/insights.py`](../companion/insights.py)

GDS algorithms run on an in-memory **projection** (Lesson 4). This one is built per user with a
**Cypher projection** (the `gds.graph.project(...)` aggregation function):

```cypher
MATCH (s:Entity|Memory {user: $user}) WHERE NOT s:User AND <current, not sensitive, not protected>
OPTIONAL MATCH (s)-[r:ABOUT|RELATED]->(t:Entity {user: $user}) WHERE NOT t:User ...
WITH gds.graph.project($graph, s, t,
       {sourceNodeLabels: labels(s), targetNodeLabels: labels(t)},
       {undirectedRelationshipTypes: ['*']}) AS g
```

What's left out, and why:

| Left out | Why |
|---|---|
| the **user's own node** | it's connected to almost everything, so everything would look related through it |
| retired, expired, **sensitive** memories | an "insight" built on a health memory is still a leak |
| **boundary-protected** entities and their memories | "related to Ben" must not bring back the ex the user asked you not to mention |

The projection gets a random name and is **always dropped** in a `finally`. GDS graphs live in
the heap until dropped, so a leak per call would eventually exhaust it.

## Two ways to measure "related"

**Node similarity (Jaccard)** compares *direct neighbours*:

```
similarity(Cy, Ana) = |memories mentioning both| / |memories mentioning either|
```

It's exact and easy to explain ("they're in 2 of the same 3 memories"). But Cy's only memory is
"Cy sets the routes at Boulder Gym", so Cy shares no neighbour with Ana or Ben: 0.

**FastRP + kNN** gives every node a vector:

1. Each node starts with a sparse **random** vector.
2. Iteration 1: a node's vector becomes the average of its neighbours' vectors.
3. Iteration 2: the average of *those*, which reaches two hops away.
4. The embedding is a weighted sum of the iterations. Here it uses `iterationWeights: [0.0,
   1.0, 1.0]`: ignore the random start, count 1 and 2 hops equally.

Cy → *Cy's memory* → Boulder Gym → *Ana and Ben's memory*: within two hops of the memory
layer, Cy's vector picks up the same neighbourhood as Ana's and Ben's. Then **kNN** finds each
entity's nearest vectors.

## Results — [`tests/test_15_graph_embeddings.py`](../tests/test_15_graph_embeddings.py)

The fixture graph is small and hand-made, so the right answer is obvious to a human: a
climbing crowd (Ana, Ben, Cy, Boulder Gym) and a work crowd (Dee, Eli, Acme).

| Related to Cy | FastRP + kNN | Node similarity |
|---|---|---|
| Boulder Gym | **0.879** | 0.333 |
| Ben | **0.731** | 0 |
| Ana | **0.710** | 0 |
| Dee, Eli, Acme (work) | 0.52 | 0 |

```
/related Cy
  Cy ~ Boulder Gym (place)  0.879
  Cy ~ Ben (person)  0.731
  Cy ~ Ana (person)  0.71
```

### Two things I had to measure

- **kNN always fills its top-k.** It returns the *k nearest* vectors, related or not. With
  k = 3, Dee got Acme, Eli and then **Cy**, which is wrong. So results need a cut-off.
- **0.5 means unrelated, not "half related".** Unrelated pairs scored about 0.5 at every
  embedding size I tried (64, 128, 256, 512). That fits kNN reporting cosine rescaled from
  [-1, 1] to [0, 1]. So the cut is 0.6, not 0.1 or 0.3.
- **Embedding size**: at 64 dimensions, Cy ~ Ana was 0.64, too close to the cut. At 128 it's
  0.71 while unrelated pairs stay ≤ 0.53. More dimensions mean less random overlap, at a
  little memory. 128 is cheap for a graph this size.

| Size | Cy ~ Ana | Dee ~ Cy (unrelated) |
|---|---|---|
| 64 | 0.639 | 0.546 |
| 128 | 0.710 | 0.524 |
| 256 | 0.749 | 0.491 |
| 512 | 0.745 | 0.479 |

### Repeatable results

FastRP starts from random vectors and kNN samples candidates. Without `randomSeed` and
`concurrency: 1`, scores, and sometimes the order, change from run to run. A test asserting
"Cy's crowd is Ana and Ben" is only meaningful if it can't change by chance.
`test_results_repeat_with_a_fixed_seed` checks this.

## When to use which

| | Node similarity | FastRP + kNN |
|---|---|---|
| Sees | shared direct neighbours | neighbourhoods 1–2+ hops out |
| Explainable | yes ("3 shared memories") | not directly |
| Deterministic | yes | with a seed and concurrency 1 |
| Good for | "who else was in these memories?" | "who belongs with whom?", features for ML |

In this repo, `/related <name>` in the terminal chat uses FastRP, and `/related` with no name
lists the strongest pairs. Lesson 17's web page shows them too. They're **suggestions**, not
facts: the companion doesn't store "Cy is Ana's friend" because embeddings put them close.

## Test it

```bash
.venv/bin/python -m pytest tests/test_15_graph_embeddings.py -v   # 6 tests, ~2 s, no LLM
```

- Cy's crowd is found by FastRP; node similarity only sees Boulder Gym.
- Dee and Eli only relate to the work crowd (this test caught the top-k filler).
- Same seed, same results.
- A boundary-protected person never appears in any pair.
- No projection is left behind, even when the call fails.

## Takeaways

- Text embeddings say what things are *about*. Graph embeddings say what they're *connected to*.
- Project only what the question needs, and leave out what must not leak (sensitive,
  protected). Always drop the projection.
- kNN always returns k neighbours. Measure what "unrelated" scores, and cut there.
- Fix seeds and concurrency for any randomized algorithm you want to test.

## Next

[Lesson 16](16-personas.md) lets several personas share one memory without sharing secrets.
