"""Knowledge-graph embeddings -- deliberately constrained to suggesting.

Read this before using it.

A KGE model learns vectors for entities and relations such that plausible
triples score higher than implausible ones. It is a *statistical prior over
graph shape*. It knows nothing about our corpus, cannot read a passage, and
cannot tell a true relation from a well-shaped one. Putting its output
anywhere near a grounding decision would undo the entire project.

So this module offers exactly two operations, and neither returns an edge:

    suggest_relations()  -> ranked candidates a human or a search step may pursue
    suggest_identities() -> candidate cross-document identity links, as questions

Both emit `OpenQuestion`s or ranked candidates. There is no function here that
adds to a KG. The monotonicity rule from the validation design holds: a
non-deterministic component may subtract or ask, never admit.

The honest limitation, stated once and loudly: our graph is small. KGE methods
were built for graphs with 10^4-10^7 triples. Below roughly a thousand, the
embeddings are mostly noise and the rankings mostly reflect node degree. The
guard below refuses to pretend otherwise. The two settings where this earns its
place are (a) after moving to a large corpus, and (b) as a candidate generator
for entity resolution, where a cheap noisy ranking still beats comparing every
pair.

    pip install pykeen torch

Self-test (runs without pykeen):  python3 kge.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from kg import KG
from schema import OpenQuestion

#: Below this, rankings are dominated by degree and random initialisation.
#: Not a hard floor in the library -- a hard floor in what you may claim.
MIN_VIABLE_TRIPLES = 1000

#: Below this, do not report numbers outside the team at all.
MIN_REPORTABLE_TRIPLES = 5000


@dataclass
class ScaleVerdict:
    n_triples: int
    n_entities: int
    viable: bool
    reportable: bool
    message: str


def assess_scale(kg: KG, grounded_only: bool = True) -> ScaleVerdict:
    """Say plainly whether this graph can support embedding at all."""
    triples = kg.to_pykeen_triples(grounded_only=grounded_only)
    ents = {e for t in triples for e in (t[0], t[2])}
    n, m = len(triples), len(ents)
    if n < MIN_VIABLE_TRIPLES:
        msg = (f"{n} triples over {m} entities. Below {MIN_VIABLE_TRIPLES} the "
               f"embedding mostly memorises degree. Usable as a candidate "
               f"generator; not usable as evidence of anything, and not "
               f"reportable as a result.")
    elif n < MIN_REPORTABLE_TRIPLES:
        msg = (f"{n} triples over {m} entities. Enough to rank candidates "
               f"internally. Still thin for a published number -- say so if "
               f"you publish one.")
    else:
        msg = f"{n} triples over {m} entities. Within normal KGE range."
    return ScaleVerdict(n, m, n >= MIN_VIABLE_TRIPLES,
                        n >= MIN_REPORTABLE_TRIPLES, msg)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


@dataclass
class Trained:
    """A trained embedding plus the facts needed to interpret it honestly."""
    pipeline: Any
    scale: ScaleVerdict
    model_name: str
    trained_on_grounded_only: bool

    @property
    def model(self):
        return self.pipeline.model

    @property
    def factory(self):
        return self.pipeline.training


def train(kg: KG, model_name: str = "RotatE", epochs: int = 200,
          dim: int = 128, device: str | None = None,
          grounded_only: bool = True, random_seed: int = 0) -> Trained:
    """Train on the grounded subgraph.

    grounded_only defaults to True for a reason that is easy to lose: derived
    edges are the output of our own inference rules. Train on them and the
    embedding learns our rules, then scores our rules as plausible, and the
    circularity is invisible in the metrics.

    RotatE is the default because it handles symmetry, antisymmetry, inversion
    and composition, and composition is the property our path equations care
    about. TransE is the cheap comparison; ComplEx is the strong baseline.
    """
    from pykeen.pipeline import pipeline
    from pykeen.triples import TriplesFactory
    import numpy as np

    scale = assess_scale(kg, grounded_only)
    triples = kg.to_pykeen_triples(grounded_only=grounded_only)
    if not triples:
        raise ValueError("no triples to train on")

    tf = TriplesFactory.from_labeled_triples(np.array(triples, dtype=str))
    train_tf, test_tf, valid_tf = tf.split([0.8, 0.1, 0.1], random_state=random_seed)

    result = pipeline(
        training=train_tf, testing=test_tf, validation=valid_tf,
        model=model_name,
        model_kwargs={"embedding_dim": dim},
        training_kwargs={"num_epochs": epochs, "use_tqdm_batch": False},
        random_seed=random_seed,
        device=device,
    )
    return Trained(result, scale, model_name, grounded_only)


# --------------------------------------------------------------------------- #
# The only two things this module is allowed to do
# --------------------------------------------------------------------------- #


@dataclass
class Candidate:
    s: str
    p: str
    o: str
    score: float
    rank: int

    def __repr__(self):
        return f"#{self.rank} ({self.s} -{self.p}-> {self.o}) {self.score:+.3f}"


def suggest_relations(trained: Trained, head: str, predicate: str,
                      top_k: int = 5,
                      exclude: Iterable[str] = ()) -> list[Candidate]:
    """Rank plausible tails for (head, predicate, ?).

    These are *leads for the evidence-search step*, not findings. The correct
    consumer is a retrieval pass that goes looking for a passage supporting the
    candidate; if it finds one, the edge is grounded by that passage and not by
    this ranking. If it finds nothing, the candidate dies.
    """
    import torch

    f = trained.factory
    if head not in f.entity_to_id or predicate not in f.relation_to_id:
        return []
    skip = set(exclude)

    hr = torch.tensor([[f.entity_to_id[head], f.relation_to_id[predicate]]])
    with torch.no_grad():
        scores = trained.model.score_t(hr).squeeze(0)

    id_to_entity = {v: k for k, v in f.entity_to_id.items()}
    order = torch.argsort(scores, descending=True).tolist()
    out: list[Candidate] = []
    for idx in order:
        name = id_to_entity[idx]
        if name == head or name in skip:
            continue
        out.append(Candidate(head, predicate, name, float(scores[idx]), len(out) + 1))
        if len(out) >= top_k:
            break
    return out


def suggest_identities(trained: Trained, participants: Iterable[str],
                       top_k: int = 3, threshold: float = 0.90
                       ) -> list[OpenQuestion]:
    """Candidate cross-document identity links, emitted as questions.

    This targets the ES2002 failure directly: speaker channel letters are
    document-local, so `ES2002d.C` and `ES2002c.C` can be different people
    while every citation on both is correct.

    Embedding similarity is a *cheap pairing heuristic* -- two participants
    with similar neighbourhoods are worth comparing. It is not identity
    evidence and this function never returns an edge. Each pair comes back as
    an OpenQuestion for G7, which the evidence-search step or a human resolves.
    """
    import torch

    f = trained.factory
    ids = [(p, f.entity_to_id[p]) for p in participants if p in f.entity_to_id]
    if len(ids) < 2:
        return []

    emb = trained.model.entity_representations[0](
        indexer=None, indices=torch.tensor([i for _, i in ids])
    ).detach()
    if emb.is_complex():
        emb = torch.view_as_real(emb).flatten(start_dim=1)
    emb = torch.nn.functional.normalize(emb.float(), dim=1)
    sim = emb @ emb.T

    out: list[OpenQuestion] = []
    for a in range(len(ids)):
        pairs = sorted(
            ((float(sim[a][b]), ids[b][0]) for b in range(len(ids)) if b != a),
            reverse=True)[:top_k]
        for score, other in pairs:
            if score < threshold:
                continue
            out.append(OpenQuestion(
                id=f"oq:kge:{ids[a][0]}:{other}",
                question=(f"Are {ids[a][0]} and {other} the same person? "
                          f"Embedding neighbourhood similarity {score:.3f}. "
                          f"This is a pairing heuristic, not evidence -- find "
                          f"a passage that identifies them, or leave them "
                          f"distinct."),
                gate="G7",
                attach_to=ids[a][0],
                missing="identity_evidence",
                resolution="C1_find_evidence",
            ))
    return out


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from schema import Citation

    corpus = {"m1": "Craig wants a bigger power button on the remote control."}
    c = Citation("m1", 0, len(corpus["m1"]), corpus["m1"])

    kg = KG()
    kg.add_grounded("req:1", "serves", "need:1", c, corpus)
    kg.add_grounded("need:1", "held_by", "stakeholder:craig", c, corpus)
    kg.add_grounded("req:1", "raised_by", "stakeholder:craig", c, corpus)

    print("=" * 68)
    print("SCALE CHECK -- runs before anything is trained")
    print("=" * 68)
    v = assess_scale(kg)
    print(f"  triples={v.n_triples} entities={v.n_entities}")
    print(f"  viable={v.viable}  reportable={v.reportable}")
    print(f"  {v.message}")

    print()
    print("=" * 68)
    print("WHAT THIS MODULE WILL AND WILL NOT DO")
    print("=" * 68)
    print("  suggest_relations()  -> ranked Candidates for the search step")
    print("  suggest_identities() -> OpenQuestions tagged G7")
    print()
    print("  There is no function here that adds an edge to a KG, scores a")
    print("  claim as true, or satisfies a gate. A ranking can start a search;")
    print("  only a located passage can end one.")

    try:
        import pykeen  # noqa: F401
        print("\n  pykeen present -- train(kg) is available.")
    except ImportError:
        print("\n  pykeen not installed here. Install with: pip install pykeen torch")
        print("  Everything above runs without it; only train() needs it.")
