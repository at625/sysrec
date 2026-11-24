# Interview prep

Grounded in this repo and its measured results. Numbers are test-window unless stated.

---

## M0: Data and splitting

**Core idea.** A recommender is trained on the past and serves the future, so the evaluation must
too. I use one global time cutoff: train before 2023-04-01, validate on the next quarter, test on the
one after. Every feature is computed "as of" a timestamp using only earlier data.

**Q1. Why not leave-one-out or a random split?**
Both let the model see the future. With leave-last-out, another user's later ratings shape
popularity and item embeddings used to predict an earlier event. A global cutoff is what production
faces, and it makes popularity trends real: recent-popular is a strong baseline here (NDCG@10 0.172),
which a leaky split would hide.

**Q2. How do you know there's no leakage?**
Three layers. (1) The harness (`build_context`) only hands models history before the cutoff and
asserts it; `evaluate` refuses a context whose cutoff differs from the window start. (2) Tests: no test
interaction appears in train; training sequences only use strictly earlier positives; ranking features
are byte-identical when every row at or after the cutoff is perturbed. (3) Sanity: when the ranker gave
a +30% jump I checked the features before believing it.

**Q3. How big are the eval windows, and why does it matter?**
4,076 val users and 4,726 test users have a positive in the window. That's small, so I report
bootstrap CIs and paired tests rather than single numbers.

**Q4. What's a cold-start user/item here?**
945 val and 1,321 test users had no history before the window; 1,924 items rated in test were never
rated before it (1.9% of test targets). Cold users get recent-popular items; cold items still get
embeddings from genre, year and MiniLM text features.

**Q5. How did you keep memory in check?**
Polars lazy scans and streaming sinks, int32 IDs and float32 ratings: preparing 32M rows peaks at
3.2 GB RSS in 5 seconds.

**Limitations / next.** MovieLens ratings are explicit feedback from a site with its own recommender;
implicit signals (watches, dwell) would be closer to production.

---

## M1: Metrics and baselines

**Core idea.** Score the full catalog (87,585 items), mask what the user already saw, and measure
Recall/NDCG/HitRate/MRR at K. Compare against simple baselines before believing a deep model.

**Q1. Why full-catalog instead of sampled metrics?**
Sampled metrics (rank the positive among 100 random items) can change which model wins, not just
the scale (Krichene & Rendle 2020). Scoring 87k items for 5k users is cheap, so there is no reason to sample.

**Q2. Recall@K vs NDCG@K: when do you use which?**
Recall@100 for retrieval (did the right items make it into the candidate set?), NDCG@10 for the
final list (are they near the top?). The two-tower model wins on Recall@100 but not NDCG@10, which is
why a ranker exists.

**Q3. Why did ALS lose to recent popularity?**
Under a time split, much next-quarter activity goes to new and trending titles. ALS trained on 28
years of history learns old tastes; restricting it to 3 years helped (val Recall@100 0.277 to 0.289)
but on test it still only ties recent-popular on Recall@100 (0.279 vs 0.284).

**Q4. What else do you track besides accuracy?**
Catalog coverage and popularity bias: ALS covers 10x more of the catalog in its top 10 than
recent-popular (2.2% vs 0.22%) and recommends far less popular items.

**Q5. How did you tune without overfitting test?**
Every choice (popularity window, ALS grid, two-tower settings, ranker head weight, MMR lambda) is
selected on val; the test window is only used for the final report.

---

## M2: Two-tower retrieval and ANN

**Core idea.** Encode users and items separately into one vector space so retrieval becomes a
nearest-neighbour search; train with in-batch negatives and correct for the fact that popular items
appear as negatives more often (logQ).

**Q1. Walk me through the towers.**
User: user-ID embedding + mean of the last 50 liked items' embeddings + two running stats -> MLP ->
64-d normalised. Item: ID embedding + genres + year bucket + MiniLM text vector of title/genres/tags
-> MLP -> 64-d normalised. Score is cosine / 0.05.

**Q2. What does logQ correction do, and how do you know it matters?**
In-batch negatives are sampled roughly in proportion to item popularity, so the softmax over-penalises
popular items. Subtracting log q(item) from each logit undoes that bias. Ablation: without it
Recall@100 drops from 0.298 to 0.132.

**Q3. Why mix in uniform negatives?**
In-batch negatives rarely include tail items, so the model never learns to push down irrelevant tail
items. Mixed negatives gave a small gain (0.298 to 0.301) at about 2x item-tower compute per step.

**Q4. Exact vs approximate search?**
At 87k x 64-d, exact search is 0.85 ms per query on one thread, so it's affordable. HNSW at
efSearch=128 keeps 98.8% of exact top-100 at 0.07 ms. At 10M+ items exact search stops being an option,
and efSearch / nprobe are the knobs that trade recall for latency.

**Q5 (system design). "Design YouTube recommendations." How does this map?**
Candidate generation: several sources (two-tower here; also co-watch, subscriptions, trending) each
returning hundreds of items from millions via ANN. Ranking: a feature-rich model (DCN-v2 here) over
about 1,000 candidates with user, item, context and cross features, trained on watch-time or engagement
labels. Re-ranking: diversity, freshness, policy and business rules (MMR here). Serving: precomputed
item embeddings in an ANN index, user features from a feature store (Redis here), models exported for
fast inference (ONNX here). Evaluation: offline replay with time splits, then online A/B tests with
guardrails. What I'd add at YouTube scale: multiple retrieval sources, watch-time objectives,
real-time features, and position-bias correction.

**Limitations / next.** Mean pooling ignores order; a SASRec tower would use the sequence. The model
has no notion of time, which is why trends had to come from the ranker.

---

## M3: Ranking

**Core idea.** Retrieval is cheap and recall-oriented; ranking is expensive and precision-oriented.
Train the ranker on retrieval's own candidates (hard negatives) with rich point-in-time features.

**Q1. How is the training data built without leakage?**
Snapshot at cutoff c: retrieval models and features use history < c; labels are positives in
[c, c + 91 days). The ranker evaluated on test is trained on the snapshot whose labels are the val
window, so it never sees test.

**Q2. Why retrieval candidates as negatives instead of random items?**
Random items are easy to reject; the ranker would learn nothing useful for separating the 200 items
it will actually see. Candidate negatives look relevant, which is the real decision.

**Q3. What did the ranker gain, and which features mattered?**
NDCG@10 0.162 to 0.210 (DCN-v2) and 0.206 (LightGBM). Top LightGBM gain: retrieval rank (23%), ALS
score (11%), 7- and 30-day popularity. Dropping user-activity features costs the most (0.206 to 0.185).

**Q4. Did multi-task learning help?**
Honestly, no. Val chose a weight of 0 for the rating head, and a single-task model matches it on
val and is slightly better on test (0.212). Only ~8% of candidate rows have a rating label, so the
auxiliary signal is thin. I kept the head because predicted rating is useful elsewhere (RMSE 0.86).

**Q5. DCN-v2 vs LightGBM: which would you ship?**
They're within 0.004 NDCG@10. LightGBM is easier to debug and has free feature importance; DCN-v2
handles embeddings and scales better with more data. With this data size I'd ship either and use
LightGBM as the debugging reference.

**Limitations / next.** Candidate recall is only 42% at 200, which caps the whole system: a
candidate mixer (two-tower + trending + ALS) is the next lever.

---

## M4: Re-ranking

**Core idea.** Accuracy alone produces near-duplicate lists. MMR trades a bit of relevance for
items that are less similar to what's already in the list.

**Q1. How does MMR work?**
Greedy: at each slot pick argmax of lambda * relevance - (1 - lambda) * max similarity to already
selected items. lambda=1 is pure relevance.

**Q2. How did you choose lambda?**
A rule declared before looking at test: the most diverse point on val within 1% of no-MMR NDCG@10.
That's lambda=0.8: diversity +10% on val at no NDCG cost; on test, -0.0003 NDCG (not significant)
for diversity 0.300 to 0.321.

**Q3. Why is the genre cap off?**
Max 3 per genre cost ~9% NDCG@10 on val for +0.08 genre coverage. It's a business decision, so
it's a config flag.

**Q4. Doesn't offline evaluation always penalise diversity?**
Yes: logged data reflects what users did when shown less diverse lists. That's why the chosen point
is "free diversity" offline, and the real test would be online.

**Q5. What other re-ranking rules would production add?**
Freshness boosts, de-duplication of franchises, policy/safety filters, creator fairness, and
"already-watched" suppression across sessions.

---

## M5: Serving

**Core idea.** Precompute everything that doesn't depend on the request (item embeddings, ANN
index, feature snapshot); at request time only run the user tower, one ANN query, one batched
ranker call and MMR.

**Q1. Walk through a request and its latency budget.**
At 300 req/s (stage p50s): Redis user fetch 0.28 ms, user tower + FAISS 1.0 ms, item features 0.02 ms
(cached), DCN-v2 on 200 candidates 0.6 ms, MMR 0.4 ms; end to end 5.7 ms p50, 19.6 ms p99.

**Q2. What was the bottleneck and how did you fix it?**
Baseline (1 worker) saturated at ~119 req/s. (1) Cached item features in process instead of a
200-key Redis MGET per request: item stage 1.06 to 0.03 ms, 1 worker sustains 200 req/s. (2) Three
workers made it worse at first: OpenBLAS in each worker spawned 4 threads, 12 threads on 4 cores, MMR
went from 0.4 ms to 11+ ms. One BLAS thread per worker fixed it: 300 req/s at 19.6 ms p99.

**Q3. How do you guarantee online = offline?**
Same feature functions and transform code online and offline, models exported once from the same
snapshot, deterministic tie-breaking. Measured: 300/300 identical top-10 lists with exact retrieval,
99% with HNSW.

**Q4. How do you avoid recommending seen items with ANN?**
Over-fetch K + |seen| from FAISS and filter, with efSearch raised to cover it. That makes the
result identical to "mask then top-K".

**Q5. How would this scale?**
Stateless API replicas behind a load balancer; Redis cluster or a managed feature store; ANN on a
dedicated service with sharded indexes; batch requests to the ranker on GPU; cache user embeddings
for active users.

---

## M6: Simulated A/B

**Core idea.** Replay the held-out quarter, score each variant on the same users, and use paired
bootstrap to get honest uncertainty.

**Q1. What did you find?**
Full stack vs recent-popular: +0.039 NDCG@10 (+22.5%), 95% CI [0.034, 0.043], p < 0.001. MMR vs
no-MMR: no significant NDCG change.

**Q2. Why paired bootstrap?**
Users differ hugely in difficulty. Pairing on the same users cancels that variance, giving much
tighter intervals for the difference than comparing two independent CIs.

**Q3. What can't an offline replay tell you?**
Feedback loops, position bias, how users respond to diversity or novelty, long-term retention. Users
only rated what the old system showed them.

**Q4. How would you run the real experiment?**
Randomise users, pick one primary metric (e.g. weekly engaged sessions) and guardrails (latency,
diversity, complaint rate), compute the needed sample size from the offline effect size, run for full
weekly cycles, and check for novelty effects.

**Q5. Why might online results differ from offline?**
The offline label is "did the user rate it in the next quarter on a site with a different
recommender". A model that recommends the same popular items people would find anyway looks great
offline but adds little online.

---

## What I'd do next

1. Raise candidate recall with a mixer of sources (two-tower, trending, ALS).
2. Sequential user tower (SASRec) and time-aware retrieval.
3. Streaming features (a "user just watched X" event updating Redis).
4. A real online test with guardrails.
