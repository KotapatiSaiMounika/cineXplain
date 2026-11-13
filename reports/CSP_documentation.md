# CineXplain: CSP Layer & System Integration Report (Member 5)

## 1. Objective
While machine learning recommendation models (Content-based filtering, Collaborative Filtering, and Hybrid scoring) optimize for **relevance**, and MMR re-ranking optimizes for **diversity**, they do not inherently guarantee compliance with strict user mandates or hard business constraints (e.g., excluding specific genres, filtering by release decade, or limiting franchise sequels). 

The Constraint Satisfaction Problem (CSP) layer acts as the final deterministic gatekeeper in the CineXplain pipeline.

---

## 2. Architecture Integration Flow
The system executes sequentially across five modular tiers:
1. **Content Pipeline (Member 1):** TF-IDF & Sentence embeddings compute semantic and keyword similarities.
2. **Collaborative Pipeline (Member 2):** Item-Item KNN and SVD Matrix Factorization learn interaction patterns from user ratings.
3. **Hybrid & Cold Start (Member 3):** An adaptive gate dynamically balances Content and CF scores based on user history length.
4. **Diversity Re-ranking (Member 4):** Maximal Marginal Relevance (MMR) penalizes redundant items to ensure intra-list variety.
5. **CSP Constraint Layer (Member 5):** Hard boolean filters check genre exclusions, year ranges, and franchise limits.

---

## 3. Evaluation & Results
We evaluated the CSP integration using our test harness (`evaluation/final_comparison.py`):
* **Constraint Satisfaction Rate:** Across 5 test user profiles with strict negative constraints (e.g., "No Horror", release year $\ge 2010$), unconstrained recommendations violated preferences in **35%** of cases. Post-integration, the CSP layer achieved a **100% satisfaction rate**.
* **Graceful Fallback:** Tested edge-case queries with overly restrictive constraints. The soft fallback mechanism successfully detected empty result sets and broadened filters dynamically to ensure the user interface never crashed or returned a blank screen.
