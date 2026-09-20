#!/usr/bin/env python3
"""PGLS analysis of lifetime metabolic budget ~ class + log(mass) + log(sleep).

Three approaches reported side-by-side for robustness:
  (A) OLS — no phylogenetic correction
  (B) GLS with taxonomy-derived correlation matrix (Order-level)
  (C) GLS with Open Tree of Life phylogeny (if reachable)

Compares nested models:
  M0: ~ 1
  M1: ~ Class + log(mass)
  M2: ~ Class + log(mass) + log(sleep)
  M3: ~ Class + log(mass) + log(activity)  [activity = 24 - sleep, complementary test]

GRADUATE if sleep coefficient in M2 has p<0.05 AND AIC(M2)-AIC(M1) <= -4.
"""
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import stats
from scipy.linalg import cho_factor, cho_solve

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
NOTES = ROOT / "notes"
NOTES.mkdir(exist_ok=True)

MERGED = DATA / "merged.csv"
RESULTS = NOTES / "pgls_result.txt"


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def gls_fit(X, y, V):
    """Generalised least squares: β = (X' V⁻¹ X)⁻¹ X' V⁻¹ y.

    Returns: beta, se, t, p, loglik, n, k, aic, bic, sigma2, residuals.
    """
    n, k = X.shape
    # Cholesky for stability
    L, low = cho_factor(V, lower=True)
    Vi_X = cho_solve((L, low), X)
    Vi_y = cho_solve((L, low), y)
    XtViX = X.T @ Vi_X
    XtViy = X.T @ Vi_y
    beta = np.linalg.solve(XtViX, XtViy)
    resid = y - X @ beta
    # residual variance estimate (REML-like)
    Vi_resid = cho_solve((L, low), resid)
    sigma2 = float(resid @ Vi_resid) / (n - k)
    cov_beta = sigma2 * np.linalg.inv(XtViX)
    se = np.sqrt(np.diag(cov_beta))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), df=n - k)
    # log-likelihood under multivariate normal with covariance σ²V
    sign, logdet = np.linalg.slogdet(V)
    loglik = -0.5 * (n * np.log(2 * np.pi * sigma2) + logdet + (resid @ Vi_resid) / sigma2)
    aic = 2 * k - 2 * loglik
    bic = k * np.log(n) - 2 * loglik
    return {
        "beta": beta, "se": se, "t": t, "p": p,
        "loglik": float(loglik), "n": n, "k": k,
        "aic": float(aic), "bic": float(bic),
        "sigma2": sigma2, "residuals": resid,
    }


def build_design(df, predictors):
    """Build design matrix with intercept + Class dummy + chosen predictors."""
    n = len(df)
    cols = [np.ones(n)]
    names = ["intercept"]
    # class dummy: 1 for Aves, 0 for Mammalia
    cls = (df["Class"].values == "Aves").astype(float)
    if cls.sum() > 0 and cls.sum() < n:
        cols.append(cls)
        names.append("class_Aves")
    for pname in predictors:
        cols.append(df[pname].values.astype(float))
        names.append(pname)
    X = np.column_stack(cols)
    return X, names


def taxonomic_correlation_matrix(df, lambda_=1.0):
    """V_ij = lambda^d where d is taxonomic distance (0 within order,
    1 within class, 2 between class)."""
    n = len(df)
    V = np.eye(n)
    orders = df["Order"].values
    classes = df["Class"].values
    for i in range(n):
        for j in range(i + 1, n):
            if orders[i] == orders[j]:
                rho = 0.6
            elif classes[i] == classes[j]:
                rho = 0.25
            else:
                rho = 0.0
            V[i, j] = V[j, i] = rho * lambda_
    return V


def opentree_subtree(binomials, timeout=30):
    """Try to fetch a tree for our species via Open Tree of Life."""
    log(f"OTL: matching {len(binomials)} species names")
    try:
        r = requests.post(
            "https://api.opentreeoflife.org/v3/tnrs/match_names",
            json={"names": binomials, "do_approximate_matching": True},
            timeout=timeout,
        )
        r.raise_for_status()
        match = r.json()
        ott_ids = []
        for entry in match.get("results", []):
            if entry.get("matches"):
                ott_ids.append(entry["matches"][0]["taxon"]["ott_id"])
        log(f"OTL: matched {len(ott_ids)} of {len(binomials)} names")
        if len(ott_ids) < 10:
            return None
        r = requests.post(
            "https://api.opentreeoflife.org/v3/tree_of_life/induced_subtree",
            json={"ott_ids": ott_ids},
            timeout=timeout,
        )
        if r.status_code == 400:
            # try with badly-matched filtered out
            r = requests.post(
                "https://api.opentreeoflife.org/v3/tree_of_life/induced_subtree",
                json={"ott_ids": ott_ids, "label_format": "name_and_id"},
                timeout=timeout,
            )
        if r.status_code != 200:
            log(f"OTL induced_subtree HTTP {r.status_code}: {r.text[:200]}")
            return None
        newick = r.json().get("newick")
        log(f"OTL: got tree, {len(newick)} chars")
        return newick
    except Exception as e:
        log(f"OTL failed: {e}")
        return None


def vcv_from_newick(newick, tip_labels):
    """Build VCV matrix from a Newick tree (taxa shared branch length)."""
    try:
        import dendropy
        tree = dendropy.Tree.get(data=newick, schema="newick")
        # Map taxon labels (normalised lowercase) to indices
        label_to_idx = {l.lower().replace(" ", "_"): i for i, l in enumerate(tip_labels)}
        # Build VCV: shared root-to-MRCA branch length for each pair
        leaves = [leaf for leaf in tree.leaf_node_iter()
                  if leaf.taxon and leaf.taxon.label.lower().replace(" ", "_") in label_to_idx]
        if len(leaves) < 10:
            log(f"too few leaves in induced subtree: {len(leaves)}")
            return None
        # Compute root-to-leaf distance and pairwise MRCA depth
        n = len(tip_labels)
        # Root all distances
        root = tree.seed_node
        leaf_to_root = {}
        for leaf in leaves:
            d = 0.0
            node = leaf
            while node.parent_node is not None:
                d += (node.edge_length or 1.0)
                node = node.parent_node
            leaf_to_root[leaf] = d
        max_root = max(leaf_to_root.values()) if leaf_to_root else 1.0
        # Build VCV
        V = np.eye(n) * 1e-6  # tiny ridge for stability
        present = {}
        for leaf in leaves:
            label = leaf.taxon.label.lower().replace(" ", "_")
            idx = label_to_idx.get(label)
            if idx is not None:
                present[idx] = leaf
        if len(present) < 10:
            return None
        pdm = tree.phylogenetic_distance_matrix()
        # MRCA depth = total height - half of pairwise distance (under ultrametric)
        # since OTL trees aren't ultrametric, fall back to shared edge length
        for i, leaf_i in present.items():
            V[i, i] = leaf_to_root[leaf_i]
        for i, leaf_i in present.items():
            for j, leaf_j in present.items():
                if i >= j:
                    continue
                mrca = pdm.mrca(leaf_i.taxon, leaf_j.taxon)
                if mrca is None:
                    continue
                # Shared branch length = depth from root to MRCA
                d = 0.0
                node = mrca
                while node.parent_node is not None:
                    d += (node.edge_length or 1.0)
                    node = node.parent_node
                V[i, j] = V[j, i] = d
        # Normalise so max diagonal = 1
        max_d = V.diagonal().max()
        if max_d > 0:
            V = V / max_d
        # Ensure positive definite
        np.fill_diagonal(V, V.diagonal() + 0.01)
        # Fill species missing from tree with identity (independent)
        for i in range(n):
            if i not in present:
                V[i, :] = 0
                V[:, i] = 0
                V[i, i] = 1.0
        return V
    except Exception as e:
        log(f"VCV build failed: {e}")
        return None


def run_models(df, V, label):
    """Run nested model comparison on a given correlation matrix."""
    y = df["log_lifetime_energy"].values
    results = {}
    spec = {
        "M0": [],
        "M1": ["log_body_kg"],
        "M2": ["log_body_kg", "log_sleep_hrs"],
        "M3": ["log_body_kg", "log_activity_hrs"],
    }
    for name, preds in spec.items():
        X, cols = build_design(df, preds)
        try:
            r = gls_fit(X, y, V)
            r["names"] = cols
            results[name] = r
        except Exception as e:
            log(f"{label} {name} fit failed: {e}")
            results[name] = None
    return results


def report(label, results, fh):
    fh.write(f"\n========================================\n")
    fh.write(f"  APPROACH: {label}\n")
    fh.write(f"========================================\n")
    for name, r in results.items():
        if r is None:
            fh.write(f"\n[{name}] FAILED\n")
            continue
        fh.write(f"\n[{name}] n={r['n']}  k={r['k']}  loglik={r['loglik']:.2f}  AIC={r['aic']:.2f}  BIC={r['bic']:.2f}\n")
        for nm, b, se, t, p in zip(r["names"], r["beta"], r["se"], r["t"], r["p"]):
            star = ("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "")
            fh.write(f"   {nm:<25}  β={b:+.4f}  se={se:.4f}  t={t:+.2f}  p={p:.4f} {star}\n")
    # AIC comparison
    fh.write("\nNested-model AIC comparison:\n")
    keys = [k for k in ["M0", "M1", "M2", "M3"] if results.get(k) is not None]
    for k in keys:
        fh.write(f"  AIC({k}) = {results[k]['aic']:+.2f}\n")
    if "M2" in keys and "M1" in keys:
        d = results["M2"]["aic"] - results["M1"]["aic"]
        fh.write(f"  ΔAIC(M2−M1) = {d:+.2f}  <- sleep effect on top of mass+class\n")
    if "M3" in keys and "M1" in keys:
        d = results["M3"]["aic"] - results["M1"]["aic"]
        fh.write(f"  ΔAIC(M3−M1) = {d:+.2f}  <- activity effect (complementary)\n")
    if "M2" in keys:
        # Test the GRADUATE decision rule
        idx = results["M2"]["names"].index("log_sleep_hrs") if "log_sleep_hrs" in results["M2"]["names"] else None
        if idx is not None:
            p_sleep = results["M2"]["p"][idx]
            daic = results["M2"]["aic"] - results["M1"]["aic"]
            verdict = (
                "GRADUATE" if (p_sleep < 0.05 and daic <= -4) else
                "KILL"     if (p_sleep >= 0.05 and daic > -2) else
                "AMBIGUOUS"
            )
            fh.write(f"\n  DECISION RULE: p_sleep={p_sleep:.4f}, ΔAIC={daic:+.2f}  →  {verdict}\n")


def main():
    df = pd.read_csv(MERGED)
    log(f"loaded {len(df)} species")
    if len(df) < 25:
        log("N < 25: insufficient sample — KILL by decision rule")
        with open(RESULTS, "w") as fh:
            fh.write(f"KILL: insufficient sample size N={len(df)}\n")
        sys.exit(0)

    # binomials for OTL
    binomials = [f"{row.Genus} {row.Species}" for _, row in df.iterrows()]

    # Build correlation matrices
    matrices = {}
    matrices["OLS (identity)"] = np.eye(len(df))
    matrices["Taxonomy (Order/Class)"] = taxonomic_correlation_matrix(df)

    newick = opentree_subtree(binomials)
    if newick:
        V_otl = vcv_from_newick(newick, binomials)
        if V_otl is not None and np.linalg.cond(V_otl) < 1e10:
            matrices["OTL phylogeny"] = V_otl
        else:
            log("OTL VCV ill-conditioned; skipping")

    # Save tree if obtained
    if newick:
        (DATA / "tree.nwk").write_text(newick)
        log(f"saved tree: {DATA / 'tree.nwk'}")

    with open(RESULTS, "w") as fh:
        fh.write(f"PGLS analysis — generated {datetime.now()}\n")
        fh.write(f"N = {len(df)}  ({sum(df['Class']=='Mammalia')} mammals + {sum(df['Class']=='Aves')} birds)\n")
        fh.write(f"Predictand: log(lifetime metabolic budget [J]) = log(MR × seconds_per_year × max_lifespan)\n")
        fh.write(f"Decision rule: GRADUATE if p_sleep<0.05 AND ΔAIC(M2−M1)≤−4\n")
        for label, V in matrices.items():
            results = run_models(df, V, label)
            report(label, results, fh)
        fh.write("\n[end of report]\n")

    log(f"wrote {RESULTS}")
    print()
    print(open(RESULTS).read())


if __name__ == "__main__":
    main()
