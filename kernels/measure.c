/* (tc, sc) of a contraction tree restricted to the residual of a partial assignment (host code, no GPU).
 *
 * Used by the branching lookahead (booltn/measure.py): the same numbers as
 *   recompute_outputs(prune_tree(tree, keep), ixs, []) and complexity(d, ixs)
 * in booltn/tree_utils.py, with label sets as uint64 bitsets instead of Python dicts.
 *
 * Tree: nn nodes in post-order (left subtree, right subtree, node). leaf_clause[j] >= 0: leaf of clause
 * leaf_clause[j]; else internal with children left[j], right[j]. Clauses in CSR form (cptr, clit: DIMACS
 * literals over variables 1..n). val[v] (v = 1..n): 0 unassigned, 1 true, -1 false.
 * A clause with a true literal is dropped (its leaf is dead); an internal node with one live child is replaced
 * by it (prune_tree). Output labels of a node: labels of its subtree that also occur outside it.
 * Work buffers: sub, ext (nn * W words each, W = n / 64 + 1), eff (nn ints), terms (nn doubles).
 * Returns the number of contractions; *tc = log2 sum 2^(B+M+N+K) summed in post-order (as log2sumexp2),
 * *sc = max(output size of every contraction, size of every live leaf).
 */
#include <math.h>
#include <stdint.h>
#include <string.h>

static inline int pc(uint64_t x) { return __builtin_popcountll(x); }

int btm_tcsc(int nn, const int32_t *leaf_clause, const int32_t *left, const int32_t *right,
             const int32_t *cptr, const int32_t *clit, int n, const int8_t *val,
             uint64_t *sub, uint64_t *ext, int32_t *eff, double *terms, double *tc, int32_t *sc)
{
    const int W = n / 64 + 1;
    int scmax = 0, nterm = 0;
    /* bottom-up: live flag (eff >= 0: the node that stands for this subtree after pruning) and subtree labels */
    for (int j = 0; j < nn; j++) {
        uint64_t *s = sub + (size_t)j * W;
        int c = leaf_clause[j];
        if (c >= 0) {
            memset(s, 0, W * sizeof(uint64_t));
            int sat = 0;
            for (int k = cptr[c]; k < cptr[c + 1]; k++) {
                int l = clit[k], v = l > 0 ? l : -l, b = val[v];
                if (b == 0)
                    s[v >> 6] |= 1ull << (v & 63);
                else if ((b > 0) == (l > 0)) { sat = 1; break; }
            }
            if (sat) { eff[j] = -1; memset(s, 0, W * sizeof(uint64_t)); continue; }
            eff[j] = j;
            int sz = 0;
            for (int w = 0; w < W; w++) sz += pc(s[w]);
            if (sz > scmax) scmax = sz;
        } else {
            int a = left[j], b = right[j];
            const uint64_t *sa = sub + (size_t)a * W, *sb = sub + (size_t)b * W;
            for (int w = 0; w < W; w++) s[w] = sa[w] | sb[w];
            eff[j] = (eff[a] >= 0 && eff[b] >= 0) ? j : (eff[a] >= 0 ? eff[a] : eff[b]);
        }
    }
    if (nn == 0 || eff[nn - 1] < 0) { *tc = 0.0; *sc = 0; return 0; }
    /* top-down (reverse post-order): labels that occur outside each subtree */
    memset(ext + (size_t)(nn - 1) * W, 0, W * sizeof(uint64_t));
    for (int j = nn - 1; j >= 0; j--) {
        if (leaf_clause[j] >= 0 || eff[j] < 0) continue;
        int a = left[j], b = right[j];
        const uint64_t *e = ext + (size_t)j * W, *sa = sub + (size_t)a * W, *sb = sub + (size_t)b * W;
        uint64_t *ea = ext + (size_t)a * W, *eb = ext + (size_t)b * W;
        for (int w = 0; w < W; w++) { ea[w] = e[w] | sb[w]; eb[w] = e[w] | sa[w]; }
    }
    /* contractions (both children live), post-order; operands: a leaf's full labels or a node's output */
    for (int j = 0; j < nn; j++) {
        if (leaf_clause[j] >= 0 || eff[j] != j) continue;
        int a = eff[left[j]], b = eff[right[j]];
        const uint64_t *sa = sub + (size_t)a * W, *sb = sub + (size_t)b * W;
        const uint64_t *ea = ext + (size_t)a * W, *eb = ext + (size_t)b * W, *e = ext + (size_t)j * W;
        int t = 0, y = 0;
        for (int w = 0; w < W; w++) {
            uint64_t A = leaf_clause[a] >= 0 ? sa[w] : (sa[w] & ea[w]);
            uint64_t B = leaf_clause[b] >= 0 ? sb[w] : (sb[w] & eb[w]);
            uint64_t Y = (A | B) & e[w];
            t += pc(A & B & Y) + pc(A & ~B & Y) + pc(B & ~A & Y) + pc(A & B & ~Y);
            y += pc(Y);
        }
        terms[nterm++] = t;
        if (y > scmax) scmax = y;
    }
    *sc = scmax;
    if (nterm == 0) { *tc = 0.0; return 0; }
    double m = terms[0];
    for (int i = 1; i < nterm; i++) if (terms[i] > m) m = terms[i];
    double s = 0.0;
    for (int i = 0; i < nterm; i++) s += ldexp(1.0, (int)(terms[i] - m));
    *tc = m + log2(s);
    return nterm;
}

/* Output labels of every contraction of the restricted tree, in the order of recompute_outputs: call right
 * after btm_tcsc with the same arguments (it reads sub, ext, eff). For each live internal node j (eff[j] == j)
 * the labels (0-based variables) of iy(j) = first occurrences in la + lb that occur outside the subtree, where
 * la is the clause's unassigned literals in clause order for a leaf child, else the child's iy, go to
 * arena[off[j] .. off[j] + len[j]). stamp: n + 1 ints (work). Returns the arena size used, or -1 if it would
 * exceed cap (call again with a larger arena).
 */
int btm_iy(int nn, const int32_t *leaf_clause, const int32_t *left, const int32_t *right,
           const int32_t *cptr, const int32_t *clit, int n, const int8_t *val,
           const uint64_t *sub, const uint64_t *ext, const int32_t *eff,
           int32_t *off, int32_t *len, int32_t *arena, int cap, int32_t *stamp)
{
    const int W = n / 64 + 1;
    int used = 0;
    for (int v = 0; v <= n; v++) stamp[v] = -1;
    for (int j = 0; j < nn; j++) {
        if (leaf_clause[j] >= 0 || eff[j] != j) continue;
        const uint64_t *e = ext + (size_t)j * W;
        off[j] = used;
        for (int side = 0; side < 2; side++) {
            int a = eff[side ? right[j] : left[j]];
            int c = leaf_clause[a];
            if (c >= 0) {
                for (int k = cptr[c]; k < cptr[c + 1]; k++) {
                    int l = clit[k], v = l > 0 ? l : -l;
                    if (val[v] != 0 || stamp[v] == j) continue;
                    stamp[v] = j;
                    if (e[v >> 6] >> (v & 63) & 1) {
                        if (used >= cap) return -1;
                        arena[used++] = v - 1;
                    }
                }
            } else {
                for (int k = off[a]; k < off[a] + len[a]; k++) {
                    int v = arena[k] + 1;
                    if (stamp[v] == j) continue;
                    stamp[v] = j;
                    if (e[v >> 6] >> (v & 63) & 1) {
                        if (used >= cap) return -1;
                        arena[used++] = v - 1;
                    }
                }
            }
        }
        len[j] = used - off[j];
    }
    return used;
}
