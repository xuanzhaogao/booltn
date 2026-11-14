/* XOR implications of a partial assignment (host code, no GPU): booltn.reduce.XorSystem.implied in C.
 *
 * The XOR system is kept in reduced row echelon form over nx compressed columns (the XOR variables), rows as
 * W-word bitsets (W = nx / 64 + 1): every pivot column occurs in its own row only. The root system (nrows0
 * rows: M0, rhs0, piv0 = pivot column of each row, prow0 = row of each pivot column or -1) is copied into the
 * work arrays (capacity nx rows: each added row has its own pivot), then each assigned column c = b is added as
 * the unit equation x_c = b:
 *   reduce e_c by the row with pivot c (if any): what is left has non-pivot columns only;
 *   nothing left: consistent iff the right-hand side is 0;
 *   else its lowest column p becomes a pivot: XOR it into every row containing p and append it.
 * The implied literals are the single-column rows (a column is fixed by the system iff its unit vector is a row
 * of the RREF) whose column is not assigned. Same result as the Python version (pivot = lowest set bit).
 *
 * Returns 1 and the implied literals out[0..*nout) as +-(column + 1), or 0 if the system is inconsistent.
 * *nrows_out: rank of the system with the units (XorSystem.rank).
 */
#include <stdint.h>
#include <string.h>

int btx_implied(int nx, int nrows0, const uint64_t *M0, const uint8_t *rhs0, const int32_t *piv0,
                const int32_t *prow0, uint64_t *M, uint8_t *rhs, int32_t *piv, int32_t *prow, uint8_t *mark,
                uint64_t *tmp, int k, const int32_t *cols, const uint8_t *vals, int32_t *out, int32_t *nout,
                int32_t *nrows_out)
{
    const int W = nx / 64 + 1;
    int nr = nrows0;
    memcpy(M, M0, (size_t)nrows0 * W * sizeof(uint64_t));
    memcpy(rhs, rhs0, (size_t)nrows0);
    memcpy(piv, piv0, (size_t)nrows0 * sizeof(int32_t));
    memcpy(prow, prow0, (size_t)nx * sizeof(int32_t));
    memset(mark, 0, (size_t)nx);
    *nout = 0;
    for (int j = 0; j < k; j++) {
        const int c = cols[j];
        mark[c] = 1;
        memset(tmp, 0, (size_t)W * sizeof(uint64_t));
        tmp[c >> 6] = 1ULL << (c & 63);
        uint8_t r = vals[j];
        const int i0 = prow[c];
        if (i0 >= 0) {
            const uint64_t *row = M + (size_t)i0 * W;
            for (int w = 0; w < W; w++) tmp[w] ^= row[w];
            r ^= rhs[i0];
        }
        int p = -1;
        for (int w = 0; w < W; w++)
            if (tmp[w]) { p = w * 64 + __builtin_ctzll(tmp[w]); break; }
        if (p < 0) {
            if (r) { *nrows_out = nr; return 0; }
            continue;
        }
        const int pw = p >> 6;
        const uint64_t pb = 1ULL << (p & 63);
        for (int i = 0; i < nr; i++) {
            uint64_t *row = M + (size_t)i * W;
            if (row[pw] & pb) {
                for (int w = 0; w < W; w++) row[w] ^= tmp[w];
                rhs[i] ^= r;
            }
        }
        memcpy(M + (size_t)nr * W, tmp, (size_t)W * sizeof(uint64_t));
        rhs[nr] = r;
        piv[nr] = p;
        prow[p] = nr;
        nr++;
    }
    *nrows_out = nr;
    for (int i = 0; i < nr; i++) {
        const int p = piv[i];
        if (mark[p]) continue;
        const uint64_t *row = M + (size_t)i * W;
        int single = 1;
        for (int w = 0; w < W && single; w++)
            if (row[w] != (w == (p >> 6) ? 1ULL << (p & 63) : 0)) single = 0;
        if (single) out[(*nout)++] = rhs[i] ? p + 1 : -(p + 1);
    }
    return 1;
}
