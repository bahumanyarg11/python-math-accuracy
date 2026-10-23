/* acc.c -- ULP-error engine for the PYACC study.

   ATTRIBUTION AND LICENCE: the functions ulp_error_fmt, ulp_error_double_fmt
   and distance64 below are ports of ulp_error() and
   ulp_error_double() (binary32_exhaustive/check_exhaustive.c) and distance()
   (binary64/check_sample2.c) from the math_accuracy repository of Paul
   Zimmermann and co-authors, https://gitlab.inria.fr/zimmerma/math_accuracy
   (commit c2b897fb), which is distributed under the terms of the GNU General
   Public License.  This file is therefore distributed under the GNU GPL as
   well; see README.md.

   Error measure: exactly that of Gladman, Innocente, Mather, Ozaki and
   Zimmermann, "Accuracy of Mathematical Functions ..." (math_accuracy repo,
   commit c2b897fb393ed8df33452b225953ccd2003cd389):

   * binary32 / binary16 / bfloat16 ("small formats"): port of ulp_error()
     and ulp_error_double() from binary32_exhaustive/check_exhaustive.c,
     with the format parameters (precision p, smallest subnormal exponent
     emin_sub, overflow exponent emax) made explicit.  For binary32
     (p=24, emin_sub=-149, emax=128) the code is the same as theirs.
   * binary64: port of distance() from binary64/check_sample2.c (USE_DOUBLE,
     rounding to nearest), as in ~/research/ARITH/harness/wc.c.

   Fast path (small formats only): when CORE-MATH provides a correctly
   rounded binary64 function g, the correctly rounded small-format value z is
   obtained by rounding g(x) to the format (exact unless g(x) is a midpoint
   of the small format, in which case MPFR is used), and the error of a
   mismatch is first estimated from g(x); this estimate differs from the
   exact one by < 2^-28 ulp.  All final maxima are recomputed exactly with
   the MPFR port (acc_err_exact).  Without g, MPFR is used for everything.

   Library outputs are given by the caller (Python), as float32 arrays for
   the small formats (binary16/bfloat16 values are exact in float32) and
   float64 arrays for binary64.
*/
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <fenv.h>
#include <pthread.h>
#include <mpfr.h>

typedef double (*g1_t) (double);
typedef double (*g2_t) (double, double);
typedef int (*m1_t) (mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
typedef int (*m2_t) (mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t);

#ifndef NO_CM
#define CM(f) extern double cr_##f (double);
#define CM2(f) extern double cr_##f (double, double);
CM(exp) CM(exp2) CM(exp10) CM(expm1) CM(log) CM(log2) CM(log10) CM(log1p)
CM(sin) CM(cos) CM(tan) CM(asin) CM(acos) CM(atan) CM(sinh) CM(cosh) CM(tanh)
CM(asinh) CM(acosh) CM(atanh) CM(cbrt) CM(erf) CM(erfc) CM(lgamma) CM(tgamma)
CM2(pow) CM2(atan2) CM2(hypot)
#define G(f) cr_##f
#else
#define G(f) NULL
#endif

static int m_lgamma (mpfr_ptr r, mpfr_srcptr x, mpfr_rnd_t rnd)
{ int s; return mpfr_lgamma (r, &s, x, rnd); }
/* rsqrt with the IEEE 754-2019 rSqrt semantics: rSqrt(-0) = -Inf (mpfr_rec_sqrt gives +Inf) */
static int m_rsqrt (mpfr_ptr r, mpfr_srcptr x, mpfr_rnd_t rnd)
{
  if (mpfr_zero_p (x) && mpfr_signbit (x)) { mpfr_set_inf (r, -1); return 0; }
  return mpfr_rec_sqrt (r, x, rnd);
}

static struct { const char *name; m1_t m; m2_t m2; g1_t g; g2_t g2; } T[] = {
  {"exp", mpfr_exp, 0, G(exp)}, {"exp2", mpfr_exp2, 0, G(exp2)},
  {"exp10", mpfr_exp10, 0, G(exp10)}, {"expm1", mpfr_expm1, 0, G(expm1)},
  {"log", mpfr_log, 0, G(log)}, {"log2", mpfr_log2, 0, G(log2)},
  {"log10", mpfr_log10, 0, G(log10)}, {"log1p", mpfr_log1p, 0, G(log1p)},
  {"sin", mpfr_sin, 0, G(sin)}, {"cos", mpfr_cos, 0, G(cos)},
  {"tan", mpfr_tan, 0, G(tan)}, {"asin", mpfr_asin, 0, G(asin)},
  {"acos", mpfr_acos, 0, G(acos)}, {"atan", mpfr_atan, 0, G(atan)},
  {"sinh", mpfr_sinh, 0, G(sinh)}, {"cosh", mpfr_cosh, 0, G(cosh)},
  {"tanh", mpfr_tanh, 0, G(tanh)}, {"asinh", mpfr_asinh, 0, G(asinh)},
  {"acosh", mpfr_acosh, 0, G(acosh)}, {"atanh", mpfr_atanh, 0, G(atanh)},
  {"cbrt", mpfr_cbrt, 0, G(cbrt)}, {"erf", mpfr_erf, 0, G(erf)},
  {"erfc", mpfr_erfc, 0, G(erfc)}, {"lgamma", m_lgamma, 0, G(lgamma)},
  {"tgamma", mpfr_gamma, 0, G(tgamma)},
  {"j0", mpfr_j0, 0, 0}, {"j1", mpfr_j1, 0, 0},
  {"y0", mpfr_y0, 0, 0}, {"y1", mpfr_y1, 0, 0},
  {"sqrt", mpfr_sqrt, 0, sqrt}, {"rsqrt", m_rsqrt, 0, 0},
  {"pow", 0, mpfr_pow, 0, G(pow)}, {"atan2", 0, mpfr_atan2, 0, G(atan2)},
  {"hypot", 0, mpfr_hypot, 0, G(hypot)},
};
#define NT (int) (sizeof T / sizeof T[0])

int acc_func_id (const char *name)
{
  for (int i = 0; i < NT; i++)
    if (strcmp (T[i].name, name) == 0) return i;
  return -1;
}
int acc_is_bivariate (int f) { return T[f].m2 != 0; }
int acc_has_fast (int f) { return T[f].g != 0 || T[f].g2 != 0; }

/* ---------- formats ---------- */
typedef struct { int p, emin_sub, emax; } fmt_t;
static const fmt_t FMTS[] = { {24, -149, 128}, {11, -24, 16}, {8, -133, 128} };
/* 0 = binary32, 1 = binary16, 2 = bfloat16 */

static int mcall (int f, mpfr_ptr r, mpfr_srcptr a, mpfr_srcptr b, mpfr_rnd_t rnd)
{ return T[f].m2 ? T[f].m2 (r, a, b, rnd) : T[f].m (r, a, rnd); }

/* correctly rounded value in the format, computed directly with MPFR
   (cr_foo of check_exhaustive.c with the format's exponent range) */
static double cr_mpfr (int f, const fmt_t *F, double x1, double x2)
{
  mpfr_exp_t emin = mpfr_get_emin (), emax = mpfr_get_emax ();
  mpfr_t a, b, y;
  mpfr_set_emin (mpfr_get_emin_min ()); mpfr_set_emax (mpfr_get_emax_max ());
  mpfr_init2 (a, 53); mpfr_init2 (b, 53); mpfr_init2 (y, F->p);
  mpfr_set_d (a, x1, MPFR_RNDN); mpfr_set_d (b, x2, MPFR_RNDN);
  mpfr_set_emin (F->emin_sub + 1); mpfr_set_emax (F->emax);
  int inex = mcall (f, y, a, b, MPFR_RNDN);
  inex = mpfr_check_range (y, inex, MPFR_RNDN);
  mpfr_subnormalize (y, inex, MPFR_RNDN);
  double r = mpfr_get_d (y, MPFR_RNDN);
  mpfr_clear (a); mpfr_clear (b); mpfr_clear (y);
  mpfr_set_emin (emin); mpfr_set_emax (emax);
  return r;
}

/* round double g to the format (round to nearest even); *mid set if g is
   exactly a midpoint (double rounding possible) */
static double round_fmt (double g, const fmt_t *F, int *mid)
{
  *mid = 0;
  if (isnan (g) || isinf (g) || g == 0) return g;
  double a = fabs (g);
  int e = ilogb (a);
  int se = e - (F->p - 1);
  if (se < F->emin_sub) se = F->emin_sub;
  double q = ldexp (a, -se);
  double fl = floor (q);
  if (q - fl == 0.5) *mid = 1;
  double v = ldexp (nearbyint (q), se);
  if (v >= ldexp (1.0, F->emax)) v = INFINITY;
  return copysign (v, g);
}

/* next value of the format after z in direction of y (z finite) */
static double next_fmt (double z, double y, const fmt_t *F)
{
  if (F->p == 24) return nextafterf ((float) z, (float) y);
  if (z == y) return y;
  double tiny = ldexp (1.0, F->emin_sub);
  if (z == 0) return copysign (tiny, y);
  double a = fabs (z);
  int away = (y > z) == (z > 0);
  int e = ilogb (a);
  int se = e - (F->p - 1); if (se < F->emin_sub) se = F->emin_sub;
  double sp = ldexp (1.0, se);
  if (!away && a == ldexp (1.0, e) && se > F->emin_sub) sp /= 2;
  double r = away ? a + sp : a - sp;
  if (r >= ldexp (1.0, F->emax)) r = INFINITY;
  return copysign (r, z);
}

/* port of ulp_error() (integer part); y = library, z = correct rounding */
static uint64_t ulp_error_fmt (double y, double z, const fmt_t *F)
{
  if (y == z) return 0;
  double half = ldexp (1.0, F->emax - 1);
  if (isinf (z))
    {
      if (isinf (y)) return UINT64_MAX;
      y = y / 2;
      z = (z > 0) ? half : -half;
    }
  else if (isinf (y))
    {
      z = z / 2;
      y = (y > 0) ? half : -half;
    }
  if (F->p == 24)
    {
      float yf = y, zf = z;
      float err = yf - zf;
      float ulp = nextafterf (zf, yf) - zf;
      err = fabsf (err / ulp);
      return (err >= (float) UINT64_MAX) ? UINT64_MAX : (uint64_t) err;
    }
  double err = fabs ((y - z) / (next_fmt (z, y, F) - z));
  return (err >= 1.8e19) ? UINT64_MAX : (uint64_t) err;
}

/* port of ulp_error_double(): |f(x) - y| / ulp, f(x) with MPFR at 100 bits,
   ulp = 2^max(e-p, emin_sub) where e = exponent of f(x); y=Inf -> 2^emax */
static double ulp_error_double_fmt (int f, const fmt_t *F, double y,
                                    double x1, double x2)
{
  mpfr_t yy, zz, zz1, a, b;
  int ret;
  mpfr_exp_t e;
  double err;
  mpfr_exp_t emin = mpfr_get_emin (), emax = mpfr_get_emax ();
  mpfr_set_emin (mpfr_get_emin_min ()); mpfr_set_emax (mpfr_get_emax_max ());
  mpfr_init2 (yy, 53); mpfr_init2 (zz, 100); mpfr_init2 (zz1, 100);
  mpfr_init2 (a, 53); mpfr_init2 (b, 53);
  if (!isinf (y)) mpfr_set_d (yy, y, MPFR_RNDN);
  else
    {
      mpfr_set_ui_2exp (yy, 1, F->emax, MPFR_RNDN);
      /* check_exhaustive.c uses +2^128 whatever the sign of y; we use the
         sign of y, as distance() of check_sample2.c does (documented) */
      if (y < 0) mpfr_neg (yy, yy, MPFR_RNDN);
    }
  mpfr_set_d (a, x1, MPFR_RNDN); mpfr_set_d (b, x2, MPFR_RNDN);
  ret = mcall (f, zz1, a, b, MPFR_RNDN);
  e = mpfr_get_exp (zz1);
  mpfr_sub (zz, zz1, yy, MPFR_RNDA);
  if (ret < 0 && mpfr_cmp_ui (zz, 0) > 0)
    { mpfr_nextabove (zz1); e = mpfr_get_exp (zz1); mpfr_sub (zz, zz1, yy, MPFR_RNDA); }
  if (ret > 0 && mpfr_cmp_ui (zz, 0) < 0)
    { mpfr_nextbelow (zz1); e = mpfr_get_exp (zz1); mpfr_sub (zz, zz1, yy, MPFR_RNDA); }
  mpfr_abs (zz, zz, MPFR_RNDN);
  if (mpfr_zero_p (zz1)) e = F->emin_sub;  /* exponent of zero: tiny */
  else e = (e - F->p < F->emin_sub) ? F->emin_sub : e - F->p;
  mpfr_mul_2si (zz, zz, -e, MPFR_RNDN);
  err = mpfr_get_d (zz, MPFR_RNDA);
  mpfr_clear (yy); mpfr_clear (zz); mpfr_clear (zz1); mpfr_clear (a); mpfr_clear (b);
  mpfr_set_emin (emin); mpfr_set_emax (emax);
  return err;
}

/* ---------- chunk evaluation (small formats) ---------- */
#define K 32
typedef struct {
  int64_t n, n_mis, n_err2, n_spnan, n_misnan, n_flush, n_yinf, n_badinf,
    n_mpfr;
  int k;
  uint64_t top_u[K]; double top_d[K]; float top_x1[K], top_x2[K], top_y[K];
  double fl_d; float fl_x1, fl_x2, fl_y;      /* worst flushed case */
  float sn_x1, sn_x2, sn_y, mn_x1, mn_x2, mn_y; /* example NaN cases */
} res_t;

static int better (uint64_t u1, double d1, uint64_t u2, double d2)
{ return u1 > u2 || (u1 == u2 && d1 > d2); }

static void top_insert (res_t *r, uint64_t u, double d, float x1, float x2, float y)
{
  if (r->k == K && !better (u, d, r->top_u[K - 1], r->top_d[K - 1])) return;
  int i = (r->k < K) ? r->k++ : K - 1;
  while (i > 0 && better (u, d, r->top_u[i - 1], r->top_d[i - 1]))
    {
      r->top_u[i] = r->top_u[i - 1]; r->top_d[i] = r->top_d[i - 1];
      r->top_x1[i] = r->top_x1[i - 1]; r->top_x2[i] = r->top_x2[i - 1];
      r->top_y[i] = r->top_y[i - 1]; i--;
    }
  r->top_u[i] = u; r->top_d[i] = d; r->top_x1[i] = x1; r->top_x2[i] = x2;
  r->top_y[i] = y;
}


typedef struct { int f, fmt; const float *x1, *x2, *y, *zref; long lo, hi; res_t r; } job_t;

static void process (int f, const fmt_t *F, float x1, float x2, float y,
                     const float *zref, res_t *r)
{
  double g = NAN, z;
  int mid = 1, have_g = 0;
  r->n++;
  if (zref) { z = *zref; mid = 0; }
  else if (T[f].g2) { g = T[f].g2 (x1, x2); have_g = 1; }
  else if (T[f].g) { g = T[f].g (x1); have_g = 1; }
  if (have_g) z = round_fmt (g, F, &mid);
  if (mid) { z = cr_mpfr (f, F, x1, x2); r->n_mpfr++; }
  /* note: +0 vs -0 is not counted as an error (y == z in C, as theirs) */
  if (y == z || (isnan (y) && isnan (z))) return;
  r->n_mis++;
  /* flush-to-zero: zero returned for a subnormal correctly rounded result,
     or a subnormal input treated as zero (result equal to f(+-0), NaN-ness
     included).  Classified before the NaN categories. */
  double minnorm = ldexp (1.0, F->emin_sub + F->p - 1);
  int sub1 = (x1 != 0 && fabs (x1) < minnorm);
  int sub2 = (T[f].m2 && x2 != 0 && fabs (x2) < minnorm);
  int flush = (y == 0 && z != 0 && fabs (z) < minnorm);
  if (!flush && (sub1 || sub2))
    {
      double z0 = cr_mpfr (f, F, sub1 ? copysign (0.0, x1) : x1,
                           sub2 ? copysign (0.0, x2) : x2);
      r->n_mpfr++;
      flush = (y == z0) || (isnan (y) && isnan (z0));
    }
  if (flush && (isnan (y) || isnan (z)))
    {
      if (r->n_flush++ == 0 || INFINITY > r->fl_d)
        { r->fl_d = INFINITY; r->fl_x1 = x1; r->fl_x2 = x2; r->fl_y = y; }
      return;
    }
  if (isnan (y))
    { if (r->n_spnan++ == 0) { r->sn_x1 = x1; r->sn_x2 = x2; r->sn_y = y; } return; }
  if (isnan (z))
    { if (r->n_misnan++ == 0) { r->mn_x1 = x1; r->mn_x2 = x2; r->mn_y = y; } return; }
  uint64_t u = ulp_error_fmt (y, z, F);
  if (u == UINT64_MAX && isinf (y) && isinf (z)) { r->n_badinf++; }
  if (isinf (y)) r->n_yinf++;
  double d;
  int exact = !have_g || isinf (g) || g == 0;
  if (!exact)
    {
      int e; double m = frexp (g, &e);
      if (fabs (m) == 0.5) exact = 1;
      else
        {
          int ue = e - F->p; if (ue < F->emin_sub) ue = F->emin_sub;
          double Y = isinf (y) ? copysign (ldexp (1.0, F->emax), y) : y;
          d = ldexp (fabs (g - Y), -ue);
        }
    }
  if (exact && !flush && r->k == K && !isinf (z) && z != 0)
    {
      /* prune: upper bound of the exact error from z = RN(f(x)):
         |f(x) - y| <= |z - y| + ulp(z)/2, and ulp(f(x)) = ulp(z) unless z is
         a power of 2 (then no pruning).  Skip MPFR if the case cannot enter
         the top-K list. */
      int e; double m = frexp (z, &e);
      if (fabs (m) != 0.5)
        {
          int ue = e - F->p; if (ue < F->emin_sub) ue = F->emin_sub;
          double Y = isinf (y) ? copysign (ldexp (1.0, F->emax), y) : y;
          double ub = ldexp (fabs (z - Y), -ue) + 0.5 + 1e-9;
          if (!better (u, ub, r->top_u[K - 1], r->top_d[K - 1]))
            { if (u > 1) r->n_err2++; return; }
        }
    }
  if (exact) { d = ulp_error_double_fmt (f, F, y, x1, x2); r->n_mpfr++; }
  if (flush)
    {
      if (r->n_flush++ == 0 || d > r->fl_d)
        { r->fl_d = d; r->fl_x1 = x1; r->fl_x2 = x2; r->fl_y = y; }
      return;
    }
  if (u > 1) r->n_err2++;
  top_insert (r, u, d, x1, x2, y);
}

static void *worker (void *p)
{
  job_t *j = p;
  fesetround (FE_TONEAREST);
  const fmt_t *F = &FMTS[j->fmt];
  for (long i = j->lo; i < j->hi; i++)
    process (j->f, F, j->x1[i], j->x2 ? j->x2[i] : 0.0f, j->y[i],
             j->zref ? j->zref + i : NULL, &j->r);
  mpfr_free_cache ();
  return NULL;
}

static void merge (res_t *a, const res_t *b)
{
  a->n += b->n; a->n_mis += b->n_mis; a->n_err2 += b->n_err2;
  if (b->n_spnan && !a->n_spnan) { a->sn_x1 = b->sn_x1; a->sn_x2 = b->sn_x2; a->sn_y = b->sn_y; }
  if (b->n_misnan && !a->n_misnan) { a->mn_x1 = b->mn_x1; a->mn_x2 = b->mn_x2; a->mn_y = b->mn_y; }
  if (b->n_flush && (!a->n_flush || b->fl_d > a->fl_d))
    { a->fl_d = b->fl_d; a->fl_x1 = b->fl_x1; a->fl_x2 = b->fl_x2; a->fl_y = b->fl_y; }
  a->n_spnan += b->n_spnan; a->n_misnan += b->n_misnan; a->n_flush += b->n_flush;
  a->n_yinf += b->n_yinf; a->n_badinf += b->n_badinf; a->n_mpfr += b->n_mpfr;
  for (int i = 0; i < b->k; i++)
    top_insert (a, b->top_u[i], b->top_d[i], b->top_x1[i], b->top_x2[i], b->top_y[i]);
}

int acc_res_size (void) { return sizeof (res_t); }

/* evaluate a chunk; x2 may be NULL for univariate functions; result merged
   into *out (which the caller zero-initialises once) */
void acc_chunk (int f, int fmt, const float *x1, const float *x2, const float *y,
                const float *zref, long n, int nthreads, res_t *out)
{
  if (nthreads < 1) nthreads = 1;
  if (nthreads > 128) nthreads = 128;
  if (n < 1024) nthreads = 1;
  pthread_t th[128];
  job_t *jb = calloc (nthreads, sizeof (job_t));
  for (int t = 0; t < nthreads; t++)
    {
      jb[t].f = f; jb[t].fmt = fmt; jb[t].x1 = x1; jb[t].x2 = x2; jb[t].y = y; jb[t].zref = zref;
      jb[t].lo = n * t / nthreads; jb[t].hi = n * (t + 1) / nthreads;
      if (nthreads == 1) worker (&jb[t]);
      else pthread_create (&th[t], NULL, worker, &jb[t]);
    }
  for (int t = 0; t < nthreads; t++)
    {
      if (nthreads > 1) pthread_join (th[t], NULL);
      merge (out, &jb[t].r);
    }
  free (jb);
}

/* exact error of one case (MPFR only), small formats: returns the
   ulp_error_double value, and the integer ulp_error in *u, the correctly
   rounded value in *zout */
double acc_err_exact (int f, int fmt, double x1, double x2, double y,
                      uint64_t *u, double *zout)
{
  const fmt_t *F = &FMTS[fmt];
  double z = cr_mpfr (f, F, x1, x2);
  *zout = z;
  *u = ulp_error_fmt (y, z, F);
  if (isnan (y) || isnan (z)) return (isnan (y) && isnan (z)) ? 0 : NAN;
  return ulp_error_double_fmt (f, F, y, x1, x2);
}

/* correctly rounded reference for an array (MPFR only) */
void acc_cr_array (int f, int fmt, const double *x1, const double *x2,
                   double *z, long n)
{
  for (long i = 0; i < n; i++)
    z[i] = cr_mpfr (f, &FMTS[fmt], x1[i], x2 ? x2[i] : 0.0);
}

/* fast reference for an array: g(x) rounded to format, MPFR on midpoints;
   returns -1 if no fast function */
int acc_cr_fast_array (int f, int fmt, const double *x1, const double *x2,
                       double *z, long n)
{
  if (!acc_has_fast (f)) return -1;
  int mid;
  for (long i = 0; i < n; i++)
    {
      double g = T[f].g2 ? T[f].g2 (x1[i], x2[i]) : T[f].g (x1[i]);
      z[i] = round_fmt (g, &FMTS[fmt], &mid);
      if (mid) z[i] = cr_mpfr (f, &FMTS[fmt], x1[i], x2 ? x2[i] : 0.0);
    }
  return 0;
}

/* ---------- binary64: port of distance() of check_sample2.c ---------- */
#define EMAX64 1024
#define EMIN64 -1074
#define PREC64 53

static double finish64 (double z, mpfr_t zz, mpfr_t yy, int f, mpfr_t xx, mpfr_t xx2)
{
  int underflow = 0;
  mpfr_exp_t expz = 0;
  if (isnan (z) && mpfr_nan_p (zz)) return 0;
  if (isnan (z) || mpfr_nan_p (zz)) return 0x1p1023 + 0x1p1023;
  mpfr_set_emax (mpfr_get_emax_max ());
  if (isinf (z) && mpfr_inf_p (zz) == 0)
    {
      mpfr_set_ui_2exp (yy, 1, EMAX64, MPFR_RNDN);
      if (z < 0) mpfr_neg (yy, yy, MPFR_RNDN);
    }
  else if (!isinf (z) && mpfr_inf_p (zz))
    {
      mcall (f, zz, xx, xx2, MPFR_RNDN);  /* recompute with unbounded exponent */
      expz = mpfr_get_exp (zz);
      underflow = expz <= EMIN64;
      mpfr_set_d (yy, z, MPFR_RNDN);
    }
  else
    mpfr_set_d (yy, z, MPFR_RNDN);
  if (mpfr_number_p (zz))
    {
      if (mpfr_zero_p (zz)) underflow = 1;
      else { expz = mpfr_get_exp (zz); underflow = expz <= EMIN64; }
    }
  mpfr_sub (zz, zz, yy, MPFR_RNDN);
  mpfr_abs (zz, zz, MPFR_RNDN);
  if (mpfr_number_p (zz) && !underflow && expz - PREC64 >= EMIN64)
    mpfr_mul_2si (zz, zz, PREC64 - expz, MPFR_RNDN);
  else
    mpfr_mul_2si (zz, zz, -EMIN64, MPFR_RNDN);
  return mpfr_get_d (zz, MPFR_RNDU);
}

static double distance64 (int f, double x1, double x2, double z)
{
  mpfr_t xx, xx2, yy, zz;
  double ret;
  if (isnan (x1) || (T[f].m2 && isnan (x2))) return 0;
  mpfr_exp_t emin = mpfr_get_emin (), emax = mpfr_get_emax ();
  mpfr_set_emin (mpfr_get_emin_min ());
  mpfr_set_emax (EMAX64);
  mpfr_init2 (xx, PREC64); mpfr_init2 (xx2, PREC64);
  mpfr_init2 (yy, PREC64 + 1); mpfr_init2 (zz, 3 * PREC64);
  mpfr_set_d (xx, x1, MPFR_RNDN); mpfr_set_d (xx2, x2, MPFR_RNDN);
  mcall (f, zz, xx, xx2, MPFR_RNDN);
  ret = finish64 (z, zz, yy, f, xx, xx2);
  mpfr_clear (xx); mpfr_clear (xx2); mpfr_clear (yy); mpfr_clear (zz);
  mpfr_set_emin (emin); mpfr_set_emax (emax);
  return ret;
}

typedef struct { int f; const double *x1, *x2, *z; double *err; long lo, hi; } job64_t;
static void *worker64 (void *p)
{
  job64_t *j = p;
  fesetround (FE_TONEAREST);
  for (long i = j->lo; i < j->hi; i++)
    j->err[i] = distance64 (j->f, j->x1[i], j->x2 ? j->x2[i] : 0.0, j->z[i]);
  mpfr_free_cache ();
  return NULL;
}

void acc_err64 (int f, const double *x1, const double *x2, const double *z,
                double *err, long n, int nthreads)
{
  if (nthreads < 1) nthreads = 1;
  if (nthreads > 128) nthreads = 128;
  if (n < 256) nthreads = 1;
  pthread_t th[128];
  job64_t jb[128];
  for (int t = 0; t < nthreads; t++)
    {
      jb[t] = (job64_t) { f, x1, x2, z, err, n * t / nthreads, n * (t + 1) / nthreads };
      if (nthreads == 1) worker64 (&jb[t]);
      else pthread_create (&th[t], NULL, worker64, &jb[t]);
    }
  if (nthreads > 1)
    for (int t = 0; t < nthreads; t++) pthread_join (th[t], NULL);
}

/* MPFR value of f at high precision (for independent checks), as string */
void acc_mpfr_str (int f, double x1, double x2, int prec, char *buf, int len)
{
  mpfr_exp_t emin = mpfr_get_emin (), emax = mpfr_get_emax ();
  mpfr_set_emin (mpfr_get_emin_min ()); mpfr_set_emax (mpfr_get_emax_max ());
  mpfr_t a, b, y;
  mpfr_init2 (a, 53); mpfr_init2 (b, 53); mpfr_init2 (y, prec);
  mpfr_set_d (a, x1, MPFR_RNDN); mpfr_set_d (b, x2, MPFR_RNDN);
  mcall (f, y, a, b, MPFR_RNDN);
  mpfr_snprintf (buf, len, "%.60Re", y);
  mpfr_clear (a); mpfr_clear (b); mpfr_clear (y);
  mpfr_set_emin (emin); mpfr_set_emax (emax);
}

/* ---------- threaded helpers ---------- */
typedef struct { int f, fmt, mode; const float *x1, *x2, *y; float *z; double *err; long lo, hi; } jobz_t;
static void *workerz (void *p)
{
  jobz_t *j = p;
  fesetround (FE_TONEAREST);
  const fmt_t *F = &FMTS[j->fmt];
  for (long i = j->lo; i < j->hi; i++)
    {
      double x2 = j->x2 ? j->x2[i] : 0.0;
      if (j->mode == 0)        /* correctly rounded reference, MPFR */
        j->z[i] = cr_mpfr (j->f, F, j->x1[i], x2);
      else                     /* per-point exact error (ulp_error_double) */
        {
          double z = cr_mpfr (j->f, F, j->x1[i], x2), y = j->y[i];
          if (y == z || (isnan (y) && isnan (z))) j->err[i] = 0;
          else if (isnan (y) || isnan (z)) j->err[i] = INFINITY;
          else j->err[i] = ulp_error_double_fmt (j->f, F, y, j->x1[i], x2);
        }
    }
  mpfr_free_cache ();
  return NULL;
}
static void runz (jobz_t proto, long n, int nthreads)
{
  if (nthreads < 1) nthreads = 1;
  if (nthreads > 128) nthreads = 128;
  if (n < 256) nthreads = 1;
  pthread_t th[128]; jobz_t jb[128];
  for (int t = 0; t < nthreads; t++)
    {
      jb[t] = proto; jb[t].lo = n * t / nthreads; jb[t].hi = n * (t + 1) / nthreads;
      if (nthreads == 1) workerz (&jb[t]); else pthread_create (&th[t], NULL, workerz, &jb[t]);
    }
  if (nthreads > 1) for (int t = 0; t < nthreads; t++) pthread_join (th[t], NULL);
}
/* CR reference of a float array (MPFR, threaded) */
void acc_cr_mt (int f, int fmt, const float *x1, const float *x2, float *z, long n, int nthreads)
{ jobz_t p = { f, fmt, 0, x1, x2, NULL, z, NULL, 0, 0 }; runz (p, n, nthreads); }
/* exact per-point errors (MPFR, threaded); NaN mismatch -> Inf */
void acc_err_mt (int f, int fmt, const float *x1, const float *x2, const float *y, double *err, long n, int nthreads)
{ jobz_t p = { f, fmt, 1, x1, x2, y, NULL, err, 0, 0 }; runz (p, n, nthreads); }
