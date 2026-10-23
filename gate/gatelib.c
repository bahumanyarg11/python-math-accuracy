/* thin batch caller of a C math library (glibc or OpenLibm), for the gate */
#define _GNU_SOURCE
#include <math.h>
#include <string.h>
typedef float (*f1f) (float); typedef double (*f1d) (double);
typedef float (*f2f) (float, float); typedef double (*f2d) (double, double);
#ifdef __APPLE__ /* smoke tests only: Apple has no float Bessel */
#define j0f(x) ((float) j0 (x))
#define j1f(x) ((float) j1 (x))
#define y0f(x) ((float) y0 (x))
#define y1f(x) ((float) y1 (x))
#endif
#ifdef NO_EXP10
#define exp10f(x) NAN
#define exp10(x) NAN
#endif
#define W(n) static float w_##n##f (float x) { return n##f (x); } \
             static double w_##n (double x) { return n (x); }
W(exp) W(exp2) W(exp10) W(expm1) W(log) W(log2) W(log10) W(log1p) W(sin) W(cos)
W(tan) W(asin) W(acos) W(atan) W(sinh) W(cosh) W(tanh) W(asinh) W(acosh)
W(atanh) W(cbrt) W(erf) W(erfc) W(lgamma) W(tgamma) W(j0) W(j1) W(y0) W(y1)
#define E(n) {#n, w_##n##f, w_##n},
static struct { const char *name; f1f f; f1d d; } T[] = {
 E(exp) E(exp2) E(exp10) E(expm1) E(log) E(log2) E(log10) E(log1p) E(sin) E(cos)
 E(tan) E(asin) E(acos) E(atan) E(sinh) E(cosh) E(tanh) E(asinh) E(acosh)
 E(atanh) E(cbrt) E(erf) E(erfc) E(lgamma) E(tgamma) E(j0) E(j1) E(y0) E(y1) };
static float p2f (float x, float y) { return powf (x, y); }
static double p2 (double x, double y) { return pow (x, y); }
static float a2f (float x, float y) { return atan2f (x, y); }
static double a2 (double x, double y) { return atan2 (x, y); }
static float h2f (float x, float y) { return hypotf (x, y); }
static double h2 (double x, double y) { return hypot (x, y); }
static struct { const char *name; f2f f; f2d d; } T2[] = {
 {"pow", p2f, p2}, {"atan2", a2f, a2}, {"hypot", h2f, h2} };
static int find (const char *n, int *bi)
{
  for (unsigned i = 0; i < sizeof T / sizeof T[0]; i++)
    if (!strcmp (T[i].name, n)) { *bi = 0; return i; }
  for (unsigned i = 0; i < 3; i++)
    if (!strcmp (T2[i].name, n)) { *bi = 1; return i; }
  return -1;
}
int call32 (const char *n, const float *x, const float *x2, float *y, long len)
{
  int bi, i = find (n, &bi); if (i < 0) return -1;
  if (bi) for (long k = 0; k < len; k++) y[k] = T2[i].f (x[k], x2[k]);
  else for (long k = 0; k < len; k++) y[k] = T[i].f (x[k]);
  return 0;
}
int call64 (const char *n, const double *x, const double *x2, double *y, long len)
{
  int bi, i = find (n, &bi); if (i < 0) return -1;
  if (bi) for (long k = 0; k < len; k++) y[k] = T2[i].d (x[k], x2[k]);
  else for (long k = 0; k < len; k++) y[k] = T[i].d (x[k]);
  return 0;
}
