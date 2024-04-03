#pragma once

//#define DEBUG_CONSOLE
#ifdef DEBUG_CONSOLE
#  define DBG(x) x;
#else
#  define DBG(x) {}
#endif

const double pi = 3.14159265358979323846;
typedef double dtype;

#define OMP_TIMER
#ifdef OMP_TIMER
#define TIME(str, s, e, x) s = omp_get_wtime(); x; e = omp_get_wtime(); cout << str << ": " << e - s << " secs\n";
#else
#define TIME(str, s, e, x) x
#endif
