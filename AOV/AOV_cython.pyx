
from cython cimport cdivision, boundscheck
from libc.stdlib cimport malloc, free

@cdivision
@boundscheck(False)
def compute(double freq, double[:] t, double[:] m, double avg, int npts, int r):

    cdef int i,idx
    cdef double aux,s1,s2,F

    cdef double *n_view = <double *>malloc(r * sizeof(double))
    cdef double *sum1_view = <double *>malloc(r * sizeof(double))
    cdef double *sum2_view = <double *>malloc(r * sizeof(double))

    cdef float phase_view[10000]

    for i in range(r):
        n_view[i] = 0.0
        sum1_view[i] = 0.0
        sum2_view[i] = 0.0

    for i in range(npts):
        aux = (t[i]-t[0])*freq
        phase_view[i] = aux-<int>(aux)

    for i in range(npts):
        idx = <int>(phase_view[i]*r)
        sum1_view[idx] = sum1_view[idx] + m[i]
        sum2_view[idx] = sum2_view[idx] + m[i]*m[i]
        n_view[idx] = n_view[idx] + 1

    s1 = 0.0
    s2 = 0.0
    for i in range(r):
        if (n_view[i] == 0): continue
        sum1_view[i] = sum1_view[i]/n_view[i]
        s1 = s1 + n_view[i]*(sum1_view[i]-avg)*(sum1_view[i]-avg)
        s2 = s2 + sum2_view[i]-n_view[i]*sum1_view[i]*sum1_view[i]        

    F = s1/s2
    F *= npts-r
    F /= r-1

    free(n_view)
    free(sum1_view)
    free(sum2_view)

    return F

