
from cython cimport cdivision, boundscheck
from libc.stdlib cimport malloc, free
from math import sin, cos, pi

cdef extern from "complex.h":
    double complex I(double complex)
    double cabs(double complex)
    double conj(double complex)

@cdivision
@boundscheck(False)
def compute(double freq, double[:] t, double[:] m, double magvariance, int nharmonics, int npts):

    cdef int i,j
    cdef double aux,theta_aov,phase
    cdef complex phi_dot_psi,phi_dot_phi,alpha

    cdef double EPSILON
    EPSILON=1.0e-8

    cdef double complex z_view[10000]
    cdef double complex psi_view[10000]
    cdef double complex phi_view[10000]
    cdef double complex zn_view[10000]

    for i in range(npts):
        aux = (t[i]-t[0])*freq
        phase = 2.0*pi*(aux-<int>(aux))
        z_view[i] = complex(cos(phase),sin(phase))
        phase *= nharmonics
        psi_view[i] = m[i]*complex(cos(phase),sin(phase))
        phi_view[i] = complex(1.0, 0.0)
        zn_view[i] = complex(1.0, 0.0)

    theta_aov = 0.0

    for i in range(2*nharmonics):
        alpha = complex(0.0, 0.0)
        phi_dot_psi = complex(0.0, 0.0)
        phi_dot_phi = complex(0.0, 0.0)

        for j in range(npts):
            phi_dot_phi +=  phi_view[j] * conj(phi_view[j])
            alpha += z_view[j]*phi_view[j]
            phi_dot_psi += psi_view[j] * conj(phi_view[j])

        aux = cabs(phi_dot_phi)
        if (aux < EPSILON):
            aux = EPSILON

        alpha /= aux
        theta_aov += cabs(phi_dot_psi)*cabs(phi_dot_psi)/aux
        for j in range(npts):
            phi_view[j] = phi_view[j]*z_view[j] - alpha*zn_view[j]*conj(phi_view[j])
            zn_view[j] = zn_view[j]*z_view[j]

    aux = magvariance-theta_aov
    if (aux < EPSILON):
        aux = EPSILON
    theta_aov = ((npts - 2.0*nharmonics-1.0)*theta_aov/2.0/nharmonics/aux)

    #free(z_view)
    #free(psi_view)
    #free(phi_view)
    #free(zn_view)

    return theta_aov

