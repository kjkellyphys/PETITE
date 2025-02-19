import numpy as np

# use exp from C
from libc.math cimport exp, cos, sin, sqrt, log, pi

cdef double m_electron = 0.5109989461e-3  # GeV
cdef double alpha_em = 1.0 / 137.035999139  # fine structure constant

def c_dsigma_brem_dimensionless(double[:, ::1] x, double ep, double Egamma_min):
        """Standard Model Bremsstrahlung in the Small-Angle Approximation
        e (ep) + Z -> e (epp) + gamma (w) + Z
        Outgoing kinematics given by w, d (delta), dp (delta'), and ph (phi)

        Input parameters needed:
            ep (incident electron energy)
            Z (Target Atomic Number)
        """

        cdef int i          # labels integration point
        cdef int dim = x.shape[1]
        cdef double[::1] ans = np.empty(x.shape[0], float)
        cdef double w, d, dp, ph, epp, qsq, PF, jacobian_factor, FF, T1, T2, T3, T4, dSigs

        for i in range(x.shape[0]):
            x1 = x[i,0] 
            x2 = x[i,1] 
            x3 = x[i,2] 
            x4 = x[i,3]
            
            w = Egamma_min + x1 * (ep - m_electron - Egamma_min)
            d = ep / (2 * m_electron) * (x2 + x3)
            dp = ep / (2 * m_electron) * (x2 - x3) 
            ph = (x4 - 1 / 2) * 2 * pi                

            epp = ep - w

            if (Egamma_min > w) or (w > ep - m_electron) or (m_electron > epp) or (epp > ep) or (d < 0.0) or (dp < 0.0):
                ans[i] = 0.0
            else:
                qsq = m_electron**2 * (
                    (d**2 + dp**2 - 2 * d * dp * cos(ph))
                    + m_electron**2 * ((1 + d**2) / (2 * ep) - (1 + dp**2) / (2 * epp)) ** 2
                )
                PF = (
                    8.0
                    / pi
                    * alpha_em
                    * (alpha_em / m_electron) ** 2
                    * (epp * m_electron**4)
                    / (w * ep * qsq**2)
                    * d
                    * dp
                )
                jacobian_factor = pi * ep**2 * (ep - m_electron - Egamma_min) / m_electron**2
                FF = 1 # g2_elastic(self.event_info, qsq)
                T1 = d**2 / (1 + d**2) ** 2
                T2 = dp**2 / (1 + dp**2) ** 2
                T3 = w**2 / (2 * ep * epp) * (d**2 + dp**2) / ((1 + d**2) * (1 + dp**2))
                T4 = -(epp / ep + ep / epp) * (d * dp * cos(ph)) / ((1 + d**2) * (1 + dp**2))
                ans[i] = PF * (T1 + T2 + T3 + T4) * jacobian_factor * FF

        return ans