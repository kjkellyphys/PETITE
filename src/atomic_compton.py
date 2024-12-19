import numpy as np
try:
    from .physical_constants import *
except:
    from physical_constants import *

def expr1(a, b):
    delta = (a - b)**2 + 1
    log_term = np.log(delta) - 2 * np.log(b)

    # Numerator parts
    num_part1 = 8 * b * (a**2 - 6 * a * b + 1) * log_term

    num_part2_inner = (
        -a * (a**2 + 1)**3 * (3 * a**2 + 5)
        + 6 * (a**4 + 4 * a**2 - 5) * b**5
        - 4 * a * (6 * a**4 + 23 * a**2 - 31) * b**4
        + (a - 1) * (a + 1) * (39 * a**4 + 190 * a**2 + 55) * b**3
        - a * (a**2 + 1) * (33 * a**4 + 98 * a**2 - 111) * b**2
        + (a**2 + 1)**2 * (15 * a**4 + 32 * a**2 - 23) * b
    )
    num_part2 = (a**2 + 1) * num_part2_inner / delta**2

    tan_arg = b / (a**2 - a * b + 1)
    cot_arg = a
    num_part3_coeff = (
        3 * (a**2 + 1)**4
        - 2 * a * (3 * a**4 + 10 * a**2 + 15) * (a**2 + 1) * b
        + 6 * (a**6 + 5 * a**4 + 15 * a**2 - 5) * b**2
    )
    num_part3 = num_part3_coeff * (np.arctan(tan_arg) + np.arctan(1/cot_arg))

    # Combine all parts
    numerator = num_part1 + num_part2 + num_part3

    denominator = 8 * (a**2 + 1)**4 * b

    # The full expression
    result = -numerator / denominator

    return result

def expr2(a, b):
    delta = (a - b)**2 + 1
    denominator = 8 * (a**2 + 1)**4 * b * delta**2

    # First big term
    term1_inner = (
        a * (a**2 + 1)**3 * (3 * a**2 + 5)
        - 6 * (a**4 + 4 * a**2 - 5) * b**5
        + 4 * a * (6 * a**4 + 23 * a**2 - 31) * b**4
        - (a - 1) * (a + 1) * (39 * a**4 + 190 * a**2 + 55) * b**3
        + a * (a**2 + 1) * (33 * a**4 + 98 * a**2 - 111) * b**2
        - (a**2 + 1)**2 * (15 * a**4 + 32 * a**2 - 23) * b
    )
    term1 = (a**2 + 1) * term1_inner

    # Second big term (with Pi)
    term2_coeff = (
        3 * (a**2 + 1)**4
        - 2 * a * (3 * a**4 + 10 * a**2 + 15) * (a**2 + 1) * b
        + 6 * (a**6 + 5 * a**4 + 15 * a**2 - 5) * b**2
    )
    term2 = np.pi * term2_coeff * delta**2

    # Third big term (with arccot)
    term3 = term2_coeff * delta**2 * np.arctan(1/a)

    # Fourth big term
    log_term = np.log(delta) - 2 * np.log(b)
    term4_inner = (
        8 * b * (a**2 - 6 * a * b + 1) * log_term
        + term2_coeff * np.arctan(b / (a**2 - a * b + 1))
    )
    term4 = delta**2 * term4_inner

    # The full expression
    numerator = term1 + term2 - term3 - term4
    result = numerator / denominator

    return result

def combine(a, b):
    # return expr1(a, b)*np.heaviside(b - (a^2+1)/a, 1) + expr2(a, b)*np.heaviside((a^2+1)/a - b, 1)
    if a**2 + 1 < a*b:
        return expr1(a, b)
    else:
        return expr2(a, b)

def sigma_atomic_comp(k,mV, Zeff):
    # Calculate components of the expression

    m_e=m_electron 

    pmin=(2*m_e*k-mV**2)/k/2
    Lambda = Zeff*alpha_em*m_e

    a=m_e/Lambda 
    b=mV**2/(2*k*Lambda)

    s=2*k*m_e+m_e**2

    beta = 2*alpha_em/np.pi*(np.log(s/m_e**2) - 1.0)

    return( (4.0*np.pi*alpha_em)*((mV**2+2*m_e**2)* (2/3/Lambda/m_e)*(1/k**2)*(beta/4)*combine(a,b)) )