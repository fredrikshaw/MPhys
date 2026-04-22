import numpy as np

def angular_poly(theta):
    # 28 cos 2θ + cos 4θ + 35
    return 28.0 * np.cos(2.0 * theta) + np.cos(4.0 * theta) + 35.0

def ann_2p(alpha, theta, G_N=1.0, r_g=1.0):
    prefactor = alpha**18 * G_N / (2**24 * np.pi * (alpha**2 + 4.0)**4 * r_g**4)
    bracket = (6.0 * alpha**3 + 40.0 * alpha - 3.0 * (alpha**2 + 4.0)**2 * np.arctan(2.0 / alpha))
    return prefactor * bracket**2 * angular_poly(theta)

def ann_3d(alpha, theta, G_N=1.0, r_g=1.0):
    prefactor = alpha**20 * G_N / (2**4 * 3**16 * np.pi * r_g**4)
    return prefactor * (np.sin(theta)**4) * angular_poly(theta)

def ann_4f(alpha, theta, G_N=1.0, r_g=1.0):
    prefactor = alpha**24 * G_N / (5**(-2) * 2**24 * np.pi * r_g**4)
    return prefactor * (np.sin(theta)**8) * angular_poly(theta)

diff_power_ann_dict = {
    "2p": ann_2p,
    "3d": ann_3d,
    "4f": ann_4f,
}


def trans_6g_5g(alpha, theta, G_N=1.0, r_g=1.0):
    prefactor = 2**28 * 3**4 * 5**5 * (alpha**12) * G_N / (11**22 * np.pi * r_g**4)
    return prefactor * (np.sin(theta)**4)

def trans_7h_6h(alpha, theta, G_N=1.0, r_g=1.0):
    prefactor = 2**31 * 3**7 * 5**2 * 7**6 * (alpha**12) * G_N / (13**26 * np.pi * r_g**4)
    return prefactor * (np.sin(theta)**4)

def trans_5f_4f(alpha, theta, G_N=1.0, r_g=1.0):
    # 22252 α^12 G_N sin^4 θk / (334 π r_g^4)
    prefactor = 2**22 * 5**2 * (alpha**12) * G_N / (3**34 * np.pi * r_g**4)
    return prefactor * (np.sin(theta)**4)

diff_power_trans_dict = {
    "6g 5g": trans_6g_5g,
    "7h 6h": trans_7h_6h,
    "5f 4f": trans_5f_4f,
}