# This function only work with GPS, single frequency.
import copy
import datetime
import numpy as np
import os
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

from read import read_rnx


OPTIONS = { 'speed_light': 2.99792458e8, \
            'spp_iteration': 10, \
            'min_ele': 10, \
            'gravitational_constant': 3.9860050e14,\
            'semi_major_axis': 6378136.55,\
            'eccentricity': 8.1819190842622e-2, \
            'rotation_rate': 7.2921151467e-5}


def cal_distance(pos1, pos2):
    """
    calculate distance between two points in ECEF
    pos: [X, Y, Z]
    """
    if type(pos1) == list:
        pos1_np = np.array(pos1)
    else:
        pos1_np = pos1
    if type(pos2) == list:
        pos2_np = np.array(pos2)
    else:
        pos2_np = pos2

    return np.linalg.norm(pos1_np - pos2_np)


def blh2xyz(pos):

    a = OPTIONS['semi_major_axis']
    e = OPTIONS['eccentricity']

    B, L, H = pos[0:2]

    N = a / (1 - (e * np.sin(B))**2)**(0.5)
    temp = (N + H) * np.cos(B)
    x = temp * np.cos(L)
    y = temp * np.sin(L)
    z = (N * (1 - e**2) + H) * np.sin(B)
    return [x, y, z]


def xyz2blh(pos):
    
    a = OPTIONS['semi_major_axis']
    e = OPTIONS['eccentricity']

    X, Y, Z = pos[0:3]

    L = np.arctan2(Y, X)
    p = (X**2 + Y**2)**(0.5)
    B = np.arctan2(Z / p, (1 - e**2))
    B0 = 999999
    H = 0
    while np.abs(B - B0) > 1e-10:
        B0 = B
        N = a / (1 - (e * np.sin(B0))**2)**(0.5)
        H = p / np.cos(B0) - N
        B = np.arctan2(Z / p, (1 - N / (N + H) * e**2))
        
    return [B, L, H]


def ecef2enu(R, B, L):

    sin_lambda = np.sin(L)
    cos_lambda = np.cos(L)
    sin_phi = np.sin(B)
    cos_phi = np.cos(B)

    trans = np.array([
        [-sin_lambda, cos_lambda, 0], \
        [-cos_lambda*sin_phi, -sin_lambda*sin_phi, cos_phi], \
        [cos_lambda*cos_phi, sin_lambda*cos_phi, sin_phi]])

    return np.matmul(trans, R)


def xyz2enu(delta, Xr):
    [B, L, H] = xyz2blh(Xr)
    return ecef2enu(delta, B, L)


def cal_azi(pos_r, pos_s):
    delta = np.array(pos_s) - np.array(pos_r)
    [E, N, U] = xyz2enu(delta, pos_r)
    temp = np.linalg.norm([E, N, U])
    return np.arctan2(E / temp, (N / temp))


def cal_ele(pos_r, pos_s):
    delta = np.array(pos_s) - np.array(pos_r)
    [E, N, U] = xyz2enu(delta, pos_r)
    temp = np.linalg.norm([E, N, U])
    return np.arcsin(U / temp)


def cal_leap(time):
    """
    calculate leap seconds
    """
    if time > datetime.datetime(2017,1,1,0,0,0):
        return 18
    elif time > datetime.datetime(2015,7,1,0,0,0):
        return 17
    elif time > datetime.datetime(2012,7,1,0,0,0):
        return 16
    elif time > datetime.datetime(2009,1,1,0,0,0):
        return 15
    elif time > datetime.datetime(2006,1,1,0,0,0):
        return 14
    elif time > datetime.datetime(1999,1,1,0,0,0):
        return 13
    elif time > datetime.datetime(1997,7,1,0,0,0):
        return 12
    elif time > datetime.datetime(1996,1,1,0,0,0):
        return 11
    elif time > datetime.datetime(1994,7,1,0,0,0):
        return 10
    elif time > datetime.datetime(1993,7,1,0,0,0):
        return 9
    elif time > datetime.datetime(1992,7,1,0,0,0):
        return 8
    elif time > datetime.datetime(1991,1,1,0,0,0):
        return 7
    elif time > datetime.datetime(1990,1,1,0,0,0):
        return 6
    elif time > datetime.datetime(1988,1,1,0,0,0):
        return 5
    elif time > datetime.datetime(1985,7,1,0,0,0):
        return 4
    elif time > datetime.datetime(1983,7,1,0,0,0):
        return 3
    elif time > datetime.datetime(1982,7,1,0,0,0):
        return 2
    elif time > datetime.datetime(1981,7,1,0,0,0):
        return 1
    else:
        return 0


def utc2gpstw(time):
    """
    convert utc to gpst
    """
    t0 = datetime.datetime(1980, 1, 6, 0, 0, 0)
    seconds = (time - t0).total_seconds() + cal_leap(time)
    
    week = int(seconds / 604800)
    second = seconds - week * 604800

    return [week, second]


def utc2gpst(time):
    """
    convert utc to gpst
    """
    t0 = datetime.datetime(1980, 1, 6, 0, 0, 0)
    seconds = (time - t0).total_seconds() + cal_leap(time)
    
    return seconds


def gpst2utc(time):
    """
    error in this function
    """
    t0 = datetime.datetime(1980, 1, 6, 0, 0, 0)
    t1 = t0 + datetime.timedelta(seconds=time)
    return t1 + datetime.timedelta(seconds=-cal_leap(t1))


def gpst2gpstw(time):
    week = int(time / 604800)
    second = time - week * 604800

    return [week, second]


def sel_nav(time, prn, nav):
    """
    select nav record
    time: transmite time of singal, satellite clock error has been considered
    """
    time_diff = [99999999] * len(nav)

    for i in range(len(nav)):
        if nav[i]['prn'] == prn:
            t2 = utc2gpst(nav[i]['time']) 
            time_diff[i] = np.abs(time - t2)
    return np.argmin(time_diff)
  

def cal_Ek(t, toe, sqrtA, e, M0, deltaN):
    """
    theory background: see esa gnss book page 57 for details
    """
    # constants
    mu = OPTIONS['gravitational_constant']
    
    tk = t - toe
    if tk >  302400: tk = 604800 - tk
    if tk < -302400: tk = 604800 + tk

    Mk = M0 + (mu**(0.5) / sqrtA**3 + deltaN) * tk

    Ek = 0
    Ek_next = Mk
    while np.abs(Ek - Ek_next) > 1e-14:
        Ek = Ek_next
        Ek_next = Ek - (Ek - e * np.sin(Ek) - Mk) / (1 - e * np.cos(Ek))
    
    return Ek
 

def cal_sat_clk(time, prn, nav):
    """
    calculate satellite clock
    time: transmite time of singal, satellite clock error has been considered
    """
    index_nav = sel_nav(time, prn, nav)
    F = - 4.442807633e-10
    t2 = utc2gpst(nav[index_nav]['time'])
    Ek = cal_Ek(time, t2, nav[index_nav]['sqrtA'], \
        nav[index_nav]['e'], nav[index_nav]['M0'], \
        nav[index_nav]['deltaN'])

    delta_tr = F * nav[index_nav]['e'] * nav[index_nav]['sqrtA'] * np.sin(Ek)

    return nav[index_nav]['clk_bias'] \
        + nav[index_nav]['clk_drift'] * (time - t2) \
        + nav[index_nav]['clk_drift_rate'] * (time - t2)**2 \
        + delta_tr


def cal_iono(alpha, beta, phi_u, lambda_u, E, A, gpst):
    """
    calculate broadcast ionosphere model
    see https://navcen.uscg.gov/pdf/gps/IS_GPS_200L.pdf for details.
    """
    c = OPTIONS['speed_light']

    PSI = 0.0137 / (E / np.pi + 0.11) - 0.022

    phi_i = phi_u / np.pi + PSI * np.cos(A)
    if phi_i < -0.416:
        phi_i = -0.416
    elif phi_i > 0.416:
        phi_i = 0.416
    
    lambda_i = lambda_u / np.pi + PSI * np.sin(A) / np.cos(phi_i * np.pi)

    phi_m = phi_i + 0.064 * np.cos((lambda_i - 1.617) * np.pi)

    t = 4.32e4 * lambda_i + gpst2gpstw(gpst)[1]
    if t >= 86400:
        t -= 86400
    elif t < 0:
        t += 86400
    
    F = 1 + 16 * (0.53 - E / np.pi)**3

    amp = 0
    for i in range(len(alpha)):
        amp += alpha[i] * phi_m**i
    if amp < 0: amp = 0
    
    per = 0
    for i in range(len(beta)):
        per += beta[i] * phi_m**i
    if per < 72000: per = 72000
    
    x = 2 * np.pi * (t - 50400) / per

    if np.abs(x) < 1.57:
        iono = F * (5e-9 + amp * (1 - x**2 / 2 + x**4 / 24))
    else:
        iono = F * 5e-9

    return c * iono
    

def cal_MF(E):
    return 1.001 / np.sqrt(0.002001 + (np.sin(E))**2)


def cal_trop(pos_rcv, pos_sat, model):
    """
    calculate troposphere, just for test!!!
    """
    trop = 0
    H = xyz2blh(pos_rcv)[2]
    p = 1013.25 * (1 - 2.2557e-5 * H)**5.2568
    RH = 0.5 * np.exp(-0.0006396 * H)
    T = 15.0 - 6.5e-3 * H + 273.15
    e = 6.108 * np.exp((17.15 * T - 4684.0) / (T - 38.45)) * RH / 100
    # Ts = 20 - 0.0065 * H
    # Ps = 1013.25 * (1 - 0.0000266 * H)**5.225
    ele = cal_ele(pos_rcv, pos_sat)
    if model == 'hopfield':
        # hw = 11000
        # hd = 40136 + 148.72 * (T - 273.16)
        # Kw = 155.2e-7 * 4810 / T**2 * es * (hw - hs)
        # Kd = 155.2e-7 * Ps / Ts * (hd - hs)
        # trop = Kd / np.sin((E**2 + 6.25)**0.5) + Kw / np.sin((E**2 + 2.25)**0.5)
        pass # todo
    elif model == 'saastamoinen':
        z = np.pi / 2 - ele
        trop = 0.002277 / np.cos(z) * (p + (1255 / T + 0.05) * e - (np.tan(z))**2)
    else:
        dry = 2.3 * np.exp(-0.116 * 0.001 * H)
        ele = cal_ele(pos_rcv, pos_sat)
        
        trop = (dry + 0.1) * 1.001 / np.sqrt(0.002001 + (np.sin(ele))**2)
    return trop


def res_code(p, Xr, Xs, dtr, t, azi, ele, alpha, beta, prn, nav):
    """
    calculate residual of code
    todo: what if there is no alpha or beta in nav
    """
    clk_sat = cal_sat_clk(t, prn, nav)
    c = OPTIONS['speed_light']
    [lat_rcv, lon_rcv, _] = xyz2blh(Xr)
    iono = cal_iono(alpha, beta, lat_rcv, lon_rcv, ele, azi, t)
    trop = cal_trop(Xr, Xs, 'saastamoinen')
    v = p - (cal_distance(Xr, Xs) + dtr - c * clk_sat + iono + trop)

    return v


def eph2pos(time, t, toe, toes, sqrtA, e, M0, omega, i0, omega0, 
            deltaN, dotI, dotOmega, Cuc, Cus, 
            Crc, Crs, Cic, Cis):
    """
    theory background: see esa gnss book page 57 for details
    """
    
    # constants
    mu = OPTIONS['gravitational_constant'] # 3.9860050e14   m3/s2 for GPS and QZSS, 
                    # 3.986004418e14 m3/s2 for Galileo
    omegaE = OPTIONS['rotation_rate']

    tk = t - toe
    if tk >  302400: tk = 604800 - tk
    if tk < -302400: tk = 604800 + tk

    Mk = M0 + (mu**(0.5) / sqrtA**3 + deltaN) * tk

    Ek = 0
    Ek_next = Mk
    while np.abs(Ek - Ek_next) > 1e-14:
        Ek = Ek_next
        Ek_next = Ek - (Ek - e * np.sin(Ek) - Mk) / (1 - e * np.cos(Ek))
    vk = np.arctan2((1-e*e)**0.5 * np.sin(Ek), (np.cos(Ek) - e))
    phi = omega + vk
    sin2phi = np.sin(2 * phi)
    cos2phi = np.cos(2 * phi)

    uk = phi + Cuc * cos2phi + Cus * sin2phi

    rk = sqrtA * sqrtA * (1 - e * np.cos(Ek)) + Crc * cos2phi + Crs * sin2phi
    
    ik = i0 + dotI * tk + Cic * cos2phi + Cis * sin2phi

    lambdaK = omega0 + (dotOmega - omegaE) * tk - omegaE * toes

    X = rk * (np.cos(uk) * np.cos(lambdaK) \
        - np.sin(uk) * np.cos(ik) * np.sin(lambdaK))
    Y = rk * (np.cos(uk) * np.sin(lambdaK) \
        + np.sin(uk) * np.cos(ik) * np.cos(lambdaK))
    Z = rk * np.sin(uk) * np.sin(ik)

    delta_alpha = omegaE * np.abs(time - t)

    X = X * np.cos(delta_alpha) + Y * np.sin(delta_alpha)
    Y = -X * np.sin(delta_alpha) + Y * np.cos(delta_alpha)

    # F = - 4.442807633e-10
    # delta_tr = F * e * sqrtA * np.sin(Ek)

    # clk_error =  clk_bias + clk_drift * tk \
    #     + clk_drift_rate * tk**2 \
    #     + delta_tr
    
    return [X, Y, Z]


def cal_sat_pos(time, prn, pseudo, nav):
    """
    time: obs time
    """
    c = OPTIONS['speed_light']
    time_diff = [99999999] * len(nav)
    t1 = []
    for i in range(len(nav)):
        if nav[i]['prn'] == prn:
            trans_time = time - pseudo / c
            trans_time += cal_sat_clk(trans_time, prn, nav)
            
            t1.append(trans_time)
            t2 = utc2gpst(nav[i]['time']) 
            time_diff[i] = np.abs(trans_time - t2)
        else:
            t1.append(99999999)
    
    index_min = np.argmin(time_diff)
      
    return eph2pos(time, t1[index_min], utc2gpst(nav[index_min]['time']), nav[index_min]['Toe'], nav[index_min]['sqrtA'], nav[index_min]['e'], \
            nav[index_min]['M0'], nav[index_min]['omega'], nav[index_min]['i0'], \
            nav[index_min]['OMEGA'], nav[index_min]['deltaN'], nav[index_min]['idot'], \
            nav[index_min]['omega_dot'], nav[index_min]['Cuc'], nav[index_min]['Cus'], \
            nav[index_min]['Crc'], nav[index_min]['Crs'], nav[index_min]['Cic'], \
            nav[index_min]['Cis'])


def sat_id2prn(sat_id):
    return int(sat_id[1:3])


def lsq(H, W, v):
    """
    least square 
    """
    HTW = np.matmul(np.transpose(H), W)
    a = np.matmul(HTW, H) # H^T * W * H
    b = np.matmul(HTW, v) # H^T * W * v
    X = np.linalg.solve(a, b) # a\b

    return X


def spp_epoch(obs, nav, pos_r, iono_alpha, iono_beta):
    """
    spp for single epoch
    """
    iteration = OPTIONS['spp_iteration']
    min_ele = OPTIONS['min_ele']
    X = [0, 0, 0, 0]
    pos_sat = []
    dtr = 0
    # step 1. calculate position of observed satellites
    for i in range(obs['sats_num']):
        pos_sat.append(cal_sat_pos(utc2gpst(obs['time']), \
            sat_id2prn(obs['obs'][i]['sat_id']), \
            obs['obs'][i]['C1'][0], nav))

    for _ in range(iteration):
        H = []
        W = []
        v = []
        for i in range(obs['sats_num']):
            pos_s = pos_sat[i]
            # step 2. calculate azimuth and elevation of observed satellite
            azi = cal_azi(pos_r, pos_s)
            ele = cal_ele(pos_r, pos_s)
            if ele < min_ele * np.pi / 180:
                continue
            else:
                dis = cal_distance(pos_s, pos_r)
                H.append(np.append((np.array(pos_r) - np.array(pos_s)) / dis, 1))
                v.append(res_code(obs['obs'][i]['C1'][0], pos_r, pos_s, dtr, \
                    utc2gpst(obs['time']), azi, ele, \
                    iono_alpha, iono_beta, \
                    sat_id2prn(obs['obs'][i]['sat_id']), nav))
                W.append((np.sin(ele))**2)
        if len(W) > 4:
            X = lsq(H, np.diag(W), v)
        else:
            print(f'lack of satellite, sats_num: {len(W)}')
            return [0, 0, 0]
        pos_r += X[0:3]
        dtr += X[3]
        if np.abs(X[0])<0.001 and np.abs(X[1])<0.001 and np.abs(X[2])<0.001:
            break
    return pos_r


def spp(obs, nav):
    """
    spp
    """
    res = []
    pos_r = obs['approx_pos']
    iono_alpha = nav['ion_alpha']
    iono_beta = nav['ion_beta']
    epoch_num = len(obs['body'])
    for i in range(epoch_num):
        temp = spp_epoch(obs['body'][i], nav['body'], pos_r, iono_alpha, iono_beta)
        print(f'epoch: {i}/{epoch_num}, {temp}')
        res.append(temp)
    return res


def draw_pos_error(err_x, err_y, err_z):
    """
    draw
    """
    fig = plt.figure()
    gs = fig.add_gridspec(3, hspace=0)
    axs = gs.subplots(sharex=True)
    fig.suptitle('error (m)')
    axs[0].plot(err_x)
    axs[1].plot(err_y)
    axs[2].plot(err_z)
    axs[0].set(ylabel='x')
    axs[1].set(ylabel='y')
    axs[2].set(xlabel='epoch', ylabel='z')
    
    for ax in axs:
        ax.label_outer()
        # ax.grid()
    plt.show()


def evaluate(res, reference):
    err_x = []
    err_y = []
    err_z = []
    for i in range(len(res)):
        err_x.append(res[i][0] - reference[0])
        err_y.append(res[i][1] - reference[1])
        err_z.append(res[i][2] - reference[2])
    
    draw_pos_error(err_x, err_y, err_z)


dirname = os.path.dirname(__file__)

nav3 = read_rnx(os.path.join(dirname, "data/brdc3550.15n"))
obs1 = read_rnx(os.path.join(dirname, "data/iqal3550.15o")) 

pos_iqal = [1.03600086899334e+06, -2.63145559079701e+06, 5.69781974243222e+06]
res = spp(obs1, nav3)
evaluate(res, pos_iqal)
