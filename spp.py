# This function only work with GPS, single frequency.
import copy
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from constants import OPTIONS
from mpl_toolkits.axes_grid1 import make_axes_locatable
from read import read_rnx, merge_nav



def cal_distance(pos1, pos2=[0, 0, 0]):
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
    if np.abs(X) < 1e-5:
        return [0, 0, 0]

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
    if time >= datetime.datetime(2017,1,1,0,0,0):
        return 18
    elif time >= datetime.datetime(2015,7,1,0,0,0):
        return 17
    elif time >= datetime.datetime(2012,7,1,0,0,0):
        return 16
    elif time >= datetime.datetime(2009,1,1,0,0,0):
        return 15
    elif time >= datetime.datetime(2006,1,1,0,0,0):
        return 14
    elif time >= datetime.datetime(1999,1,1,0,0,0):
        return 13
    elif time >= datetime.datetime(1997,7,1,0,0,0):
        return 12
    elif time >= datetime.datetime(1996,1,1,0,0,0):
        return 11
    elif time >= datetime.datetime(1994,7,1,0,0,0):
        return 10
    elif time >= datetime.datetime(1993,7,1,0,0,0):
        return 9
    elif time >= datetime.datetime(1992,7,1,0,0,0):
        return 8
    elif time >= datetime.datetime(1991,1,1,0,0,0):
        return 7
    elif time >= datetime.datetime(1990,1,1,0,0,0):
        return 6
    elif time >= datetime.datetime(1988,1,1,0,0,0):
        return 5
    elif time >= datetime.datetime(1985,7,1,0,0,0):
        return 4
    elif time >= datetime.datetime(1983,7,1,0,0,0):
        return 3
    elif time >= datetime.datetime(1982,7,1,0,0,0):
        return 2
    elif time >= datetime.datetime(1981,7,1,0,0,0):
        return 1
    else:
        return 0


def utc2gpstw(time, leap=True):
    """
    convert utc to gpst
    """
    t0 = datetime.datetime(1980, 1, 6, 0, 0, 0)
    seconds = (time - t0).total_seconds()
    if leap: seconds += cal_leap(time)
    
    week = int(seconds / 604800)
    second = seconds - week * 604800

    return [week, second]


def gpst2bdst(time):
    """
    Convert gpst to bdst.
    second
    """
    return time - 1356 * 604800 - 14


def gpst2galst(time):
    """
    Convert gpst to bdst.
    second
    """
    return time - 1024 * 604800


def bdst2gpst(time):
    """
    Convert bdst to gpst.
    second
    """
    return time + 1356 * 604800 + 14


def galst2gpst(time):
    """
    Convert bdst to gpst.
    second
    """
    return time + 1024 * 604800


def utc2gpst(time, leap=True):
    """
    convert utc to gpst
    """
    t0 = datetime.datetime(1980, 1, 6, 0, 0, 0)
    seconds = (time - t0).total_seconds() 
    if leap: seconds += cal_leap(time)
    
    return seconds


def gpst2utc(time, leap=True):
    """
    error in this function
    """
    t0 = datetime.datetime(1980, 1, 6, 0, 0, 0)
    t1 = t0 + datetime.timedelta(seconds=time)
    if leap:
        return t1 + datetime.timedelta(seconds=-cal_leap(t1))
    else:
        return t1


def gpst2gpstw(time):
    week = int(time / 604800 + 1e-5)
    second = time - week * 604800

    return [week, second]


def gpst2glost(time):
    pass


def gpst2other(time, sys):
    if sys == 'G':
        pass
    elif sys == 'E':
        time = gpst2galst(time)
    elif sys == 'C':
        time = gpst2bdst(time)
    elif sys == 'R':
        time = gpst2glost(time)
    else:
        time = 0

    return time



def other2gpst(time, sys):
    if sys == 'G':
        pass
    elif sys == 'E':
        time = galst2gpst(time)
    elif sys == 'C':
        time = bdst2gpst(time)
    elif sys == 'R':
        pass # todo
    else:
        time = 0
    return time


def sel_nav(time, sat_id, nav):
    """
    select nav record
    time: transmite time of singal, satellite clock error has been considered
    """
    time = gpst2other(time, sat_id[0])
    
    time_diff = [99999999] * len(nav)

    for i in range(len(nav)):
        if nav[i]['sat_id'] == sat_id:
            t2 = gpst2other((nav[i]['week'] * 604800 + nav[i]['Toe']), sat_id[0])
            time_diff[i] = np.abs(time - t2)
    return np.argmin(time_diff)
  

def cal_Ek(trans_time, week, toes, sqrtA, e, M0, deltaN, sys):
    """
    theory background: see esa gnss book page 57 for details
    """
    trans_time = gpst2other(trans_time, sys)
    
    # constants
    mu = OPTIONS['gravitational_constant'] # 3.9860050e14   m3/s2 for GPS and QZSS, 
                    # 3.986004418e14 m3/s2 for Galileo
    
    tk = trans_time - (week*604800+toes)
    while tk > 302400 or tk < -302400:
        if tk >  302400: tk = 604800 - tk
        if tk < -302400: tk = 604800 + tk

    Mk = M0 + (mu**(0.5) / sqrtA**3 + deltaN) * tk

    Ek = 0
    Ek_next = Mk
    while np.abs(Ek - Ek_next) > 1e-14:
        Ek = Ek_next
        Ek_next = Ek - (Ek - e * np.sin(Ek) - Mk) / (1 - e * np.cos(Ek))
    return Ek


def cal_sat_clk(time, sat_id, nav):
    """
    calculate satellite clock
    time: transmite time of singal, satellite clock error has been considered
    """

    index_nav = sel_nav(time, sat_id, nav)
    # F = - 4.442807633e-10
    F = - 2 * OPTIONS['gravitational_constant']**0.5 / OPTIONS['speed_light']**2
    t2 = utc2gpst(nav[index_nav]['time'],sat_id[0]=='R')
    Ek = cal_Ek(time, nav[index_nav]['week'], nav[index_nav]['Toe'], nav[index_nav]['sqrtA'], \
        nav[index_nav]['e'], nav[index_nav]['M0'], \
        nav[index_nav]['deltaN'], sat_id[0])

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
    if alpha == None:
        return 0
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


def res_code(p, Xr, Xs, dtr, t, azi, ele, alpha, beta, sat_id, nav, clk_sat):
    """
    calculate residual of code
    todo: what if there is no alpha or beta in nav
    """
    # clk_sat = cal_sat_clk(t, sat_id, nav)
    c = OPTIONS['speed_light']
    [lat_rcv, lon_rcv, _] = xyz2blh(Xr)
    iono = cal_iono(alpha, beta, lat_rcv, lon_rcv, ele, azi, t)
    trop = cal_trop(Xr, Xs, 'saastamoinen')
    if sat_id[0] == 'G':
        dtr_temp = dtr[0]
    elif sat_id[0] == 'R':
        dtr_temp = dtr[1]
    elif sat_id[0] == 'E':
        dtr_temp = dtr[2]
    elif sat_id[0] == 'C':
        dtr_temp = dtr[3]
    else:
        dtr_temp = 0
    v = p - (cal_distance(Xr, Xs) + dtr_temp - c * clk_sat + iono + trop)

    return v


def eph2pos(time, trans_time, toc, week, toes, 
            sqrtA, e, M0, omega, i0, omega0, 
            deltaN, dotI, dotOmega, Cuc, Cus, 
            Crc, Crs, Cic, Cis, sys, prn, 
            clk_bias, clk_drift, clk_drift_rate):
    """
    theory background: see esa gnss book page 57 for details
    """
    time = gpst2other(time, sys)
    trans_time = gpst2other(trans_time, sys)
    toc = gpst2other(toc, sys)
    # constants
    mu = OPTIONS['gravitational_constant'] # 3.9860050e14   m3/s2 for GPS and QZSS, 
                    # 3.986004418e14 m3/s2 for Galileo
    omegaE = OPTIONS['rotation_rate']

    tk = trans_time - (week*604800+toes)
    # tk = gpst2gpstw(trans_time)[1] - toes
    # if sys == 'C': tk -= 14
    while tk > 302400 or tk < -302400:
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
    if sys == 'C' and (int(prn) <= 5 or int(prn) >= 59): # BDS GEO
        lambdaK = omega0 + dotOmega * tk - omegaE * toes
        X = rk * (np.cos(uk) * np.cos(lambdaK) \
            - np.sin(uk) * np.cos(ik) * np.sin(lambdaK))
        Y = rk * (np.cos(uk) * np.sin(lambdaK) \
            + np.sin(uk) * np.cos(ik) * np.cos(lambdaK))
        Z = rk * np.sin(uk) * np.sin(ik)

        Rx = np.array([[1, 0, 0], \
            [0, np.cos(-5*np.pi/180), np.sin(-5*np.pi/180)], \
            [0, -np.sin(-5*np.pi/180), np.cos(-5*np.pi/180)]])
        Rz = np.array([[np.cos(omegaE * tk), np.sin(omegaE * tk), 0], \
            [-np.sin(omegaE * tk), np.cos(omegaE * tk), 0], \
            [0, 0, 1]])
        
        pos = np.matmul(np.matmul(Rz, Rx), [X, Y, Z])
    else:
        lambdaK = omega0 + (dotOmega - omegaE) * tk - omegaE * toes

        X = rk * (np.cos(uk) * np.cos(lambdaK) \
            - np.sin(uk) * np.cos(ik) * np.sin(lambdaK))
        Y = rk * (np.cos(uk) * np.sin(lambdaK) \
            + np.sin(uk) * np.cos(ik) * np.cos(lambdaK))
        Z = rk * np.sin(uk) * np.sin(ik)
        pos = [X, Y, Z]
    delta_alpha = omegaE * np.abs(time - trans_time)

    rotation = np.array([[np.cos(delta_alpha), np.sin(delta_alpha),0], 
        [-np.sin(delta_alpha), np.cos(delta_alpha),0], 
        [0,0,1]])
    
    F = - 2 * OPTIONS['gravitational_constant']**0.5 / OPTIONS['speed_light']**2
    delta_tr = F * e * sqrtA * np.sin(Ek)
    sat_clk = clk_bias + clk_drift * (trans_time - toc) \
        + clk_drift_rate * (trans_time - toc)**2 + delta_tr
    return [np.matmul(rotation, pos), sat_clk]


def eph2pos_glo(time, t):
    """
    calculate glo sat pos by broadcast 
    """
    pass # todo


def cal_sat_pos(time, sat_id, pseudo, nav):
    """
    time: obs time
    """
    if pseudo < 1:
        return [0, 0, 0]
    c = OPTIONS['speed_light']
    time_diff = [99999999] * len(nav)
    t1 = []
    for i in range(len(nav)):
        if nav[i]['sat_id'] == sat_id:
            trans_time = time - pseudo / c
            trans_time += cal_sat_clk(trans_time, sat_id, nav)
            
            t1.append(trans_time)
            t2 = other2gpst(nav[i]['week'] * 604800 + nav[i]['Toe'], sat_id[0]) 
            time_diff[i] = np.abs(trans_time - t2)
        else:
            t1.append(99999999)
    index_min = np.argmin(time_diff)
    dtoe = OPTIONS['max_dtoe_'+sat_id[0]]
    if time_diff[index_min] > dtoe:
        return [0, 0, 0]

    return eph2pos(time, t1[index_min], utc2gpst(nav[index_min]['time'], sat_id[0]=='R'), nav[index_min]['week'], \
        nav[index_min]['Toe'], nav[index_min]['sqrtA'], nav[index_min]['e'], \
        nav[index_min]['M0'], nav[index_min]['omega'], nav[index_min]['i0'], \
        nav[index_min]['OMEGA'], nav[index_min]['deltaN'], nav[index_min]['idot'], \
        nav[index_min]['omega_dot'], nav[index_min]['Cuc'], nav[index_min]['Cus'], \
        nav[index_min]['Crc'], nav[index_min]['Crs'], nav[index_min]['Cic'], \
        nav[index_min]['Cis'], sat_id[0], sat_id[1:], \
        nav[index_min]['clk_bias'], nav[index_min]['clk_drift'], nav[index_min]['clk_drift_rate'])


def sat_id2prn(sat_id):
    if sat_id[0] == 'G':   
        return int(sat_id[1:3])
    elif sat_id[0] == 'R': 
        return int(sat_id[1:3]) + OPTIONS['sats_num_G']
    elif sat_id[0] == 'E': 
        return int(sat_id[1:3]) + OPTIONS['sats_num_G'] + OPTIONS['sats_num_R']
    elif sat_id[0] == 'C': 
        return int(sat_id[1:3]) + OPTIONS['sats_num_G'] + OPTIONS['sats_num_R'] \
            + OPTIONS['sats_num_E']
    else:
        return 0


def prn2sat_id(prn):
    """
    Convert id of satellite to prn
    example: 14->'G14', 34->'R02' 
    """
    if prn < 10:
        return 'G0' + str(prn)
    elif prn < OPTIONS['sats_num_G'] + 1:
        return 'G' + str(prn)
    elif prn < OPTIONS['sats_num_G'] + 10:
        return 'R0' + str(prn - OPTIONS['sats_num_G'])
    elif prn < OPTIONS['sats_num_G'] + OPTIONS['sats_num_R'] + 1:
        return 'R' + str(prn - OPTIONS['sats_num_G'])
    elif prn < OPTIONS['sats_num_G'] + OPTIONS['sats_num_R'] + 10:
        return 'E0' + str(prn - OPTIONS['sats_num_G'] - OPTIONS['sats_num_R'])
    elif prn < OPTIONS['sats_num_G'] + OPTIONS['sats_num_R'] + OPTIONS['sats_num_E'] + 1:
        return 'E' + str(prn - OPTIONS['sats_num_G'] - OPTIONS['sats_num_R'])
    elif prn < OPTIONS['sats_num_G'] + OPTIONS['sats_num_R'] + OPTIONS['sats_num_E'] + 10:
        return 'C0' + str(prn - OPTIONS['sats_num_G'] \
            - OPTIONS['sats_num_R'] - OPTIONS['sats_num_E'])
    elif prn < OPTIONS['sats_num_G'] + OPTIONS['sats_num_R'] + OPTIONS['sats_num_E'] + OPTIONS['sats_num_C'] + 1:
        return 'C' + str(prn - OPTIONS['sats_num_G'] \
            - OPTIONS['sats_num_R'] - OPTIONS['sats_num_E'])
    else:
        return '000'


def lsq(H, W, v):
    """
    least square 
    """
    HTW = np.matmul(np.transpose(H), W)
    a = np.matmul(HTW, H) # H^T * W * H
    b = np.matmul(HTW, v) # H^T * W * v
    X = np.linalg.lstsq(a, b, rcond=None) # a\b
    # X = np.linalg.solve(a, b) # a\b
    # return X
    return X[0]


def init_cal(obs):
    obs_sys = []
    for sat_info in obs['obs']:
        if sat_info['sys'] not in obs_sys:
            obs_sys.append(sat_info['sys'])
    H = []
    W = []
    v = []
    if 'G' not in obs_sys:
        H.append([0, 0, 0, 1, 0, 0, 0])
        W.append(1)
        v.append(0)
    if 'R' not in obs_sys:
        H.append([0, 0, 0, 0, 1, 0, 0])
        W.append(1)
        v.append(0)
    if 'E' not in obs_sys:
        H.append([0, 0, 0, 0, 0, 1, 0])
        W.append(1)
        v.append(0)
    if 'C' not in obs_sys:
        H.append([0, 0, 0, 0, 0, 0, 1])
        W.append(1)
        v.append(0)
    return H, W, v
    # return [], [], []


def spp_epoch(obs, nav, pos_r, time_sys):
    """
    spp for single epoch
    """
    iteration = OPTIONS['spp_iteration']
    min_ele = OPTIONS['min_ele'] # degree
    X = [0, 0, 0, 0]
    pos_sat = []
    clk_sat = []
    dtr = [0, 0, 0, 0]
    # step 1. calculate position of observed satellites
    for sat_info in obs['obs']:
        temp_sat = cal_sat_pos(utc2gpst(obs['time'], time_sys=='UTC'), \
            sat_info['sat_id'], \
            sat_info['C'][0], nav['body'])
        pos_sat.append(temp_sat[0])
        clk_sat.append(temp_sat[1])
    
    H_temp, W_temp, V_temp = init_cal(obs)
    for _ in range(iteration):
        H = copy.deepcopy(H_temp)
        W = copy.deepcopy(W_temp)
        v = copy.deepcopy(V_temp)
        for idx_sat, sat_info in enumerate(obs['obs']):
            pos_s = pos_sat[idx_sat]
            clk_s = clk_sat[idx_sat]
            if cal_distance(pos_s) < OPTIONS['semi_major_axis']:
                continue
            # step 2. calculate azimuth and elevation of observed satellite
            azi = cal_azi(pos_r, pos_s)
            ele = cal_ele(pos_r, pos_s)
            if ele < min_ele * np.pi / 180 or (sat_info['C'][0]) < 1:
                continue
            else:
                dis = cal_distance(pos_s, pos_r)
                # H.append(np.append((np.array(pos_r) - np.array(pos_s)) / dis, 1))
                if sat_info['sys'] == 'G':
                    H.append(np.append((np.array(pos_r) - np.array(pos_s)) / dis, [1, 0, 0, 0]))
                elif sat_info['sys'] == 'R':
                    H.append(np.append((np.array(pos_r) - np.array(pos_s)) / dis, [0, 1, 0, 0]))
                elif sat_info['sys'] == 'E':
                    H.append(np.append((np.array(pos_r) - np.array(pos_s)) / dis, [0, 0, 1, 0]))
                elif sat_info['sys'] == 'C':
                    H.append(np.append((np.array(pos_r) - np.array(pos_s)) / dis, [0, 0, 0, 1]))
                if 'ion_alpha_'+sat_info['sys'] in nav:
                    ion_alpha = nav['ion_alpha_'+sat_info['sys']]
                else:
                    ion_alpha = None
                if 'ion_beta_'+sat_info['sys'] in nav:
                    ion_beta = nav['ion_beta_'+sat_info['sys']]
                else:
                    ion_beta = None
                v.append(res_code(sat_info['C'][0], pos_r, pos_s, dtr, \
                    utc2gpst(obs['time'], time_sys=='UTC'), azi, ele, \
                    ion_alpha, ion_beta, \
                    (sat_info['sat_id']), nav['body'], clk_s))
                W.append((np.sin(ele))**2)
        if len(W) > 4:
            X = lsq(H, np.diag(W), v)
        else:
            print(f'lack of satellite, sats_num: {len(W)}')
            return pos_r
        pos_r += X[0:3]
        dtr += X[3:]
        if np.abs(X[0])<0.001 and np.abs(X[1])<0.001 and np.abs(X[2])<0.001:
            break
    return pos_r


def spp(obs, nav):
    """
    spp
    """
    res = []
    if 'approx_pos' in obs:
        pos_r = obs['approx_pos']
    else:
        pos_r = [1e2, 1e2, 1e2]

    epoch_num = len(obs['body'])
    for i in range(epoch_num):
        temp = spp_epoch(obs['body'][i], nav, pos_r, obs['time_sys'])
        print(f'epoch: {i}/{epoch_num}, {temp}')
        res.append(temp)
    with open('spp.pkl', 'wb') as f:  
        pickle.dump(res, f)
    f.close()
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

nav1 = read_rnx(os.path.join(dirname, "data/brdc1880.21n"))
nav2 = read_rnx(os.path.join(dirname, "data/CKSV00TWN_S_20211880000_01D_EN.rnx"))
nav4 = read_rnx(os.path.join(dirname, "data/CKSV00TWN_S_20211880000_01D_CN.rnx"))
nav_ls = []
nav_ls.append(nav1)
nav_ls.append(nav2)
# nav_ls.append(nav3)
nav_ls.append(nav4)
nav = merge_nav(nav_ls)

obs1 = read_rnx(os.path.join(dirname, "data/CKSV00TWN_S_20211880000_01D_30S_MO.rnx")) 
# nav3 = read_rnx(os.path.join(dirname, "data/brdc3550.15n"))
# obs1 = read_rnx(os.path.join(dirname, "data/iqal3550.15o")) 

pos_cksv = [-2.95661939424981e+06, 5.07590217189893e+06, 2.47662550044099e+06]
pos_iqal = [1.03600086899334e+06, -2.63145559079701e+06, 5.69781974243222e+06]
res = spp(obs1, nav)
evaluate(res, pos_cksv)



# dirname = os.path.dirname(__file__)
# with open('spp_new.pkl', 'rb') as f:  
#     res_new = pickle.load(f)
# f.close()
# with open('spp.pkl', 'rb') as f:  
#     res_old = pickle.load(f)
# f.close()
# com = []
# for i in range(len(res_old)):
#     com.append(res_old[i] - res_new[i])
# evaluate(com, [0, 0, 0])



# dirname = os.path.dirname(__file__)


# # nav = read_rnx(os.path.join(dirname, "data/HKSL00HKG_R_20171230000_01D_30S_CN.rnx"))
# nav = read_rnx(os.path.join(dirname, "data/brdc1230.17n"))
# obs1 = read_rnx(os.path.join(dirname, "data/HKSL00HKG_R_20171230000_01D_30S_MO.rnx")) 
# # nav3 = read_rnx(os.path.join(dirname, "data/brdc3550.15n"))
# # obs1 = read_rnx(os.path.join(dirname, "data/iqal3550.15o")) 

# pos_cksv = [-2.95661939424981e+06, 5.07590217189893e+06, 2.47662550044099e+06]
# pos_iqal = [1.03600086899334e+06, -2.63145559079701e+06, 5.69781974243222e+06]
# res = spp(obs1, nav)
# evaluate(res, pos_cksv)
