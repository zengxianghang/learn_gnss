import os

from constants import OPTIONS
from datetime import datetime


#------------------------------------------------------------------------------#
# exclude: an easy way to do, but slow
def exclude_obs(obs):
    for i in range(len(obs)):
        exclude_index = []
        for j in range(obs[i]['sats_num']):
            if obs[i]['obs'][j]['sys'] not in OPTIONS['enable_sys']:
                exclude_index.append(j)
        for j in range(len(exclude_index)-1, -1, -1):
            del obs[i]['obs'][exclude_index[j]]
        obs[i]['sats_num'] -= len(exclude_index)


#------------------------------------
#             read sp3              #
def str2time(time_str):
    """
    Convert time in string format to datetime format
    """
    time_ls = time_str.split()
    year   = int(time_ls[0])
    month  = int(time_ls[1])
    day    = int(time_ls[2])
    hour   = int(time_ls[3])
    minute = int(time_ls[4])
    second = (float(time_ls[5]))
    
    if year<100:
        if year<80:
            year += 2000
        else:
            year += 1900
    time_str = f'{year} {month} {day} {hour} {minute} {second}'
    return datetime.strptime(time_str, '%Y %m %d %H %M %S.%f')
    

def str2int(line, num=999999):
    """
    Convert string to int
    """
    if line.strip() == '':
        return num
    else:
        return int(line)


def str2float(line, num=0.0):
    if line.strip() == '':
        return num
    else:
        return float(line.replace("D", "E"))


def str2sat_status(line):
    """
    Convert sp3 epoch in string format to dict
    """
    line = '{:<80}'.format(line)
    sat_status = {}
    sat_status['sat_id'] = line[1:4]
    sys = line[1]
    if sys in OPTIONS['enable_sys']:
        sat_status['x'] = float(line[4:18])
        sat_status['y'] = float(line[18:32])
        sat_status['z'] = float(line[32:46])
        sat_status['clk'] = float(line[46:60])
        sat_status['x_sdev'] = str2int(line[61:63])
        sat_status['y_sdev'] = str2int(line[64:66])
        sat_status['z_sdev'] = str2int(line[67:69])
        sat_status['clk_sdev'] = str2int(line[70:73])
        sat_status['clk_event_flag'] = line[74:75]
        sat_status['clk_pred_flag'] = line[75:76]
        sat_status['maneuver_flag'] = line[78:79]
        sat_status['orbit_pred_flag'] = line[79:80]
    return sat_status


def str2sat_pos_sdev(line):
    sat_status = {}
    sat_status['x_sdev_new'] = int(line[4:8])
    sat_status['y_sdev_new'] = int(line[9:13])
    sat_status['z_sdev_new'] = int(line[14:18])
    sat_status['clk_sdev_new'] = int(line[19:26])
    sat_status['xy_correlation'] = int(line[27:35])
    sat_status['xz_correlation'] = int(line[36:44])
    sat_status['xc_correlation'] = int(line[45:53])
    sat_status['yz_correlation'] = int(line[54:62])
    sat_status['yc_correlation'] = int(line[63:71])
    sat_status['zc_correlation'] = int(line[72:80])

    return sat_status


def str2sat_vel(line):
    sat_status = {}
    sat_status['x_vel'] = float(line[4:18])
    sat_status['y_vel'] = float(line[18:32])
    sat_status['z_vel'] = float(line[32:46])
    sat_status['clk_rate_chg'] = float(line[46:60])
    sat_status['x_vel_sdev'] = int(line[61:63])
    sat_status['y_vel_sdev'] = int(line[64:66])
    sat_status['z_vel_sdev'] = int(line[67:69])
    sat_status['clk_rate_sdev'] = int(line[70:73])
    return sat_status


def str2sat_vel_sdev(line):
    sat_status = {}
    sat_status['x_vel_sdev_new'] = int(line[4:8])
    sat_status['y_vel_sdev_new'] = int(line[9:13])
    sat_status['z_vel_sdev_new'] = int(line[14:18])
    sat_status['clk_rate_sdev_new'] = int(line[19:26])
    sat_status['xy_vel_correlation'] = int(line[27:35])
    sat_status['xz_vel_correlation'] = int(line[36:44])
    sat_status['xc_vel_correlation'] = int(line[45:53])
    sat_status['yz_vel_correlation'] = int(line[54:62])
    sat_status['yc_vel_correlation'] = int(line[63:71])
    sat_status['zc_vel_correlation'] = int(line[72:80])

    return sat_status


def read_sp3_header(filename):
    """
    read header of sp3 file
    """
    header = {}
    line_num = 1
    sat_id = []
    orbit_accuracy = []

    with open(filename, 'r') as file:
        for line in file:
            if line_num == 1:
                header['version'] = line[1]
                header['record_type'] = line[2]
                header['time'] = str2time(line[3:31])
                header['epochs_num'] = int(line[32:39])
                header['coordinate_sys'] = line[46:51]
            if line[0:1] == '+':
                if line[1:2] =='+':
                    for i in range(9, 60, 3):
                        if len(orbit_accuracy) < len(sat_id):
                            orbit_accuracy.append(int(line[i:i+3]))
                        else:
                            break
                else:
                    if 'sats_num' not in header:
                        header['sats_num'] = int(line[3:6])
                    for i in range(9, 60, 3):
                        if line[i:i+3] != '  0' and line[i:i+3] != ' 00':
                            sat_id.append(line[i:i+3])
            if line[0:2] == '%c':
                if 'time_sys' not in header:
                    header['time_sys'] = line[9:12]
            if line[0:2] == '%f':
                if 'base_pos' not in header:
                    header['base_pos'] = float(line[ 3:13])
                    header['base_clk'] = float(line[14:26])
            line_num += 1
    header['sat_id'] = sat_id
    header['orbit_accuracy'] = orbit_accuracy

    file.close()
    return header


def read_sp3_body(filename, record_type, sats_num):
    """
    read body of sp3 file
    """
    body = []
    with open(filename, 'r') as file:
        while (line := file.readline()):
            if line[0:3] == 'EOF': break
            ephemerics = {}
            if line[0:1] == '*': # begin of body
                ephemerics['time'] = str2time(line[3:31])
                if record_type == 'P':
                    sat = []
                    for _ in range(sats_num):
                        line = (file.readline()).rstrip()
                        sat_status = str2sat_status(line)
                        sat.append(sat_status)   
                else:
                    sat = []
                    for _ in range(sats_num):
                        sat = {}
                        sat_status = str2sat_status(line)
                        if sat_status['sat_id'][0] in OPTIONS['enable_sys']:
                            line = file.readline()
                            sat_status = {**sat_status, **str2sat_pos_sdev(line)}
                            line = file.readline()
                            sat_status = {**sat_status, **str2sat_vel(line)}
                            line = file.readline()
                            sat_status = {**sat_status, **str2sat_vel_sdev(line)}
                            sat.append(sat_status)
                ephemerics['sat'] = sat
                body.append(ephemerics)

    file.close()
    return body


def read_sp3(filename):
    """
    read sp3 file
    filepath: full path of sp3 file

    return dict of sp3 header and body
    """
    header = read_sp3_header(filename)
    body = {'body': read_sp3_body(filename, header['record_type'], header['sats_num'])}

    return {**header, **body}


#---------------------------------
#           read rinex           #
def read_rnx_version(filename):
    """
    read version of rinex file
    """
    with open(filename, 'r') as file:
        line = file.readline()
        if line[40:41] == ' ':
            sys = 'G'
        else:
            sys = line[40:41]
    file.close()
    return float(line[0:9]), line[20:21], sys


def read_rnx_header_v210_obs(filename):
    # rinex 2.10 GPS OBSERVATION DATA FILE - HEADER SECTION
    header = {}
    obs_type = []
    with open(filename, 'r') as file:
        for line in file:
            if 'RINEX VERSION / TYPE' in line: 
                header['version'] = float(line[0:20])
                header['file_type'] = line[20:40].rstrip()
                header['sat_sys'] = line[40:60].rstrip()
            elif 'PGM / RUN BY / DATE' in line:
                header['pgm'] = line[0:20].rstrip()
                header['agency'] = line[21:40].rstrip()
                header['date'] = line[40:60].rstrip()
            elif 'COMMENT' in line: # optional
                continue
            elif 'MARKER NAME' in line:
                header['antenna_marker_name'] = line[0:60].rstrip()
            elif 'MARKER NUMBER' in line: # optional
                header['antenna_marker_num'] = line[0:20].rstrip()
            elif 'OBSERVER / AGENCY' in line:
                header['obs_name'] = line[0:20].rstrip()
                header['obs_agency'] = line[20:40].rstrip()
            elif 'REC # / TYPE / VERS' in line:
                header['receiver_num'] = line[0:20].rstrip()
                header['receiver_type'] = line[20:40].rstrip()
                header['receiver_vers'] = line[40:60].rstrip()
            elif 'ANT # / TYPE' in line:
                header['antenna_num'] = line[0:20].rstrip()
                header['antenna_type'] = line[20:40].rstrip()
            elif 'APPROX POSITION XYZ' in line:
                header['approx_pos'] = [float(line[0:14]), float(line[14:28]), float(line[28:42])]
            elif 'ANTENNA: DELTA H/E/N' in line:
                header['antenna_delta'] = [float(line[0:14]), float(line[14:28]), float(line[28:42])]
            elif 'WAVELENGTH FACT L1/2' in line: # optional
                if 'WAVELENGTH FACT L1/2' in header: # optional
                    pass # todo
                else:
                    header['wavelength_factors'] = [int(line[0:6]), int(line[6:12])]
            elif '# / TYPES OF OBSERV' in line:
                if 'obs_type_num' not in header:
                    header['obs_type_num'] = int(line[0:6])
                for i in range(6, 60, 6):
                    if line[i+4:i+6] != '  ':
                        obs_type.append(line[i+4:i+6])
            elif 'INTERVAL' in line: # optional
                header['interval'] = float(line[0:10])
            elif 'TIME OF FIRST OBS' in line:
                header['time_first_obs'] = str2time(line[0:43])
                header['time_sys'] = line[48:51].strip()
                if header['time_sys'] == '' and header['sat_sys'] == 'GPS':
                    header['time_sys'] = 'GPS'
            elif 'TIME OF LAST OBS' in line: # optional
                header['time_last_obs'] = str2time(line[0:43])
                header['time_sys'] = line[48:51].strip()
                if header['time_sys'] == '' and header['sat_sys'] == 'GPS':
                    header['time_sys'] = 'GPS'
            elif 'RCV CLOCK OFFS APPL' in line: # optional
                header['rcv_clk_off'] = int(line[0:6])
            elif 'LEAP SECONDS' in line: # optional
                header['leap'] = int(line[0:6])
            elif '# OF SATELLITES' in line: # optional
                header['sats_num'] = int(line[0:6])
            elif 'PRN / # OF OBS' in line: # optional
                pass # todo
            elif 'END OF HEADER' in line:
                break
    header['obs_type'] = obs_type
    file.close()
    return header


def read_rnx_body_v210_obs(filename, obs_type_num, obs_type):
    """
    read rinex 2.10 GPS OBSERVATION DATA FILE - DATA RECORD 
    """
    body = []
    end_header = False
    with open(filename, 'r') as file:
        while (line := file.readline()):
            if 'END OF HEADER' in line:
                end_header = True
                break
        if end_header:
            while (line := file.readline()):
                if 'COMMENT' in line:
                    continue
                epoch = {}
                epoch['flag'] = int(line[28:29])
                if epoch['flag'] != 0:
                    continue
                else:
                    epoch['time'] = str2time(line[0:26])
                    epoch['sats_num'] = int(line[29:32])
                    sat_id = []
                    for i in range(int(epoch['sats_num']/12-1e-5)+1):
                        if i > 0:
                            line = file.readline().rstrip()
                        line = '{:<80}'.format(line)
                        for j in range(32, 68, 3):
                            if line[j:j+3].strip() != '':
                                sat_id.append(line[j:j+3])
                        
                    epoch['sat_id'] = sat_id
                    if line[68:80].strip() != '': # optional
                        epoch['rcv_clk_off'] = float(line[68:80])
                    
                    obs = []
                    for i in range(epoch['sats_num']):
                        index_obs_type = 0
                        obs_d = {}
                        obs_d['sat_id'] = epoch['sat_id'][i]
                        for j in range(int(obs_type_num/5-1e-5)+1):
                            line = file.readline().rstrip()
                            line = '{:<80}'.format(line)
                            for k in range(5):
                                if len(obs_d) < obs_type_num:
                                    obs_d[obs_type[index_obs_type]] \
                                        = [str2float(line[k*16:k*16+14]), \
                                        str2int(line[k*16+14:k*16+15]), \
                                        str2int(line[k*16+15:k*16+16])]
                                    
                                    index_obs_type += 1
                                else:
                                    break
                        obs.append(obs_d)
                    epoch['obs'] = obs  
                    body.append(epoch)
    file.close()
    return body


def read_rnx_v210_obs(filename):
    header = read_rnx_header_v210_obs(filename)
    body = {'body': read_rnx_body_v210_obs(filename, header['obs_type_num'], header['obs_type'])}

    return {**header, **body}


def read_rnx_header_v210_nav(filename):
    """
    work with mixed GNSS navigation message data
    """
    header = {}
    with open(filename, 'r') as file:
        for line in file:
            if 'RINEX VERSION / TYPE' in line: 
                header['version'] = float(line[0:20])
                header['file_type'] = line[20:40].rstrip()
                header['sat_sys'] = line[40:60].rstrip()
            elif 'PGM / RUN BY / DATE' in line:
                header['pgm'] = line[0:20].rstrip()
                header['agency'] = line[21:40].rstrip()
                header['date'] = line[40:60].rstrip()
            elif 'COMMENT' in line: # optional
                continue
            elif 'ION ALPHA' in line: # optional
                header['ion_alpha'] = [str2float(line[2:14]), \
                    str2float(line[14:26]), str2float(line[26:38]), \
                    str2float(line[38:50])]
            elif 'ION BETA' in line: # optional
                header['ion_beta'] = [str2float(line[2:14]), \
                    str2float(line[14:26]), str2float(line[26:38]), \
                    str2float(line[38:50])]
            elif 'CORR TO SYSTEM TIME' in line: # optional, glonass
                header['reference_time'] = datetime( \
                    int(line[0:6]), int(line[6:12]), int(line[12:18]))
                header['correction_scale'] = str2float(line[21:21+19])
            elif 'DELTA-UTC: A0,A1,T,W' in line: # optional
                header['delta_utc'] = [str2float(line[3:22]), \
                    str2float(line[22:22+19]), int(line[22+19:22+28]), \
                    int(line[22+28:22+37])]
            elif 'LEAP SECONDS' in line: # optional
                header['leap'] = int(line[0:6])
            elif 'END OF HEADER' in line:
                break
    file.close()
    return header


def read_rnx_body_v210_nav(filename):
    body = []
    end_header = False
    with open(filename, 'r') as file:
        while (line := file.readline()):
            if 'END OF HEADER' in line:
                end_header = True
                break
        if end_header:
            while (line := file.readline()):
                nav = {}
                nav['prn'] = int(line[0:2])
                nav['time'] = str2time(line[3:22])
                nav['clk_bias'] = str2float(line[22:22+19])
                nav['clk_drift'] = str2float(line[22+19:22+19*2])
                nav['clk_drift_rate'] = str2float(line[22+19*2:22+19*3])

                line = file.readline()
                nav['iode'] = str2float(line[3:3+19])
                nav['Crs'] = str2float(line[22:22+19])
                nav['deltaN'] = str2float(line[22+19:22+19*2])
                nav['M0'] = str2float(line[22+19*2:22+19*3])

                line = file.readline()
                nav['Cuc'] = str2float(line[3:3+19])
                nav['e'] = str2float(line[22:22+19])
                nav['Cus'] = str2float(line[22+19:22+19*2])
                nav['sqrtA'] = str2float(line[22+19*2:22+19*3])
                
                line = file.readline()
                nav['Toe'] = str2float(line[3:3+19])
                nav['Cic'] = str2float(line[22:22+19])
                nav['OMEGA'] = str2float(line[22+19:22+19*2])
                nav['Cis'] = str2float(line[22+19*2:22+19*3])
                
                line = file.readline()
                nav['i0'] = str2float(line[3:3+19])
                nav['Crc'] = str2float(line[22:22+19])
                nav['omega'] = str2float(line[22+19:22+19*2])
                nav['omega_dot'] = str2float(line[22+19*2:22+19*3])
                
                line = file.readline()
                nav['idot'] = str2float(line[3:3+19])
                nav['codes_L2'] = str2float(line[22:22+19])
                nav['week'] = str2float(line[22+19:22+19*2])
                nav['L2_P_flag'] = str2float(line[22+19*2:22+19*3])
                
                line = file.readline()
                nav['sv_accuracy'] = str2float(line[3:3+19])
                nav['sv_health'] = str2float(line[22:22+19])
                nav['TGD'] = str2float(line[22+19:22+19*2])
                nav['iodc'] = str2float(line[22+19*2:22+19*3])
                
                line = file.readline()
                nav['transmission_time'] = str2float(line[3:3+19])
                if len(line) > 23:
                    nav['interval'] = str2float(line[22:22+19])
                else:
                    nav['interval'] = 0

                body.append(nav)
    file.close()
    return body


def read_rnx_body_v210_nav_glo(filename):
    body = []
    end_header = False
    with open(filename, 'r') as file:
        while (line := file.readline()):
            if 'END OF HEADER' in line:
                end_header = True
                break
        if end_header:
            while (line := file.readline()):
                nav = {}
                nav['prn'] = int(line[0:2])
                nav['time'] = str2time(line[3:22])
                nav['clk_bias'] = str2float(line[22:22+19])
                nav['relative_frequency_bias'] = str2float(line[22+19:22+19*2])
                nav['message_frame_time'] = str2float(line[22+19*2:22+19*3])

                line = file.readline()
                nav['pos_x'] = str2float(line[3:3+19])
                nav['vel_x'] = str2float(line[22:22+19])
                nav['acc_x'] = str2float(line[22+19:22+19*2])
                nav['health'] = str2float(line[22+19*2:22+19*3])

                line = file.readline()
                nav['pos_y'] = str2float(line[3:3+19])
                nav['vel_y'] = str2float(line[22:22+19])
                nav['acc_y'] = str2float(line[22+19:22+19*2])
                nav['frequency_num'] = str2float(line[22+19*2:22+19*3])
                
                line = file.readline()
                nav['pos_z'] = str2float(line[3:3+19])
                nav['vel_z'] = str2float(line[22:22+19])
                nav['acc_z'] = str2float(line[22+19:22+19*2])
                nav['age_oper'] = str2float(line[22+19*2:22+19*3])

                body.append(nav)
    file.close()
    return body


def read_rnx_v210_nav(filename):
    header = read_rnx_header_v210_nav(filename)
    if header['file_type'][0] == 'G':
        body = {'body': read_rnx_body_v210_nav_glo(filename)}
    else:
        body = {'body': read_rnx_body_v210_nav(filename)}

    return {**header, **body}


def read_rnx_header_v303_obs(filename):
    # rinex 3.03 GPS OBSERVATION DATA FILE - HEADER SECTION
    header = {}
    obs_type = []
    phase_shift = []
    with open(filename, 'r') as file:
        for line in file:
            if 'RINEX VERSION / TYPE' in line: 
                header['version'] = float(line[0:20])
                header['file_type'] = line[20:40].rstrip()
                header['sat_sys'] = line[40:60].rstrip()
            elif 'PGM / RUN BY / DATE' in line:
                header['pgm'] = line[0:20].rstrip()
                header['agency'] = line[21:40].rstrip()
                header['date'] = line[40:60].rstrip()
            elif 'COMMENT' in line: # optional
                continue
            elif 'MARKER NAME' in line:
                header['antenna_marker_name'] = line[0:60].rstrip()
            elif 'MARKER NUMBER' in line: # optional
                header['antenna_marker_num'] = line[0:20].rstrip()
            elif 'MARKER TYPE' in line:
                header['marker_type'] = line[0:20].rstrip()
            elif 'OBSERVER / AGENCY' in line:
                header['obs_name'] = line[0:20].rstrip()
                header['obs_agency'] = line[20:40].rstrip()
            elif 'REC # / TYPE / VERS' in line:
                header['receiver_num'] = line[0:20].rstrip()
                header['receiver_type'] = line[20:40].rstrip()
                header['receiver_vers'] = line[40:60].rstrip()
            elif 'ANT # / TYPE' in line:
                header['antenna_num'] = line[0:20].rstrip()
                header['antenna_type'] = line[20:40].rstrip()
            elif 'APPROX POSITION XYZ' in line:
                header['approx_pos'] = [float(line[0:14]), float(line[14:28]), float(line[28:42])]
            elif 'ANTENNA: DELTA H/E/N' in line:
                header['antenna_delta_hen'] = [float(line[0:14]), float(line[14:28]), float(line[28:42])]
            elif 'ANTENNA: DELTA X/Y/Z' in line: # optional
                header['antenna_delta_xyz'] = [float(line[0:14]), float(line[14:28]), float(line[28:42])]
            elif 'ANTENNA:PHASECENTER' in line: # optional
                pass # todo
            elif 'ANTENNA: B.SIGHT XYZ' in line: # optional
                header['b_sight'] = [float(line[0:14]), float(line[14:28]), float(line[28:42])]
            elif 'ANTENNA: ZERODIR AZI' in line: # optional
                header['azi_zero_dir'] = float(line[0:14])
            elif 'ANTENNA: ZERODIR XYZ' in line: # optional
                header['zero_dir'] = [float(line[0:14]), float(line[14:28]), float(line[28:42])]
            elif 'CENTER OF MASS: XYZ' in line: # optional
                header['mass_center'] = [float(line[0:14]), float(line[14:28]), float(line[28:42])]
            elif 'SYS / # / OBS TYPES' in line:
                sys = line[0:1]
                type_num = int(line[3:6])
                sys_type = []
                for i in range(int(type_num/13-1e-5)+1):
                    for j in range(7, 59, 4):
                        if line[j:j+3] != '   ':
                            sys_type.append(line[j:j+3])
                        else:
                            break
                    if i < int(type_num/13-1e-5-1e-5):
                        line = file.readline()
                obs_type.append({sys: sys_type})
            elif 'SIGNAL STRENGTH UNIT' in line: # optional
                header['signal_strength'] = line[0:20]
            elif 'INTERVAL' in line: # optional
                header['interval'] = float(line[0:10])
            elif 'TIME OF FIRST OBS' in line:
                header['time_first_obs'] = str2time(line[0:43])
                header['time_sys'] = line[48:51].strip()
                if header['time_sys'] == '' and header['sat_sys'] == 'GPS':
                    header['time_sys'] = 'GPS'
            elif 'TIME OF LAST OBS' in line: # optional
                header['time_last_obs'] = str2time(line[0:43])
                header['time_sys'] = line[48:51].strip()
                if header['time_sys'] == '' and header['sat_sys'] == 'GPS':
                    header['time_sys'] = 'GPS'
            elif 'RCV CLOCK OFFS APPL' in line: # optional
                header['rcv_clk_off'] = int(line[0:6])
            elif 'SYS / DCBS APPLIED' in line: # optional
                pass # todo
            elif 'SYS / PCVS APPLIED' in line: # optional
                pass # todo
            elif 'SYS / SCALE FACTOR' in line: # optional
                pass # todo
            elif 'SYS / PHASE SHIFT' in line:
                sys = line[0:1]
                phase_type = line[2:5]
                correction = float(line[6:14])
                involved_sats_num = str2int(line[16:18])
                # todo 
            elif 'GLONASS SLOT / FRQ #' in line:
                pass # todo
            elif 'GLONASS COD/PHS/BIS' in line:
                pass # todo
            elif 'LEAP SECONDS' in line: # optional
                header['leap'] = int(line[0:6])
            elif '# OF SATELLITES' in line: # optional
                header['sats_num'] = int(line[0:6])
            elif 'PRN / # OF OBS' in line: # optional
                pass # todo
            elif 'END OF HEADER' in line:
                break
    header['phase_shift'] = phase_shift
    header['obs_type'] = obs_type
    file.close()
    return header


def read_rnx_body_v303_obs(filename, obs_type):
    body = []
    end_header = False
    with open(filename, 'r') as file:
        while (line := file.readline()):
            if 'END OF HEADER' in line:
                end_header = True
                break
        if end_header:
            while (line := file.readline()):
                if 'COMMENT' in line or line.strip() == '':
                    continue
                epoch = {}
                epoch['flag'] = int(line[31:32])
                if epoch['flag'] != 0:
                    continue
                else:
                    epoch['time'] = str2time(line[2:29])
                    epoch['sats_num'] = int(line[32:35])
                    epoch['clk_off'] = str2float(line[41:41+15], 999999.99)
                    obs = []
                    for _ in range(epoch['sats_num']):
                        obs_d = {}
                        line = file.readline()
                        obs_d['sys'] = line[0:1]
                        obs_d['sat_id'] = line[0:3]
                        index_sys = 0
                        # next((index_sys for index_sys, item in enumerate(obs_type) if list(item)[0] == line[0:1]), None)
                        for index_sys, item in enumerate(obs_type):
                            if list(item)[0] == line[0:1]:
                                break
                        obs_type_temp = obs_type[index_sys][obs_d['sys']]
                        for k in range(len(obs_type_temp)):
                            obs_d[obs_type_temp[k]] \
                                = [str2float(line[3+k*16:3+k*16+14]), \
                                   str2int(line[3+k*16+14:3+k*16+15]), \
                                   str2int(line[3+k*16+15:3+k*16+16])]
                        obs.append(obs_d)
                    epoch['obs'] = obs
                body.append(epoch)
    file.close()
    return body


def read_rnx_v303_obs(filename):
    header = read_rnx_header_v303_obs(filename)
    body = {'body': read_rnx_body_v303_obs(filename, header['obs_type'])}

    return {**header, **body}


def read_rnx_header_v303_nav(filename):
    header = {}
    with open(filename, 'r') as file:
        for line in file:
            if 'RINEX VERSION / TYPE' in line: 
                header['version'] = float(line[0:20])
                header['file_type'] = line[20:40].rstrip()
                header['sat_sys'] = line[40:60].rstrip()
            elif 'PGM / RUN BY / DATE' in line:
                header['pgm'] = line[0:20].rstrip()
                header['agency'] = line[21:40].rstrip()
                header['date'] = line[40:60].rstrip()
            elif 'COMMENT' in line: # optional
                continue
            elif 'IONOSPHERIC CORR' in line: # optional
                if 'GPSA' in line:
                    header['gps_ion_alpha'] = [str2float(line[2:14]), \
                        str2float(line[14:26]), str2float(line[26:38]), \
                        str2float(line[38:50])]
                elif 'GPSB' in line:
                    header['gps_ion_beta'] = [str2float(line[2:14]), \
                        str2float(line[14:26]), str2float(line[26:38]), \
                        str2float(line[38:50])]
                elif 'BDSA' in line:
                    header['bds_ion_alpha'] = [str2float(line[2:14]), \
                        str2float(line[14:26]), str2float(line[26:38]), \
                        str2float(line[38:50])]
                elif 'BDSB' in line:
                    header['bds_ion_beta'] = [str2float(line[2:14]), \
                        str2float(line[14:26]), str2float(line[26:38]), \
                        str2float(line[38:50])]
                elif 'QZSA' in line:
                    header['qzs_ion_alpha'] = [str2float(line[2:14]), \
                        str2float(line[14:26]), str2float(line[26:38]), \
                        str2float(line[38:50])]
                elif 'QZSB' in line:
                    header['qzs_ion_beta'] = [str2float(line[2:14]), \
                        str2float(line[14:26]), str2float(line[26:38]), \
                        str2float(line[38:50])]
                elif 'IRNA' in line:
                    header['irn_ion_alpha'] = [str2float(line[2:14]), \
                        str2float(line[14:26]), str2float(line[26:38]), \
                        str2float(line[38:50])]
                elif 'IRNB' in line:
                    header['irn_ion_beta'] = [str2float(line[2:14]), \
                        str2float(line[14:26]), str2float(line[26:38]), \
                        str2float(line[38:50])]
                elif 'GAL' in line:
                    header['gal_ion_beta'] = [str2float(line[2:14]), \
                        str2float(line[14:26]), str2float(line[26:38])]
            elif 'TIME SYSTEM CORR' in line: # optional
                header['delta_utc'] = [str2float(line[3:22]), \
                    str2float(line[22:22+19]), int(line[22+19:22+28]), \
                    int(line[22+28:22+37])]
            elif 'LEAP SECONDS' in line: # optional
                header['leap'] = int(line[0:6])
            elif 'END OF HEADER' in line:
                break
    file.close()
    return header


def read_rnx_body_v303_nav(filename):
    body = []
    end_header = False
    with open(filename, 'r') as file:
        while (line := file.readline()):
            if 'END OF HEADER' in line:
                end_header = True
                break
        if end_header:
            while (line := file.readline()):
                if line[0] in OPTIONS['enable_sys']:
                    nav = {}
                    nav['sat_id'] = (line[0:3])
                    nav['time'] = str2time(line[4:23])
                    nav['clk_bias'] = str2float(line[23:23+19])
                    nav['clk_drift'] = str2float(line[23+19:23+19*2])
                    nav['clk_drift_rate'] = str2float(line[23+19*2:23+19*3])

                    line = file.readline()
                    nav['iode'] = str2float(line[4:23])
                    nav['Crs'] = str2float(line[23:23+19])
                    nav['deltaN'] = str2float(line[23+19:23+19*2])
                    nav['M0'] = str2float(line[23+19*2:23+19*3])

                    line = file.readline()
                    nav['Cuc'] = str2float(line[4:23])
                    nav['e'] = str2float(line[23:23+19])
                    nav['Cus'] = str2float(line[23+19:23+19*2])
                    nav['sqrtA'] = str2float(line[23+19*2:23+19*3])
                    
                    line = file.readline()
                    nav['Toe'] = str2float(line[4:23])
                    nav['Cic'] = str2float(line[23:23+19])
                    nav['OMEGA'] = str2float(line[23+19:23+19*2])
                    nav['Cis'] = str2float(line[23+19*2:23+19*3])
                    
                    line = file.readline()
                    nav['i0'] = str2float(line[4:23])
                    nav['Crc'] = str2float(line[23:23+19])
                    nav['omega'] = str2float(line[23+19:23+19*2])
                    nav['omega_dot'] = str2float(line[23+19*2:23+19*3])
                    
                    line = file.readline()
                    nav['idot'] = str2float(line[4:23])
                    nav['codes_L2'] = str2float(line[23:23+19])
                    nav['week'] = str2float(line[23+19:23+19*2])
                    nav['L2_P_flag'] = str2float(line[23+19*2:23+19*3])
                    
                    line = file.readline()
                    nav['sv_accuracy'] = str2float(line[4:23])
                    nav['sv_health'] = str2float(line[23:23+19])
                    nav['TGD'] = str2float(line[23+19:23+19*2])
                    nav['iodc'] = str2float(line[23+19*2:23+19*3])
                    
                    line = file.readline()
                    nav['transmission_time'] = str2float(line[4:23])
                    if len(line) > 23:
                        nav['interval'] = str2float(line[23:23+19])
                    else:
                        nav['interval'] = 0

                    body.append(nav)
                else:
                    for _ in range(7):
                        line = file.readline()
    file.close()
    return body


def read_rnx_v303_nav(filename):
    header = read_rnx_header_v303_nav(filename)
    body = {'body': read_rnx_body_v303_nav(filename)}

    return {**header, **body}


def read_rnx(filename):
    rnx_vers, data_type, sys = read_rnx_version(filename)
    if rnx_vers < 3:
        if data_type == 'O':
            rnx = read_rnx_v210_obs(filename)
        else:
            rnx = read_rnx_v210_nav(filename)
    else:
        if data_type == 'O':
            rnx = read_rnx_v303_obs(filename)
        else:
            rnx = read_rnx_v303_nav(filename)
            
    return rnx


# dirname = os.path.dirname(__file__)
# # # # test read sp3 file
# # # nav1 = read_sp3(os.path.join(dirname, "data/G.sp3"))
# # # nav2 = read_sp3(os.path.join(dirname, "data/CERG.sp3"))

# # # # test read rinex obs file
# # obs1 = read_rnx_v210_obs(os.path.join(dirname, "data/210_obs_gps.05o"))
# # # obs2 = read_rnx_v210_obs(os.path.join(dirname, "data/210_obs_mix.11o"))
# # nav3 = read_rnx_v210_nav(os.path.join(dirname, "data/210_nav_gps.05n"))
# # obs3 = read_rnx_v303_obs(os.path.join(dirname, "data/303_obs_mix.00o"))
# # exclude_obs(obs3['body'])
# # nav3 = read_rnx_v210_nav(os.path.join(dirname, "data/210_nav_glo.20g"))
# nav3 = read_rnx_v303_nav(os.path.join(dirname, "data/304_nav_mix.rnx"))
# c = 1