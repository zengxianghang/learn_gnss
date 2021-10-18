from datetime import datetime

def str2time(time_str):
    """
    time in string format to datetime format
    """
    year   = int(time_str[ 0: 4])
    month  = int(time_str[ 5: 7])
    day    = int(time_str[ 8:10])
    hour   = int(time_str[11:13])
    minute = int(time_str[14:16])
    second = int(time_str[17:19])
    ms     = int(time_str[20:28])
    return datetime(year, month, day, hour, minute, second, ms)


def str2int(line):
    if line.strip() == '':
        return 999999
    else:
        return int(line)


def str2sat_status(line):
    """
    """
    line = '{:<80}'.format(line)
    sat_status = {}
    sat_status['sat_id'] = line[1:4]
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
                        if line[i:i+3] != '  0':
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


def read_sp3(filepath):
    """
    read sp3 file
    filepath: full path of sp3 file

    return dict of sp3 header and body
    """
    header = read_sp3_header(filename)
    body = {'body': read_sp3_body(filename, header['record_type'], header['sats_num'])}

    return {**header, **body}

filename = '/Users/zengxianghang/Downloads/work_hx/learning_rtklib-master/data/sp3/igr21194.sp3'
h = read_sp3(filename)
c = 1