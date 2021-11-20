
OPTIONS = { 'speed_light': 2.99792458e8, \
            'gravitational_constant': 3.9860050e14, \
            'gravitational_constant_G': 3.9860050e14, \
            'gravitational_constant_R': 398600.44e14, \
            'gravitational_constant_C': 3.986004418e14, \
            'gravitational_constant_atmosphere_R': 0.35e9, \
            'semi_major_axis': 6378137,\
            'semi_major_axis_C': 6378136.55,\
            'semi_major_axis_R': 6378136, \
            'eccentricity': 8.1819190842622e-2, \
            'rotation_rate': 7.2921151467e-5, \
            'rotation_rate_G': 7.2921151467e-5, \
            'rotation_rate_R': 7.292115e-5, \
            'rotation_rate_C': 7.292115e-5, \
            'flattening_R': 1 / 298.257839303, \
            'flattening_C': 1 / 298.257222101, \
            'gravity_equator_R': 9.780328, \
            'corr_gravity_R': -0.9e-5, \
            'J20_R': 1082625.7e-9, \
            'J40_R': -2370.9e-9, \
            'normal_potential_R': 62636861.074, \
            # code priority, G: GPS, R: GLO, E: GAL, C: BDS, num: band/frequency
            'code_priority_G1': 'CPYWMNSL', \
            'code_priority_G2': 'PYWCMNDSLX', \
            'code_priority_G5': 'IQX', \
            'code_priority_R1': 'PC', \
            'code_priority_R2': 'PC', \
            'code_priority_R3': 'IQX', \
            'code_priority_R4': 'ABX', \
            'code_priority_R6': 'ABX', \
            'code_priority_E1': 'CABXZ', \
            'code_priority_E5': 'IQX', \
            'code_priority_E6': 'ABCXZ', \
            'code_priority_E7': 'IQX', \
            'code_priority_E8': 'IQX', \
            'code_priority_C1': 'DPXAN', \
            'code_priority_C2': 'IQX', \
            'code_priority_C5': 'DPX', \
            'code_priority_C6': 'IQXA', \
            'code_priority_C7': 'DPZIQX', \
            'code_priority_C8': 'DPX', \
            'frequency_num_G' : ['1', '2', '5'],
            'frequency_num_R' : ['1', '2', '3', '4', '6'],
            'frequency_num_E' : ['1', '5', '6', '7', '8'],
            'frequency_num_C' : ['1', '2', '5', '6', '7', '8'],
            'frequency_G': [1575.42, 1227.60, 1176.45], \
            'frequency_E': [1575.42, 1176.45, 1278.75, 1207.140, 1191.795], \
            'frequency_C': [1575.42, 1561.098, 1176.45, 1268.52, 1207.140, 1191.795], \
            'sats_num_G': 32, \
            'sats_num_R': 24, \
            'sats_num_E': 36, \
            'sats_num_C': 63, \
            'max_dtoe_G': 7200, \
            'max_dtoe_R': 1800, \
            'max_dtoe_E': 10800, \
            'max_dtoe_C': 21600, \
            'GEO_C': [], \
            #------------------#
            # user defined
            'spp_iteration': 10, \
            'min_ele': 10, \
            # 'GC': enable GPS and BDS, 'GER': enable GPS, GAL, and GLO
            'enable_sys': 'GC', \
            'frequency_num': 1, \
            }

