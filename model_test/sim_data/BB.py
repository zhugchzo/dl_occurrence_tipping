#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 16 2024

@author: Chengzuo Zhuge

Compute lag-1 autocorrelation by BB method
See Christopher Boettner and Niklas Boers, PHYSICAL REVIEW RESEARCH 4, 013230 (2022)

"""

import numpy as np
import math
from sklearn import linear_model

def ac_red(x,b,rw_length):

    x = np.array(x)
    b = np.array(b)
    rw = int(rw_length*len(x))
    window_over = len(x) - rw - 1
    ac_list = []
    b_list = []

    for i in range(window_over):
        rolling_window_1 = x[i:i+rw]
        rolling_window_2 = x[i+1:i+rw+1]
        rolling_window_1 = rolling_window_1.reshape(-1,1)
        rolling_window_2 = rolling_window_2.reshape(-1,1)
        reg_x = linear_model.LinearRegression()
        reg_x.fit(rolling_window_1,rolling_window_2)
        fai_b = reg_x.coef_[0][0]

        v_list_1 = []
        for j in range(rw):
            v_j = x[i+j+1] - fai_b*x[i+j]
            v_list_1.append(v_j)
        v_list_2 = v_list_1[1:]
        v_list_2.append(x[i+rw+1] - fai_b*x[i+rw])

        # v_list_1 = np.array(v_list_1).reshape(-1,1)
        # v_list_2 = np.array(v_list_2).reshape(-1,1)
        # reg_v = linear_model.LinearRegression()
        # reg_v.fit(v_list_1,v_list_2)
        # rho_b = reg_v.coef_[0][0]

        frac_n = np.dot(v_list_1,v_list_2)
        frac_d = np.dot(v_list_1,v_list_1)
        rho_b = frac_n/frac_d

        if fai_b > rho_b and (fai_b+rho_b)**2-4*rho_b/fai_b >= 0:
            fai = (fai_b+rho_b + math.sqrt((fai_b+rho_b)**2-4*rho_b/fai_b))/2
        elif fai_b <= rho_b and (fai_b+rho_b)**2-4*rho_b/fai_b >= 0:
            fai = (fai_b+rho_b - math.sqrt((fai_b+rho_b)**2-4*rho_b/fai_b))/2
        else:
            fai = math.sqrt(rho_b/fai_b)
        
        ac_list.append(fai)
        b_list.append(b[i+rw])
    
    return ac_list,b_list
        
         





