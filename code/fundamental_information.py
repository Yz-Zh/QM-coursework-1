# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 13:55:53 2020

@author: DIY
"""

import numpy as np


work_path='C:\\Users\\DIY\\Desktop\\QM_python\\data_cw_1\\coursework_1_data_2019.csv'
save_path='C:\\Users\\DIY\\Desktop\\QM_python\\data_cw_1\\fundamental information.csv'


def get_fund_info(work_path, save_path, work_col_y=1):
    file = np.genfromtxt(work_path,delimiter = ',')
    lines=file[1:,work_col_y]
    mean=0.0
    temp=0
    for line in lines:
        mean+=float(line)
        temp+=1
    mean=mean/temp
    for line in lines:
        var=(line-mean)**2
        var=var/temp
        stdev=var**0.5
        
    print(min(lines))
    print(max(lines))
    
    with open(save_path, 'a') as save_file_object:
        save_file_object.write("mean="+str(mean)+"\nvar="+str(var)+"\nstdev="+str(stdev))
    
    print(mean)
    print(stdev)
    print(var)
    print("processed")

    
get_fund_info(work_path, save_path, work_col_y=2)

