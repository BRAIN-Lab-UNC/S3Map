#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 23:36:22 2021

@author: Fenqiang Zhao
contact: zhaofenqiang0221@gmail.com
"""

import sys

sys.path.append('/proj/ganglilab/users/Fenqiang/sunetpkg_py39/lib/python3.9/site-packages')
sys.path.append('/nas/longleaf/rhel8/apps/python/3.9.6/lib/python3.9/site-packages')

import argparse
import os

abspath = os.path.abspath(os.path.dirname(__file__))

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Superfast spherical surface pipeline \n'+\
                                     '1. Computing mean curvature and area \n'+\
                                     '2. Inflating the inner surface and computing sulc \n'+\
                                     '3. Spherical mapping \n'+\
                                     '4. Spherical registration \n'+\
                                     '5. Parcellation',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', '-i', default=None, required=True, 
                        help="the input inner surface in vtk format, containing vertices and faces")
    
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
   
    input_name = args.input
        
    
    print('\nProcessing input surface:', input_name)
    
    # compute curv and area
    os.system(" ".join(['python', abspath+'/compute_curv_area.py', '-i', input_name]))
    
    # inflate and compute sulc
    os.system(" ".join(['python', abspath+'/s3inflate.py', '-i', input_name]))
   
    # initial spherical mapping and resampling
    os.system(" ".join(['python', abspath+'/initial_spherical_mapping.py', '-i', input_name.replace('.vtk', '.inflated.vtk')]))
   