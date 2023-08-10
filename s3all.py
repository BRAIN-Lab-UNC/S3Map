#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 23:36:22 2021

@author: Fenqiang Zhao
contact: zhaofenqiang0221@gmail.com
"""

import glob
import argparse
import os

abspath = os.path.abspath(os.path.dirname(__file__))

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Superfast spherical surface pipeline \n'+\
                                     '1. Computing mean curvature and area \n'+\
                                     '2. Inflating the inner surface and computing sulc \n'+\
                                     '3. Initial spherical mapping \n'+\
                                     '4. Distortion correction for initial spherical surface.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--input', '-i', default=None, required=True, 
                        help="the input inner surface (white matter surface) in vtk format, containing vertices and faces. "+\
                             "There should be 'lh' or 'rh' in the filename for identifying left hemisphere or right hemisphere.")
    parser.add_argument('--save_interim_results', default='False', help="save intermediate results or not, "+\
                        "if Ture, there will be many results generated and need more storage", 
                        choices=['True', 'False'])
        
    # try:
    #     args = parser.parse_args()
    # except:
    #     parser.print_help()
    #     sys.exit(0)
    
    args = parser.parse_args()
    input_name = args.input
    save_interim_results = args.save_interim_results == 'True'
        
    if 'lh' in input_name:
        hemi = 'lh'
    elif 'rh' in input_name:
        hemi = 'rh'
    else:
        raise NotImplementedError('The filename of the input inner surface should contain lh or rh for identifying the hemisphere.')
    
    print('\nProcessing input surface:', input_name)
    print('hemi:', hemi)
    print('save_interim_results:', save_interim_results)
    
    # compute curv and area
    os.system(" ".join(['python', abspath+'/compute_curv_area.py', '-i', input_name]))
    
    # inflate and compute sulc
    os.system(" ".join(['python', abspath+'/s3inflate.py', '-i', input_name]))
   
    # initial spherical mapping and resampling
    os.system(" ".join(['python', abspath+'/initial_spherical_mapping.py', '-i', input_name.replace('.vtk', '.inflated.vtk')]))
   
    # distortion correction
    os.system(" ".join(['python', abspath+'/s3map.py',
                        '--resp_inner_surf', input_name.replace('.vtk', '.inflated.SIP.RespInner.vtk'), 
                        '--hemi', hemi, 
                        '--save_interim_results', args.save_interim_results]))
   
    
    if not save_interim_results:
        print('\nremoving intermediate results...')
        for f in glob.glob(input_name.replace('.vtk', '.*.RespSphe.vtk')):
            os.remove(f)
        for f in glob.glob(input_name.replace('.vtk', '.*.RespInner.vtk')):
            os.remove(f)
        print('Done.')