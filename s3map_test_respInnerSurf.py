#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 20:22:52 2022

@author: fenqiang
"""
longleaf = True


import argparse
import numpy as np

from s3pipe.utils.vtk import read_vtk
from s3pipe.utils.utils import get_neighs_faces, get_sphere_template
from s3pipe.surface.s3map import moveOrigSphe, computeAndWriteDistortionOnOrigSphe, \
    computeAndWriteDistortionOnRespSphe, computeAndSaveDistortionFile
   
if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='move original sphere and resample it',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--std_sphe_moved', default='None', help="filename of std_sphe_moved", required=True)
    parser.add_argument('--orig_sphe', default='None', help="filename of orig_sphere")
    parser.add_argument('--inner_surf', default='None', help="inner_surf filename")
    parser.add_argument('--out_name', default='None', help="output filename of orig_sphe_moved")

    args = parser.parse_args()
    std_sphe_moved = args.std_sphe_moved
    orig_sphe = args.orig_sphe
    inner_surf = args.inner_surf
    out_name = args.out_name
    print('std_sphe_moved:', std_sphe_moved)
    print('orig_sphe:', orig_sphe)
    print('inner_surf:', inner_surf)
    print('out_name:', out_name)
    
    moved_surf = read_vtk(std_sphe_moved)
    orig_surf = read_vtk(orig_sphe)
    inner_surf = read_vtk(inner_surf)
    
    n_ver = len(moved_surf['vertices'])
    template = get_sphere_template(n_ver)
    neighs_faces = get_neighs_faces(n_ver)
    
    print("Moving original sphere according to ico sphere deformation...")
    orig_sphere_moved = moveOrigSphe(template, orig_surf, moved_surf, inner_surf, neighs_faces, out_name)
    print("Moving original sphere done!")
     
    computeAndWriteDistortionOnOrigSphe(orig_sphere_moved, inner_surf, out_name)
    print("computeAndWriteDistortionOnOrigSphe done!")
    
    print("Resampling inner surface...")
    template_163842 = get_sphere_template(163842)
    computeAndWriteDistortionOnRespSphe(orig_sphere_moved, template_163842, 
                                        inner_surf, out_name.replace('.vtk', '.RespInner.vtk'),
                                        compute_distortion=True)
    print("computeAndWriteDistortionOnRespSphe done!")

    # computeAndSaveDistortionFile(os.path.dirname(orig_sphe) +'/'+ sub_id +'.Inner.vtk', out_name)
    # print("Compute and save distortion npy done!")
    
    a = read_vtk(out_name.replace('.vtk', '.RespInner.vtk'))
    a = a['vertices']
    np.save(out_name.replace('.vtk', '.RespInner.npy'), a)
    print("Save RespInner npy done!")