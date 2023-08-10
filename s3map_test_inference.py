#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 01:13:04 2021
Modified on 04/12/2023 for lifespan atlas construction

@author: Fenqiang Zhao

@contact: zhaofenqiang0221@gmail.com
"""

import argparse

longleaf = True


import numpy as np
import glob
import torch
import math
from s3pipe.models.models import SUnet
from s3pipe.utils.interp_torch import convert2DTo3D, getEn, diffeomorp_torch, get_bi_inter
from s3pipe.utils.utils import get_neighs_order, get_sphere_template, get_neighs_faces
from s3pipe.surface.s3map import ResampledInnerSurf, computeMetrics_torch,\
    moveOrigSphe, computeAndWriteDistortionOnOrigSphe, \
    computeAndWriteDistortionOnRespSphe, computeAndSaveDistortionFile
from s3pipe.surface.prop import countNegArea
from s3pipe.utils.vtk import read_vtk, write_vtk


###############################################################################


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='run s3map inference',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--file', default=None, help="filename of the input resampled inner surface vertices")
    parser.add_argument('--files_pattern', default=None, help="pattern of files, note the pattern needs to be quoted for python otherwise it will be parsed by the shell. "+\
                        "Either single file or files pattern should be given")
    parser.add_argument('--hemi', default=None, help="hemisphere, lh or rh", required=True)
    parser.add_argument('--vertex', default=0, help="number of vertices, 10242, or 40962, or 163842", required=True)
    parser.add_argument('--smooth', default=0.0, help="weights of smoth for pre-training the model")
    parser.add_argument('--weight_area', default=0.0, help="weights of area distortion for pre-training the model")
    args = parser.parse_args()
    file = args.file
    files_pattern = args.files_pattern
    hemi = args.hemi
    n_vertex = int(args.vertex)
    weight_smooth = float(args.smooth)
    weight_area = float(args.weight_area)
    print('\nfile:', file)
    print('files_pattern:', files_pattern)
    print('hemi:', hemi)
    print('n_vertex:', n_vertex)
    print('weight_smooth:', weight_smooth)
    print('weight_area:', weight_area)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print('device:', device)
    
    if file is None and files_pattern is None:
        raise NotImplementedError('Either single file or files pattern should be given.')
    if file is not None:
        files = [file]
    else:
        files = sorted(glob.glob(files_pattern))
    print('len(files):', len(files))
    
    dataset = 'HighRes'   # 'HighRes' for dataset BCP and HCP with vertices number between 140000 - 250000, 'LowRes' for dataset dHCP with vertices between 30000 to 140000
    weight_distan = 1.0
    batch_size = 1
    
    
    if n_vertex == 10242:
        deform_scale = 10.0
        level = 6
        n_res = 4   
    elif n_vertex == 40962:
        deform_scale = 30.0
        level = 7
        n_res = 4
    elif n_vertex == 163842:
        deform_scale = 30.0
        level = 8
        n_res = 5
    else:
        raise NotImplementedError('Error')
        
    test_dataset = ResampledInnerSurf(files, n_vertex)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    model = SUnet(in_ch=12, out_ch=2, level=level, n_res=n_res, rotated=0, complex_chs=32)
    print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
    model.to(device)
    if longleaf:
        model_file = '/proj/ganglilab/users/Fenqiang/S3Pipeline/pretrained_models/S3Map_'+ \
                                         dataset+'_'+hemi+'_ver'+ str(n_vertex) + '_area'+ str(weight_area) + \
                                             '_dist'+ str(weight_distan) +'_smooth'+ str(weight_smooth) +'.mdl'
        
    else:
        model_file = '/media/ychenp/fq/S3_pipeline/SphericalMapping/pretrained_models/S3Map_'+ \
                                         dataset+'_'+hemi+'_ver'+ str(n_vertex) + '_area'+ str(weight_area) + \
                                             '_dist'+ str(weight_distan) +'_smooth'+ str(weight_smooth) +'.mdl'
    print('Loading pre-trained model:', model_file)
    model.load_state_dict(torch.load(model_file))
    print()
    
    neigh_orders = get_neighs_order(n_vertex)
    neigh_sorted_orders = np.concatenate((np.arange(n_vertex)[:, np.newaxis], neigh_orders[:, 0:6]), axis=1)
    template = get_sphere_template(n_vertex)
    fixed_xyz = torch.from_numpy(template['vertices']).to(device) / 100.0
    bi_inter = get_bi_inter(n_vertex, device)[0]
    En = getEn(n_vertex, device)[0]
    
    # dataiter = iter(test_dataloader)
    # inner_vert, file = dataiter.next()
        
    for batch_idx, (inner_vert, file) in enumerate(test_dataloader):
        model.train()
        file = file[0]
        print(file)
        inner_vert = inner_vert.squeeze().to(device)  

        with torch.no_grad():
            inner_dist, inner_area, _ = computeMetrics_torch(inner_vert, neigh_sorted_orders, device)
            # feat = inner_dist  # features  # 2023.6.21 change feat to only distance
            feat = torch.cat((inner_dist, inner_area), 1)  # features
            feat = feat.permute(1,0).unsqueeze(0)
            deform_2d = model(feat) / deform_scale  # 10.0 for 10242, 30.0 for 40962
            deform_2d = deform_2d[0,:].permute(1,0)
            deform_3d = convert2DTo3D(deform_2d, En, device)
   
            # diffeomorphic implementation
            velocity_3d = deform_3d/math.pow(2, 6)
            moved_sphere_loc = diffeomorp_torch(fixed_xyz, velocity_3d, 
                                              num_composition=6, bi=True, 
                                              bi_inter=bi_inter, 
                                              device=device)
        
            # moved_dist, moved_area, __ = computeMetrics_torch(moved_sphere_loc, neigh_sorted_orders, device)
        
               
        inner_surf = read_vtk(file)
        orig_surf =  read_vtk(file.replace('.vtk', '.SIP.Sphe.vtk'))

        orig_sphere = {'vertices': fixed_xyz.cpu().numpy() * 100.0,
                        'faces': template['faces'],
                        'sulc': inner_surf['sulc'][0:n_vertex],
                        # 'curv': inner_surf['curv'][0:n_vertex],
                        'deformation': deform_3d.cpu().numpy() * 100.0}
        write_vtk(orig_sphere, file.replace('RespInner.npy', 
                                            'RespSphe.'+ str(n_vertex) +'deform.vtk'))
        
        moved_surf = {'vertices': moved_sphere_loc.cpu().numpy() * 100.0,
                            'faces': template['faces'],
                            'sulc': inner_surf['sulc'][0:n_vertex],
                            # 'curv': inner_surf['curv'][0:n_vertex],
                            }
        neg_area = countNegArea(moved_surf['vertices'], moved_surf['faces'][:, 1:])
        print("Negative areas of the deformation: ", neg_area)
        write_vtk(moved_surf, file.replace('RespInner.npy', 
                                                 'RespSphe.'+ str(n_vertex) +'moved.vtk'))
    
        neighs_faces = get_neighs_faces(n_vertex)
        print("Moving original sphere according to ico sphere deformation...")
        orig_sphere_moved = moveOrigSphe(template, orig_surf, moved_surf, inner_surf, neighs_faces, file.replace('.vtk', ''))
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
