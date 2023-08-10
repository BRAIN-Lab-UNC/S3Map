#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 01:13:04 2021

@author: Fenqiang Zhao

@contact: zhaofenqiang0221@gmail.com
"""
import argparse
import numpy as np
import glob
import os
import torch
import math

from s3pipe.models.models import SUnet
from s3pipe.utils.interp_torch import convert2DTo3D, getEn, diffeomorp_torch, get_bi_inter
from s3pipe.utils.utils import get_neighs_order, get_sphere_template
from s3pipe.surface.s3map import ResampledInnerSurf, computeMetrics_torch
from s3pipe.surface.prop import countNegArea
from s3pipe.utils.vtk import read_vtk, write_vtk
from s3pipe.utils.interp_numpy import resampleSphereSurf

abspath = os.path.abspath(os.path.dirname(__file__))


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='Perform spherical mapping of cortical surfaces with minimal metric distortion. '+\
                                     'It needs the initially spherical mapped and resampled surface using initial_spherical_mapping.py,' +\
                                     'and its corresponding inner surface in .vtk format, the vtk files should contain vertices and faces fields.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--inner_surf', default=None, help="filename of the input resampled inner surface")
    parser.add_argument('--files_pattern', default=None, help="pattern of inner surface files, this can help process multiple files in one command. "+\
                                                              "Note the pattern needs to be quoted for python otherwise it will be parsed by the shell by default. "+\
                                                              "Either single file or files pattern should be given")
    parser.add_argument('--hemi', default=None, help="hemisphere, lh or rh", required=True)
    parser.add_argument('--device', default='GPU', choices=['GPU', 'CPU'], 
                        help='The device for running the model.')
    parser.add_argument('--model_path', default=None, help="model folder, if not given will be ./pretrained_models ")
    args = parser.parse_args()
    inner_surf = args.inner_surf
    files_pattern = args.files_pattern
    hemi = args.hemi
    device = args.device
    model_path = float(args.model_path)
    print('\ninner_surf:', inner_surf)
    print('files_pattern:', files_pattern)
    print('hemi:', hemi)
    print('model_path:', model_path)
        
    # check device
    if device == 'GPU':
        device = torch.device('cuda:0')
    elif device =='CPU':
        device = torch.device('cpu')
    else:
        raise NotImplementedError('Only support GPU or CPU device')
    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print('device:', device)
    
    if inner_surf is None and files_pattern is None:
        raise NotImplementedError('Either single inner_surf file or files pattern should be given.')
    if inner_surf is not None:
        files = [inner_surf]
    else:
        files = sorted(glob.glob(files_pattern))
    print('len(files):', len(files))
    
    dataset = 'HighRes'   # 'HighRes' for dataset BCP and HCP with vertices number between 140000 - 250000, 'LowRes' for dataset dHCP with vertices between 30000 to 140000
    weight_distan = 1.0
    weight_area = 0.1
    weight_smooth = 10.0
    batch_size = 1
     
    # starting distortion correction    
    print("Starting distortion correction...")
    n_vertexs = [10242, 40962, 163842]
    


    for n_vertex in n_vertexs:
        n_level = n_vertexs.index(n_vertex)
        print("Multi-level distottion correction -", n_level+1, "level with", n_vertex, "vertices.")
    
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
            
        
        if level == 6:
            resp_inner_file = [ x.replace('.vtk', '.SIP.RespInner.vtk') for x in files]
        else:
            resp_inner_file = [ x.replace('.vtk', '.SIP.RespSphe.'+str(n_vertexs[n_level-1])+'moved.RespInner.vtk') for x in files ]
        test_dataset = ResampledInnerSurf(resp_inner_file, n_vertex)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
        
        model = SUnet(in_ch=12, out_ch=2, level=level, n_res=n_res, rotated=0, complex_chs=32)
        print("{} paramerters in total".format(sum(x.numel() for x in model.parameters())))
        model.to(device)
        model_file = os.path.join(model_path, 'S3Map_'+ dataset+'_'+hemi+'_ver'+ str(n_vertex) + '_area'+ str(weight_area) + \
                                                 '_dist'+ str(weight_distan) +'_smooth'+ str(weight_smooth) +'.mdl')
        print('Loading pre-trained model:', model_file)
        model.load_state_dict(torch.load(model_file))
        print()
        model.load_state_dict(torch.load())
        
        neigh_orders = get_neighs_order(n_vertex)
        neigh_sorted_orders = np.concatenate((np.arange(n_vertex)[:, np.newaxis], neigh_orders[:, 0:6]), axis=1)
        template = get_sphere_template(n_vertex)
        fixed_xyz = torch.from_numpy(template['vertices']).to(device) / 100.0
        bi_inter = get_bi_inter(n_vertex, device)[0]
        En = getEn(n_vertex, device)[0]
        
        model.eval()
        for batch_idx, (inner_vert, file) in enumerate(test_dataloader):
            file = file[0]
            print("processing surface:", file)
            inner_vert = inner_vert.squeeze().to(device)  
    
            with torch.no_grad():
                inner_dist, inner_area, _ = computeMetrics_torch(inner_vert, neigh_sorted_orders, device)
                feat = torch.cat((inner_dist, inner_area), 1)  # features
                feat = feat.permute(1,0).unsqueeze(0)
                deform_2d = model(feat) / deform_scale
                deform_2d = deform_2d[0,:].permute(1,0)
                deform_3d = convert2DTo3D(deform_2d, En, device)
       
                # diffeomorphic implementation
                velocity_3d = deform_3d/math.pow(2, 6)
                moved_sphere_loc = diffeomorp_torch(fixed_xyz, velocity_3d, 
                                                  num_composition=6, bi=True, 
                                                  bi_inter=bi_inter, 
                                                  device=device)
            
            orig_surf = read_vtk(file)
    
            orig_sphere = {'vertices': fixed_xyz.cpu().numpy() * 100.0,
                            'faces': template['faces'],
                            'sulc': orig_surf['sulc'][0:n_vertex],
                            # 'curv': orig_surf['curv'][0:n_vertex],
                            'deformation': deform_3d.cpu().numpy() * 100.0}
            write_vtk(orig_sphere, file.replace('RespInner.vtk', 
                                                'RespSphe.'+ str(n_vertex) +'deform.vtk'))
            
            corrected_sphere = {'vertices': moved_sphere_loc.cpu().numpy() * 100.0,
                                'faces': template['faces'],
                                'sulc': orig_surf['sulc'][0:n_vertex],
                                # 'curv': orig_surf['curv'][0:n_vertex],
                                }
            neg_area = countNegArea(corrected_sphere['vertices'], corrected_sphere['faces'][:, 1:])
            print("Negative areas of the deformation: ", neg_area)
            write_vtk(corrected_sphere, file.replace('RespInner.vtk', 
                                                     'RespSphe.'+ str(n_vertex) +'moved.vtk'))
        

            # postprocessing
            orig_inner_surf = read_vtk(file)
            orig_sphe_surf = read_vtk(file.replace('.vtk', '.SIP.vtk'))
            orig_sphere_moved = resampleSphereSurf(template['vertices'], 
                                                    orig_sphe_surf['vertices'],
                                                    moved_sphere_loc.cpu().numpy() * 100.0,
                                                    neigh_orders=neigh_orders)
            orig_sphere_moved = 100 * orig_sphere_moved / np.linalg.norm(orig_sphere_moved, axis=1)[:, np.newaxis]
            moved_orig_sphe_surf = {'vertices': orig_sphere_moved,
                                    'faces': orig_inner_surf['faces'],
                                    'sulc': orig_inner_surf['sulc']}
            write_vtk(moved_orig_sphe_surf, file.replace('.vtk', '.SIP.RespSphe.'+str(n_vertex)+'moved.OrigSpheMoved.vtk'))
            print("Move original sphere done.")
             
            template_163842 = get_sphere_template(163842)
            computeAndWriteDistortionOnRespSphe(orig_sphere_moved, template_163842, 
                                                orig_inner_surf, 
                                                file.replace('.vtk', 
                                                             '.SIP.RespSphe.'+str(n_vertex)+'moved.RespInner.vtk'))
            print("Resample original inner and sphere surface done!")
         