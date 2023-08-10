# S3Map
This is the code for fast spherical mapping of cortical surfaces using [S3Map algorithm](https://link.springer.com/chapter/10.1007/978-3-031-16446-0_16).
![Figure framework](https://github.com/zhaofenqiang/S3Map/blob/main/examples/fig_framework.png)

# Usage
1. Download or clone this repository into a local folder
2. Open a terminal and run the follwing code (better do this in a conda environment):
```
pip install s3pipe pyvista  tensorboard torch torchvision torchaudio
```
if only cpu is available, you can install the cpu version torch
```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
```
3. Prepare your data, i.e., inner surfaces in vtk format (the file name should end in '.vtk')
4. Simply run "python s3all.py -h" for the spherical mapping, and the expected output should be.
```
usage: s3map.py [-h] [--inner_surf INNER_SURF] [--files_pattern FILES_PATTERN] --hemi HEMI [--device {GPU,CPU}]
                [--model_path MODEL_PATH]

Perform spherical mapping of cortical surfaces with minimal metric distortion. It needs the initially spherical mapped
and resampled surface using initial_spherical_mapping.py,and its corresponding inner surface in .vtk format, the vtk
files should contain vertices and faces fields.

optional arguments:
  -h, --help            show this help message and exit
  --inner_surf INNER_SURF
                        filename of the input resampled inner surface (default: None)
  --files_pattern FILES_PATTERN
                        pattern of inner surface files, this can help process multiple files in one command. Note the
                        pattern needs to be quoted for python otherwise it will be parsed by the shell by default. Either
                        single file or files pattern should be given (default: None)
  --hemi HEMI           hemisphere, lh or rh (default: None)
  --device {GPU,CPU}    The device for running the model. (default: GPU)
  --model_path MODEL_PATH
                        model folder, if not given will be ./pretrained_models (default: None)
```
5. Use [paraview](https://www.paraview.org/) to visualize all generated .vtk surfaces, or [read_vtk](https://github.com/zhaofenqiang/S3Map/blob/a96c103f66db443ba52cdafee28af798a527fc54/sphericalunet/utils/vtk.py#L26) into python environment for further processing.

## Train a new model on a new dataset
After data prepration, modify the train.py file to match the training data in your own path. Then, run:
```
python s3map_train.py
```


