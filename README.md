# 5390

## install different BLAS

1. create the environment for different experiment
conda create -n ENV_NAME --no-default-packages
2. activate the env

3.install the common dependency

basic:
cffi cmake hypothesis ninja numpy  typing_extensions pyyaml setuptools

4. install the special dependency

5. set the environment variables
conda env config vars set
BLAS=Eigen
USE_CUDA=0
USE_ROCM=0

6. deactivate and re-activate

7. install

python setup.py install --cmake
