# CorticalDynamics
Code for training neural network controllers to emulate posture and reach tasks.

## Dependecies.

### Python Libraries:
  1. PyTorch deep learning library
  2. Numpy
  3. Scipy
  4. Matplotlib
  
## File information
Each folder has three python scripts.
1. muscularArmClass.py contains dynamics of 2-DOF planar arm containing shoulder and elbow joints and muscle activation dynamics. 
2. NetworkClass.py contains script for neural network and cost computation for the executed movements.
3. optimizingscript.py runs the posture/reach tasks and optimizes/trains neural networks to learn an optimal control policy.


## To Run.
1. Go to posture/reach folder
2. Open and execute script titled "optimizingscript_xxtask.py" where xx = reach for reach task and xx = posture for posture task
3. Data is saved as .mat files in the folders named "data"


## Reference

The code is a part of research article titled "Rotational dynamics in motor cortex are consistent with a feedback controller" in eLife (DOI: 10.7554/eLife.67256). For citing this code, please use

eLife 2021;10:e67256



