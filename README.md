Directional and magnitude dropout for neural network with weight normalization (WN) parametrization.

In WN instead of patameter weight vector <img src="https://latex.codecogs.com/gif.latex?w"/>, we have  <img src="https://latex.codecogs.com/gif.latex?w=g\,v/\|v\|" /> with norm <img src="https://latex.codecogs.com/gif.latex?g"/> and direction vector <img src="https://latex.codecogs.com/gif.latex?v"/>. 
 
We experiment with different ways of applying directional noise in WN network, effectively we need to sample a unit vector from some distribution on a unit sphere with mean <img src="https://latex.codecogs.com/gif.latex?v/\|v\|" />:
* sample Gaussian noise cenetered at <img src="https://latex.codecogs.com/gif.latex?v/\|v\|" /> and then project it to unit sphere
* sample Gaussian noise cenetered at <img src="https://latex.codecogs.com/gif.latex?v/\|v\|" />, project it to tangent space (so that the maximum angle between <img src="https://latex.codecogs.com/gif.latex?v/\|v\|" /> and sampled vector would not exceed <img src="https://latex.codecogs.com/gif.latex?\pi/2" />) and then project it to unit sphere
* choose random direction on the unit <img src="https://latex.codecogs.com/gif.latex?n-1" /> dimensional sphere, and then sample the angle from some one dimentional distribution on <img src="https://latex.codecogs.com/gif.latex?[-\pi/2\,;\,\pi/2]" /> (like stretched Beta or truncated Normal)

For noise on magnitude, we choose multiplicative Gaussian noise (it is not equivalent to Gaussian dropout). 
