mmllhs_x.pickle, mmllhs_y.pickle, mmllhs_1x2D.pickle, mmllhs_2x1D.pickle, mmllhs_2x2D.pickle

The files contain numpy.ndarrays of shape (7, 15, 50). 

Parameters of the exploration:
sigmas = [.1, .2, .3, .4, .5, .6, .7]
num_sim = 15
Ts = [6, 6, 10, 25, 35, 42, 50]  # number of datapoints taken into account depends on sigma


1x1D vs 2x2D: 15 datasets (Ddiag_1, Ddiag_2, ..., Ddiag_15) consisting of 50 paired diagonal (+- 45 degree) data points were generated in advance.
1x2D vs 2x1D: 15 datasets (Dcard_1, Dcard_2, ..., Dcard_15) consisting of 50 paired cardinal (0/90 degree) data points were generated in advance. 

The arrays mmllhs_x, mmllhs_y, etc were initialized as numpy arrays containing zeros. 15 simulations were performed for each sigma. 
In every simulation, the amount of data (from the dataset with index corresponding to the index of simulation) specified by the list "Ts" was iteratively added to the set,
for which the mmLLH was computed.
The values of mmLLH were assigned to the corresponding entry of the arrays. For instance, by sigma = .3, in the fourth simulation, when
only 6 data points (out of 10, which is the corresponding maximum value by sigma = .3) were taken into account (from dataset Dcard_4), the mmLLH of 1x2D and 2x1D models
were computed for that 6 data points, and the resulting values were assigned to mmllhs_1x2D[2, 3, 5] and mmllhs_2x1D[2, 3, 5] respectively.





