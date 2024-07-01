import numpy as np
import lib.python.nearest_neighbors as nearest_neighbors
import time
import torch
from tqdm import tqdm

def gather_neighbour(pc, neighbor_idx):  # pc: batch*npoint*channel
    # gather the coordinates or features of neighboring points
    batch_size = pc.shape[0]
    num_points = pc.shape[1]
    d = pc.shape[2]
    index_input = neighbor_idx.reshape(batch_size, -1)
    features = torch.gather(pc, 1, index_input.unsqueeze(-1).repeat(1, 1, pc.shape[2]))
    features = features.reshape(batch_size, num_points, neighbor_idx.shape[-1], d)  # batch*npoint*nsamples*channel
    
    return features

if __name__ == "__main__":
        
    batch_size = 1
    K = 1024
    scan = np.fromfile("/data/zclin/hl/PolarSeg/data/sequences/00/velodyne/000000.bin", dtype=np.float32)
    scan = scan.reshape((-1, 4))
    points = scan[:8000, 0:3]
    points = torch.from_numpy(points).unsqueeze(0)

    Time = 0
    for i in tqdm(range(50)):
        # nearest neighbours
        start = time.time()
        neigh_idx = nearest_neighbors.knn_batch(points, points, K, omp=True)
        neigh_idx = torch.from_numpy(neigh_idx)
        
        f_neighbours = gather_neighbour(points, neigh_idx)
        end = time.time() - start
        Time += end

    Time = (Time / 50)*1000
    print("耗时为: %.2fms" % Time)
