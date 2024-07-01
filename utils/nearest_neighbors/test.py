import numpy as np
import lib.python.nearest_neighbors as nearest_neighbors
import time
import torch
import open3d as o3d

batch_size = 1
num_points = 4000
K = 16
pc = np.load('/data/zclin/hl/RandLA-Net-pytorch/process_dataset/00/velodyne/000000.npy')
pc = torch.from_numpy(pc).unsqueeze(0)
idx = np.random.choice(pc.shape[1], (pc.shape[1]//4))
idx = torch.from_numpy(idx).unsqueeze(0)
sub_pc = torch.gather(pc.permute(0, 2, 1), 2, idx.unsqueeze(1).repeat(pc.shape[0], pc.shape[2], 1)).permute(0, 2, 1)
down1 = sub_pc.numpy()
sub_pc2 = torch.rand(batch_size, (num_points // 16), 3)
# nearest neighbours
start = time.time()
neigh_idx = nearest_neighbors.knn_batch(pc, pc, K, omp=True)
up_idx = nearest_neighbors.knn_batch(sub_pc, pc, 1, omp=True)

def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, d, N] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, up_num_points, d] interpolated features matrix
        """
        batch_size = interp_idx.shape[0]
        up_num_points = interp_idx.shape[1]
        interp_idx = interp_idx.reshape(batch_size, up_num_points)
        interpolated_features = torch.gather(feature, 2, interp_idx.unsqueeze(1).repeat(1, feature.shape[1], 1))
        up1 = interpolated_features.permute(0, 2, 1).numpy()
        interpolated_features = interpolated_features.unsqueeze(3)  # batch*channel*npoints*1
        return up1
up_idx = torch.from_numpy(up_idx)
sub_pc = sub_pc.permute(0, 2, 1)
up1 = nearest_interpolation(sub_pc, up_idx)

vis = o3d.visualization.Visualizer()

gt = o3d.geometry.PointCloud()
down = o3d.geometry.PointCloud()
up = o3d.geometry.PointCloud()
gt.points = o3d.utility.Vector3dVector(pc[0])
gt.paint_uniform_color([0.5,1,0])
down.points = o3d.utility.Vector3dVector(down1[0])
down.paint_uniform_color([0,0,0])
up.points = o3d.utility.Vector3dVector(up1[0])
up.paint_uniform_color([0,1,0.5])

vis.create_window()
#vis.add_geometry(gt)
vis.add_geometry(down)
vis.add_geometry(up)

vis.poll_events()
vis.update_renderer()
vis.run()

print(time.time() - start)


