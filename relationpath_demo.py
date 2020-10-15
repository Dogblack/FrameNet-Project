import numpy as np
import time

path  = './parsed-v1.5/'

def serve(i, j, pathes, dis):
    print(f"distance {dis[i][j]}")
    if (i, j) not in pathes:
        return
    p_ij = pathes[(i, j)]
    p = ['->'.join([str(v) for v in pp]) for pp in p_ij]
    print(f"num of pathes {len(p_ij)}")
    print(f"pathes {p}")
    return


if __name__ == "__main__":
    t0 = time.time()
    pathes = np.load(path+'shortestpathes_between_frames.npy', allow_pickle=True).item()
    t1 = time.time()
    distance_matrix = np.load(path+'frame_dis_matrix.npy')
    t2 = time.time()
    print(f"loading pathes using {t1-t0}s")
    print(f"loading distance matrix using {t2-t1}s")
    serve(264, 165, pathes, distance_matrix)


# dicts = np.load(path, allow_pickle=True)

# print(dicts.item()[(0, 2)])