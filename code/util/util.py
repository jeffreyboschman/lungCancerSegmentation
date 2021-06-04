import collections
import numpy as np




IrcTuple = collections.namedtuple('IrcTuple', ['index', 'row', 'col'])
XyzTuple = collections.namedtuple('XyzTuple', ['x', 'y', 'z']) #Named tuples assign meaning to each position in a tuple and allow for more readable, self-documenting code. They can be used wherever regular tuples are used, and they add the ability to access fields by name instead of position index

def irc2xyz(coord_irc, origin_xyz, vxSize_xyz, direction_a):
    cri_a = np.array(coord_irc)[::-1] #flip from irc to cri to align with xyz
    origin_a = np.array(origin_xyz) #shape is (3,)
    vxSize_a = np.array(vxSize_xyz) #shape is (3,)
    coords_xyz = (direction_a @ (cri_a * vxSize_a)) + origin_a #scale the indices with the voxel sizes (element-wise multiplication with *), then matrix-multiiply (dot product)  with the directions matrix (using the @), add the offset for the origin
    # coords_xyz = (direction_a @ (idx * vxSize_a)) + origin_a
    return XyzTuple(*coords_xyz)

def xyz2irc(coord_xyz, origin_xyz, vxSize_xyz, direction_a):
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coord_a = np.array(coord_xyz)
    cri_a = ((coord_a - origin_a) @ np.linalg.inv(direction_a)) / vxSize_a
    cri_a = np.round(cri_a) #rounds here before converting to int in the next step
    return IrcTuple(int(cri_a[2]), int(cri_a[1]), int(cri_a[0]))
