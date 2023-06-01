"""BuildNetwork package has the steps to construct a network
from random points (or after loading existing points) using
Voronoi tessellation / Delauney triangulation.

The original scripts are from Jenny Liu (Keten lb alumni)
"""


import numpy as np
import pandas as pd
import scipy as sp
from scipy import spatial
from scipy.special import comb
# from scipy.sparse import linalg

import sys
sys.path.append(r'..')
import Angle as angle  # noqa: E402


RADIUS = 20
NUM_PTS = 12
NUM_DIM = 2
NUM_MESH = 100
NUM_ITER = 5  # iterations in Lloyd relaxation


def rand_pts_circle(num_pts=NUM_PTS, r=RADIUS):
    """random points within circle
    """
    u = np.random.rand(num_pts)
    radii = np.sqrt(u) * r * 0.95
    theta = np.random.rand(num_pts)*2*np.pi
    # x, y
    x = radii*np.cos(theta)
    y = radii*np.sin(theta)
    points = np.column_stack([x, y])
    return points


# ----------------------------------------------------------------
#                            Lloyd relaxation
# ----------------------------------------------------------------
def Lloyd_relaxation(points, r=RADIUS, num_mesh=NUM_MESH, num_iter=NUM_ITER):
    """Lloyd relaxation creates a Voronoi tessellation
    Equally spaced regions, based on distance to point
    """
    mesh_x, mesh_y, mesh_xy = get_mesh(r, num_mesh)
    ref_points = points
    num_pts = np.shape(ref_points)[0]
    for i in range(num_iter):
        # assign mesh to regions around points
        mesh_split = sp.spatial.distance.cdist(mesh_xy, ref_points)
        region = np.argmin(mesh_split, axis=1) + 1
        reg_xy = np.reshape(region, [num_mesh, num_mesh])
        # remove mesh points outside of circle
        reg_circ = np.where(mesh_x**2 + mesh_y**2 > r**2, 0, reg_xy)
        # list of regions at each mesh location
        reg_list = reg_circ.flatten()
        cx = []
        cy = []
        for region_no in range(1, num_pts+1):
            reg_cx, reg_cy = calc_region_centroid(region_no, reg_list,
                                                  mesh_x, mesh_y)
            cx.append(reg_cx)
            cy.append(reg_cy)
        ref_points = np.column_stack([cx, cy])
    return ref_points


def get_mesh(r=RADIUS, num_mesh=NUM_MESH):
    """helper function for Lloyd relaxation
    to get grid in Cartesian coordinates
    """
    mesh_x_ls = np.linspace(-r, r, num_mesh)
    mesh_y_ls = np.linspace(-r, r, num_mesh)
    mesh_x, mesh_y = np.meshgrid(mesh_x_ls, mesh_y_ls)
    mesh_xy = np.column_stack([mesh_x.flatten(), mesh_y.flatten()])
    return mesh_x, mesh_y, mesh_xy


def calc_region_centroid(reg_no, reg_list, mesh_x, mesh_y):
    """helper function for Lloyd relaxation
    numerically estimate centroid of region in Cartesian space
    """
    ind_region = find_indices(reg_list, lambda e: e == reg_no)
    cx = np.mean(mesh_x.flatten()[ind_region])
    cy = np.mean(mesh_y.flatten()[ind_region])
    return cx, cy


def find_indices(array, condition):
    """helper function for Lloyd relaxation's centroid calculation
    identify region for mesh location
    """
    ind = []
    for i, val in enumerate(array):
        if condition(val):
            ind.append(i)
    return ind


# ----------------------------------------------------------------
#           Make network (and make it prettier)
# ----------------------------------------------------------------
def reorder_points(points, r=RADIUS):
    """reorder points by layer, and counterclockwise
    """
    # ring is distance from center
    r = points[:, 0].max() - points[:, 0].min()
    c = np.mean(points, axis=0)
    ring_dist = np.linalg.norm(points - c, axis=1)
    ring_group = np.round(ring_dist*10/r).astype(int)
    group_list = set(ring_group)
    # angle
    ang = angle.xy2ang(points[:, 0], points[:, 1])
    ang = np.vectorize(angle.wrap)(ang, 0, np.pi*2)
    # save ring and angle in df
    points_df = pd.DataFrame(data=points, columns=['x', 'y'])
    points_df['ring'] = ring_group
    points_df['ang'] = ang
    # sort
    points_df = points_df.sort_values(by=['ring', 'ang'])
    points = points_df[['x', 'y']].values
    return points


def reorder_points_X(points):
    """reorder points and edges from left ro right
    NOTE: would have to rewrite other things to also renumber edges
    """
    num_pts, num_dim = np.shape(points)
    # points
    pt_order = np.argsort(points[:, 0])
    points = points[pt_order, :]
    return points


def make_network(points):
    """Make Voronoi tesselation network using Delaunay triangulation
    """
    tri = sp.spatial.Delaunay(points)
    adj = tri2adj(tri)
    return adj


def tri2adj(tri):
    """helper function for make_network
    adjacency matrix from Delauney triangulation object"""
    indices, indptr = tri.vertex_neighbor_vertices
    num_pts = tri.npoints
    adj = np.zeros([num_pts, num_pts])
    for k in range(num_pts):
        k_neighbors = indptr[indices[k]:indices[k+1]]
        adj[k, k_neighbors] = 1
    return adj


def get_edges2D(adj, points, stiff=1):
    """build edgelist"""
    num_edges = int(np.sum(adj) / 2)
    num_pts = np.shape(adj)[0]
    edge_df = pd.DataFrame(data=np.zeros([num_edges, 7]),
                           index=np.arange(num_edges),
                           columns=['edge', 'i', 'j', 'bx', 'by', 'k', 'd'])
    edge_df.index.name = 'index'
    edge_ctr = 0
    for i in range(num_pts):
        for j in range(i+1, num_pts):
            if adj[i, j] > 0:
                edge_df.loc[edge_ctr, 'edge'] = str([i, j])
                edge_df.loc[edge_ctr, ['i', 'j']] = i, j
                #b = points[i, :] - points[j, :]
                b = points[j, :] - points[i, :]
                b = b / np.linalg.norm(b)
                edge_df.loc[edge_ctr, ['bx', 'by']] = b
                edge_df.loc[edge_ctr, 'k'] = stiff
                dist = np.linalg.norm(points[i, :] - points[j, :])
                edge_df.loc[edge_ctr, 'd'] = dist
                edge_ctr += 1
    return edge_df


def get_edge_dict(edge_ref_df):
    edge_dict = {}
    edge_ref_df = edge_ref_df.copy()
    edge_ref_df = edge_ref_df.reset_index().set_index('index')
    for ctr in edge_ref_df.index:
        i, j = edge_ref_df.iloc[ctr][['i', 'j']]
        edge_dict.update({ctr: [i, j]})
    return edge_dict


def nCr(n, r):
    val = sp.misc.comb(n, r, exact=True)
    return val


def cuts2adj(adj, edge_ref_df, cuts):
    """turn list of cuts into adjacency matrix"""
    new_adj = np.copy(adj)
    for c in cuts:
        i, j = edge_ref_df.iloc[c][['i', 'j']]
        new_adj[i, j] = 0
        new_adj[j, i] = 0
    return new_adj


def adj2barcode(adj, adj_ref):
    """turn adjacency matrix into barcode of cuts"""
    num_pts = np.shape(adj_ref)[0]
    barcode = []
    ctr = 0
    for i in range(num_pts):
        for j in range(i+1, num_pts):
            if adj_ref[i, j] > 0:
                barcode.append(int(adj[i, j]))
                ctr = ctr + 1
    return barcode


def bin2int(bin_list):
    """turn barcode of cuts into integer ID"""
    res = int("".join(str(x) for x in bin_list), 2)
    return res


def int2bin_cuts(val, num_edges):
    """Decode after bin2int to list of cut edges"""
    cut_str = "{0:b}".format(int(val))
    while len(cut_str) < num_edges:
        cut_str = '0' + cut_str
    # string of 1, 0 to list of 1, 0
    cut_bin = [int(s) for s in cut_str]
    cut_list = [i for i, s in enumerate(cut_bin) if s == 0]
    return cut_list


# ----------------------------------------------------------------
#          Simplifying network
# ----------------------------------------------------------------
def get_realedge(adj):
    """Remove edges just sticking out (degree = 1)"""
    adj_M = np.copy(adj)
    c = np.sum(adj_M, axis=0)
    ind_stick = np.argwhere(c == 1)
    while len(ind_stick) > 0:
        adj_M[ind_stick, :] = 0
        adj_M[:, ind_stick] = 0
        c = np.sum(adj_M, axis=0)
        ind_stick = np.argwhere(c == 1)
    return adj_M


def check_mouth_target(adj_M, s, t):
    """Check there are edges at the mouth, target"""
    c = np.sum(adj_M[[*s, *t], :], axis=1)
    bad = np.any(c == 0)
    check = not bad
    return check


# ----------------------------------------------------------------
#           Modifying network
# ----------------------------------------------------------------
def evolve_edge(edge_no, edge_df, edge_ref_df, adj, disp=False):
    """
       Evolution step to add/remove edge, assumes binary.
       Edges after pruning are renumbered (continuous numbering)
    """
    edge_ref = edge_ref_df.index
    test_edge = edge_ref[edge_no]
    # ri, rj is the end nodes of the edge being cut
    ri, rj = edge_ref_df.loc[test_edge][['i', 'j']]
    new_edge_df = edge_df.copy()
    new_adj = np.copy(adj)
    if test_edge in edge_df.index:
        k = 0
        new_edge_df = new_edge_df.drop(test_edge)
        # if disp:
        #     print(f'removed {test_edge}')
    else:
        k = 1
        ref_entry = edge_ref_df.loc[test_edge]
        new_edge_df = new_edge_df.append(ref_entry)
        # if disp:
        #     print(f'added {test_edge}')
    new_adj[ri, rj] = k
    new_adj[rj, ri] = k
    return new_edge_df, new_adj


def evolve_adj(edge_no, edge_dict, adj):
    """assumes binary"""
    new_adj = adj.copy()
    i, j = edge_dict[edge_no]
    val = np.abs(new_adj[i, j] - 1)
    new_adj[i, j] = val
    new_adj[j, i] = val
    return new_adj


def restore_net(points_ref, adj_ref, edge_ref_df):
    """Revert to un-evolved network"""
    points = points_ref.copy()
    adj = adj_ref.copy()
    edge_df = edge_ref_df.copy()
    return points, adj, edge_df


def get_rand_edge(adj, adj_ref):
    num_edges = np.shape(adj)[0]
    code = net.adj2barcode(adj, adj_ref)
    edge_no = np.random.randint(0, num_edges)
    while code[edge_no] == 0:
        edge_no = np.random.randint(0, num_edges)
    return edge_no


def copy_adj_neighbors(adj, node, adj_temp):
    """copy a node's neighbors from the template"""
    n_list = np.argwhere(adj_temp[node, :] > 0)
    adj[node, n_list] = adj_temp[node, n_list]
    adj[n_list, node] = adj_temp[n_list, node]
    return adj
