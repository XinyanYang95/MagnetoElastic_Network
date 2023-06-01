import sys
import os
import glob
import os.path as path
import time

# import sympy as sym
# from sympy import *
import scipy
import scipy as sp
from scipy import optimize
from scipy import spatial

import pickle as pkl

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as plticker

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

#-----------------------------define magnet variables-----------------------------------
sysname = "4cm-s"
origin_net_info = pkl.load(open('origin_net_info-'+sysname, 'rb')) # start equilibration from initial config.
# origin_net_info = pkl.load(open('origin_net_info-'+sysname+'-eq', 'rb')) # start pulling after equilibration
#[MAG]
magr = 1/8*2.54 # 1/16 inch magnet (cm) [RADIUS]
tm = 1/8*25.4   # in mm [HIGHT]
Br = 1.31 # in T (~N42) [FLUX]
mag_mass = 0.754/1000  # in kg (from K&J magnet spec.s) [MASS]
#[MAG]

mu0 = 4*jnp.pi*1e-7    # permeability of vacuum (H/m), H = J/A2
M = Br/mu0    # magnetization (A/m)
# magnet shape (assume height and width are both 1/16 in.)
Rm = magr*10    # in mm
V = (tm/1000)*jnp.pi*(Rm/1000)**2    # bond magnet volume (m3)
mag_amp = M*V  # magnet moment (A*m2)  
# moment of inertia of cylinder: I = 1/2*m*R^2
mag_J = 0.5*mag_mass*(magr/100)**2 # in kg*m^2

conn = origin_net_info['connec']
rots = origin_net_info['rots']
poss = jnp.hstack([origin_net_info['points'], jnp.zeros([len(rots),1])])
lo = origin_net_info['df'][['d']].to_numpy().astype(float)
netlen = lo + 2*magr*0
magrs = jnp.array([magr]*len(poss))
magts = jnp.array([tm]*len(poss))/10.0 #in cm
tms = jnp.array([tm]*len(poss))/10.0 #in cm
Brs = jnp.array([Br]*len(poss))
# kstiff = origin_net_info['df'][['k']].iloc[0].to_numpy().astype(float)[0]
kstiff = 0
#---------------------------predefine system variables----------------------------------
# [SYS]
eps=3e-2*20.0 # [mageps = eps # LJ 12-6 energy well depth (for magnetic dipole-dipole repulsion calc), float (in J).]
k=kstiff #[bar stiffness]
dc=magr*1.0 #[bar width]
krep=50000.0 #[stiffness for harmonic sterip repulsion]
kang=1000.0 #[stiffness for harmonic angle]
kwallrep = 50000.0 #[stiffness for harmonic boundary wall]
magcut="all" # all magnets are considered
# [SYS]

# system nature
magnum = len(magrs) # number of points/end dipoles, int.
bondnum = len(conn) # number of bonds/elastic springs, int.
mag_rs = magrs # magnet radius array, 1D array [magnum,] (in cm).
mag_ts = magts # magnet thickness array, 1D array [magnum,] (in cm)
mag_ms = Brs/mu0 * (magts/100)*jnp.pi*(magrs/100)**2 # magnetic strength array, 1D array [magnum,] (in A*m2).
mag_phis = rots # end dipole orientation array, 1D array [magnum,] (in rad).
bond_connec = conn # connectivity matrix, 2D array, [bondnum, 2].
mag_ns = jnp.vstack([jnp.cos(mag_phis), jnp.sin(mag_phis), jnp.zeros(magnum)]).T # stacked dipole unit vector, 2D array [magnum, 3].
mag_poss = poss # end dipole position array, 2D array [magnum, 3], the 3rd col. are zeros (in cm).
bondlo = lo # spring original length array, 1D array [bondnum,] (in cm).
mageps = eps # LJ 12-6 energy well depth (for magnetic dipole-dipole repulsion calc), float (in J).
k = k # elastic bond stiffness, float (in N/m).
k_rep = krep # elastic repulsive bond stiffness (for bond crossing penalty calc), float (in N/m).
k_wallrep = kwallrep # elastic repulsive wall stiffness (for boundary crossing penalty calc), float (in N/m).
bondwidth = dc # half spring thinkness (bond critical distance for checking bond crossing), i.e., repulsive radius for bond + magnet radius, float (in cm).
k_ang= kang # harmonic angle stiffness (for end dipole rotation penalty calc), float (in J). 
magcut = magcut # cutoff distance for magnetic interaction, float (in cm).

# elvoving properties
maglist = [] # create an attribute for saving magnetic pair in effect (within cutoff distance).
for i in range(magnum):
    for j in range(i+1, magnum):
        maglist.append([i,j])
maglist_ori = maglist.copy()
new_poss = mag_poss
new_rots = mag_phis
cmap = [] # svae mag-mag distance

# derived properites
# map node/magnet number to edge/bond number
bondconn_dic = np.zeros(magnum,dtype=np.int32)
for jj in range(bondnum):
    bondconn_dic[bond_connec[jj][0]] = jj
    bondconn_dic[bond_connec[jj][1]] = jj
# find dipole angle relative to bond in range [-pi, pi]
# use numpy for mutated array
magrots_re = np.zeros(np.shape(mag_phis))
for thisconn in bond_connec:
    temp = mag_poss[thisconn[1]] - mag_poss[thisconn[0]]
    temp = temp[:2]/jnp.linalg.norm(temp[:2])
    thisrot = jnp.arctan2(temp[1], temp[0]) # bond orentation angle relative to pos-x
    # dipole angle relative to bond
    magrots_re[thisconn[1]] = mag_phis[thisconn[1]] - thisrot
    magrots_re[thisconn[0]] = mag_phis[thisconn[0]] - thisrot
# convert to [-pi, pi] 
magrots_re = jnp.arctan2(jnp.sin(magrots_re), jnp.cos(magrots_re)) # end dipole relative orientation array, 1D array [magnum,] (in rad).

#---------------------define functions--------------------
@jax.jit
def dist(pt1, pt2, pt3): # x3,y3 is the point
    
    x1, y1 = pt1[:-1]
    x2, y2 = pt2[:-1]
    x3, y3 = pt3[:-1]
    
    px = x2-x1
    py = y2-y1

    norm = px*px + py*py
    u =  ((x3 - x1) * px + (y3 - y1) * py) / jnp.float32(norm)
    u=jnp.where(u>1,1,u)
    u=jnp.where(u<0,0,u)
        
    x = x1 + u * px
    y = y1 + u * py
    dx = x - x3
    dy = y - y3
    foot = jnp.array([x, y, 0])
    dist = (dx*dx + dy*dy)**.5
    return u, dist, foot    

def dist_map(index, bond_1=0, bond_2=0,pos=0):
    pos_bond_1=pos[bond_1]
    pos_bond_2=pos[bond_2]
    pos_bond_3=pos[index]
    u, ds, foot = dist(pos_bond_1, pos_bond_2, pos_bond_3) 
    return u, ds, foot

def init_update_maglist():

    def update_maglist(maglist,udofs):
        return maglist
    if magcut=="all":
        maglist = maglist_ori.copy()
        maglist=jnp.array(maglist)
    return maglist,update_maglist

def Energy_bond_single_static(lo, newpos1, newpos2, k=1.0):
    """Get steric repulsive energy due to the exclusive volume of magnets.
       
    PARAMETERS
        l: deformed length, float (in cm).
        lo: original length, float (in cm).
        k: spring stiffness, float (in N/m)
       
    RETURN
        E_el: elastic bond energy, float (in J).
    """
    l = jnp.linalg.norm(newpos1-newpos2)
    E_el = 0.5*k*(l/100-lo/100)**2
    return E_el

@jax.jit
def Energy_magnet_map(pair, mag_ms, new_rots, new_poss):
    mu0 = 4*jnp.pi*1e-7 
    r_vec12 = new_poss[pair[1]] - new_poss[pair[0]] #
    r_vec = r_vec12/100.0

    mag_m1=mag_ms[pair[0]]
    mag_m2=mag_ms[pair[1]]

    mag_phi1=new_rots[pair[0]]
    mag_phi2=new_rots[pair[1]]

    m1 = mag_m1*jnp.array([jnp.cos(mag_phi1), jnp.sin(mag_phi1), 0])
    m2 = mag_m2*jnp.array([jnp.cos(mag_phi2), jnp.sin(mag_phi2), 0])

    E_from_vec = mu0/(jnp.pi*4)*(jnp.inner(m1, m2)/jnp.linalg.norm(r_vec)**3  
                    - 3*jnp.inner(m1, r_vec)*jnp.inner(m2, r_vec)/jnp.linalg.norm(r_vec)**5) # in J
    return E_from_vec

def init_Energy_mag():
    @jax.jit
    def Energy_mag(udofs,maglist=None):
        """ A method to calculate the magnetic energy of the magneto-elastic network after deformations, udofs.

        PARAMETERS 
            udofs: displacement of free DOFs, 1D array [#freedofs,] (x1, y1, theta1, x2, y2, theta2), 
                    1D array [6,] (in cm).
    
        RETURN 
            E_mag_sys: magnetic energy of the system, float (in J).

        """
        # update position
        udofs_arr = udofs.reshape([magnum, 3])
        udofs_xyz = jnp.hstack([udofs_arr[:,:-1], jnp.zeros([magnum,1])])
        new_poss = mag_poss + udofs_xyz
        
        # update dipole orientations
        udofs_rot = udofs_arr[:, -1]
        new_rots = mag_phis + udofs_rot

        # Magnetic energy
        E_mag_sys = 0
        Energy_magnet_vmap_fn=partial(Energy_magnet_map,mag_ms=mag_ms,new_rots=new_rots,new_poss=new_poss)
        Energy_magnet_vmap=jax.vmap(Energy_magnet_vmap_fn)
        
        E_mag_sys=jnp.sum(Energy_magnet_vmap(maglist))
        
        return E_mag_sys
    return Energy_mag

@jax.jit
def Energy_mag_steric_repulsion_map(pair, magr, pos, eps=mageps):
    
    # eps: energy well depth
    # critical c2c distance between two magnets
    magr1=magr[pair[0]]
    magr2=magr[pair[1]]

    pos1=pos[pair[0]]
    pos2=pos[pair[1]]

    cridis = magr1+magr2 # in cm
    distemp = jnp.linalg.norm(pos1-pos2) # in cm
    # sig: zero-energy position
    # # LJ12-6
    # sig = cridis/2**(1/6) # in cm
    # E_rep=jnp.where(distemp < cridis, 4*eps*((sig/distemp)**12-(sig/distemp)**6) + eps, 0)
    # LJ50-49
    sig = cridis*49/50 # in cm
    E_rep=jnp.where(distemp < cridis, 50*(50/49)**49*eps*((sig/distemp)**50-(sig/distemp)**49) + eps, 0)
    return E_rep

# NEW!!!
@jax.jit
def CalcAngle_map(new_rots,bondconn_dic_index,srot_spring):
    newrots_re = new_rots - srot_spring[bondconn_dic_index]
    return newrots_re

@jax.jit
def calc_newsrot_spring(bond_connec,poss):
    temp = poss[bond_connec[1]] - poss[bond_connec[0]]
    temp = temp/jnp.linalg.norm(temp)
    rot_spring=jnp.arctan2(temp[1], temp[0])
    return rot_spring

def init_Energy_part():
    @jax.jit
    def ElasticBond_map(bond_connec,bondlo,pos):
        atom1=bond_connec[0]
        atom2=bond_connec[1]
        E_el_sys = Energy_bond_single_static(bondlo, pos[atom1], pos[atom2], k) 
        return E_el_sys

    @jax.jit
    def Energy_part(udofs,maglist=None):
        """ A method to calculate part of the total energy of the magneto-elastic network after deformations, udofs.
        bond-bond repulsion and magnetic interaction are not included. 
        (cause that former force is defined externally instead of from the derivative of total energy,
        the latter is calculated parallelly on different cores)

        PARAMETERS 
            udofs: displacement of free DOFs, 1D array [#freedofs,] (x1, y1, theta1, x2, y2, theta2, ...), 
                    1D array [6,] (in cm).
        
        RETURN 
            E_tot: sum of magnetic repulsive energy, elastic bond and angle energy, float (in J).

        """
        start = time.time()
        # update position
        udofs_arr = udofs.reshape([magnum, 3])
        udofs_xyz = jnp.hstack([udofs_arr[:,:-1], jnp.zeros([magnum,1])])
        new_poss = mag_poss + udofs_xyz
        
        # update dipole orientations
        udofs_rot = udofs_arr[:, -1]
        new_rots = mag_phis + udofs_rot

        # Steric repulsion
        E_mag_rep_sys = 0
        Energy_mag_steric_repulsion_vmap_fn=partial(Energy_mag_steric_repulsion_map,magr=mag_rs,pos=new_poss,eps=eps)
        Energy_mag_steric_repulsion_vmap=jax.vmap(Energy_mag_steric_repulsion_vmap_fn)
        E_mag_rep_sys=jnp.sum(Energy_mag_steric_repulsion_vmap(maglist))     
        #print(f'Steric repulsion: {(time.time() - start):.3f} sec.')
        
        # NEW!!!
        # Elastic bond  
        #E_el_sys = jnp.sum(jnp.array([Energy_bond_single_static(bondlo[pp], new_poss[bond_connec[pp][0]], 
        #                    new_poss[bond_connec[pp][1]], k) for pp in range(len(bond_connec))]))
        ElasticBond_vmap_fn=partial(ElasticBond_map,pos=new_poss)
        ElasticBond_vmap=jax.vmap(ElasticBond_vmap_fn)
        E_el_sys=jnp.sum(ElasticBond_vmap(bond_connec,bondlo)) 
        #print(f'Elastic bond : {(time.time() - start):.3f} sec.')
        # harmonic angle energy
        # calculate new rotation of springs 
        newsrot_spring_vmap_fn=partial(calc_newsrot_spring,poss=new_poss)
        newsrot_spring_vmap=jax.vmap(newsrot_spring_vmap_fn)
        newsrot_spring=newsrot_spring_vmap(bond_connec)
        
        # NEW!!!
        # relative angle between the spring and the magnetic dipole
        #newrots_re = jnp.array([new_rots[nn] - newsrot_spring[bondconn_dic[nn]] for nn in range(magnum)])
        CalcAngle_vmap_fn=partial(CalcAngle_map,srot_spring=newsrot_spring)
        CalcAngle_vmap=jax.vmap(CalcAngle_vmap_fn)
        newrots_re=CalcAngle_vmap(new_rots,bondconn_dic)
        # convert to [-pi, pi] 
        newrots_re = jnp.arctan2(jnp.sin(newrots_re), jnp.cos(newrots_re))
        temp2 = newrots_re - magrots_re
        temp2 = jnp.arctan2(jnp.sin(temp2), jnp.cos(temp2))
        energy_ang = k_ang*temp2**2 # temp2 is now in [-pi, pi]
        E_ang_sys = jnp.sum(energy_ang)
        #print(f'E_ang_sys : {(time.time() - start):.3f} sec.')
        E_tot = E_mag_rep_sys + E_el_sys + E_ang_sys
        
        return E_tot
    return Energy_part

def init_Energy_bond_cross():
    # hard coded
    # entry i: save a list for atom i of other atoms nonbonded to i
    def create_neigh_bonds():
        neigh_bonds=[]
        for i in range(magnum):
            neigh_bonds_this=[]
            for bondi in bond_connec:
                if i not in bondi:
                    neigh_bonds_this.append(bondi)
            neigh_bonds.append(neigh_bonds_this)
        return jnp.array(neigh_bonds)

    neighbor_bonds=create_neigh_bonds() # lists of bonded pairs

    @jax.jit
    def calc_bond_repulsive_energy(bond_connec,particle_index=None,pos=None):
        u, ds, foot = dist(pos[bond_connec[0]], pos[bond_connec[1]], pos[particle_index]) 
        return u,ds,foot

    @jax.jit
    def calc_node_energy(idx,neigh_bonds,pos=None):
        calc_bond_repulsive_energy_vmap_fn=partial(calc_bond_repulsive_energy,particle_index=idx,pos=pos)
        calc_bond_repulsive_energy_vmap=jax.vmap(calc_bond_repulsive_energy_vmap_fn)
        u, ds, foot=calc_bond_repulsive_energy_vmap(neigh_bonds)
        thisf = k_rep*(bondwidth*0.01-ds*0.01)*(u>0)*(u<1)
        thisf = thisf*(thisf>0) 
        thise = 0.5*thisf*(bondwidth*0.01-ds*0.01) # in J
        return thise

    @jax.jit
    def Energy_bond_cross(udofs):
        # update position
        udofs_arr = udofs.reshape([magnum, 3])
        udofs_xyz = jnp.hstack([udofs_arr[:,:-1], jnp.zeros([magnum,1])])
        new_poss = mag_poss + udofs_xyz

        # node force
        calc_node_energy_vmap_fn=partial(calc_node_energy,pos=new_poss)
        calc_node_energy_vmap=jax.vmap(calc_node_energy_vmap_fn)
        node_energy = calc_node_energy_vmap(jnp.arange(magnum,dtype=jnp.int32),neighbor_bonds)        

        return jnp.sum(node_energy)

    return Energy_bond_cross

def init_Energy_boundary_cross():
    
    @jax.jit
    def calc_boundary_energy(idx, bs=None, pos=None):
        dx_bc1 = (bs[0] - pos[idx][0])*0.01 # in m
        dx_bc2 = (bs[2] - pos[idx][0])*0.01 # in m
        dy_bc1 = (bs[1] - pos[idx][1])*0.01 # in m
        dy_bc2 = (bs[3] - pos[idx][1])*0.01 # in m
        tempex = 0.5*k_wallrep*((dx_bc1*(dx_bc1>0))**2 + (dx_bc2*(dx_bc2<0))**2) # in J
        tempey = 0.5*k_wallrep*((dy_bc1*(dy_bc1>0))**2 + (dy_bc2*(dy_bc2<0))**2) # in J
        Eb = tempex + tempey
        return Eb

    @jax.jit
    def Energy_boundary_cross(udofs,newbs):
        """ A method to calculate the boundary repulsive energy of the magneto-elastic network after deformations, 
            udofs.

            PARAMETERS 
                udofs: displacement of free DOFs, 1D array [#freedofs,] (x1, y1, theta1, x2, y2, theta2, ...), 
                       (in cm, cm, rad, cm, cm, rad, ...).

            RETURN 
                Total boundary crossing energy, float [J].

        """
        # update position
        udofs_arr = udofs.reshape([magnum, 3])
        udofs_xyz = jnp.hstack([udofs_arr[:,:-1], jnp.zeros([magnum,1])])
        new_poss = mag_poss + udofs_xyz

        calc_boundary_energy_vmap_fn=partial(calc_boundary_energy, bs=newbs, pos=new_poss)
        calc_boundary_energy_vmap=jax.vmap(calc_boundary_energy_vmap_fn)
        boundary_energy=calc_boundary_energy_vmap(jnp.arange(magnum,dtype=jnp.int32))

        return jnp.sum(boundary_energy)
    
    return Energy_boundary_cross

@jax.jit
def Force_magnet_map(pair, mag_ms, new_rots, new_poss):
    mu0 = 4*jnp.pi*1e-7 
    r_vec12 = new_poss[pair[0]] - new_poss[pair[1]] #
    r_vec = r_vec12/100.0

    mag_m1=mag_ms[pair[0]]
    mag_m2=mag_ms[pair[1]]

    mag_phi1=new_rots[pair[0]]
    mag_phi2=new_rots[pair[1]]

    m1 = mag_m1*jnp.array([jnp.cos(mag_phi1), jnp.sin(mag_phi1), 0])
    m2 = mag_m2*jnp.array([jnp.cos(mag_phi2), jnp.sin(mag_phi2), 0])

    F_from_vec1 = 3*mu0/(jnp.pi*4*jnp.linalg.norm(r_vec)**5)*(jnp.inner(m1, r_vec)*m2 + jnp.inner(m2, r_vec)*m1 
        + jnp.inner(m1, m2)*r_vec - 5*jnp.inner(m1, r_vec)*jnp.inner(m2, r_vec)*r_vec/jnp.linalg.norm(r_vec)**2)

    tm1 = mu0/(jnp.pi*4)*(3*jnp.inner(r_vec, m2)*r_vec/jnp.linalg.norm(r_vec)**5 - m2/jnp.linalg.norm(r_vec)**3) # on dipole m1
    tm1 = jnp.cross(m1, tm1)

    tm2 = mu0/(jnp.pi*4)*(3*jnp.inner(r_vec, m1)*r_vec/jnp.linalg.norm(r_vec)**5 - m1/jnp.linalg.norm(r_vec)**3) # on m2
    tm2 = jnp.cross(m2, tm2)

    forces = jnp.array([[F_from_vec1[0], F_from_vec1[1], tm1[2]],[-F_from_vec1[0], -F_from_vec1[1], tm2[2]]])
    
    tempspace = jnp.zeros(jnp.shape(new_poss))
    tempspace = tempspace.at[pair].set(forces)
    
    return tempspace

def init_Force_mag():
    @jax.jit
    def Force_mag(udofs,maglist=None):
        """ A method to calculate the magnetic energy of the magneto-elastic network after deformations, udofs.

        PARAMETERS 
            udofs: displacement of free DOFs, 1D array [#freedofs,] (x1, y1, theta1, x2, y2, theta2), 
                    1D array [6,] (in cm).
        
        RETURN 
            F_mag_sys: magnetic forces of the system, 1D array [#freedofs,] (in N, N, 0, N, N, 0, ...).

        """
        # update position
        udofs_arr = udofs.reshape([magnum, 3])
        udofs_xyz = jnp.hstack([udofs_arr[:,:-1], jnp.zeros([magnum,1])])
        new_poss = mag_poss + udofs_xyz
        
        # update dipole orientations
        udofs_rot = udofs_arr[:, -1]
        new_rots = mag_phis + udofs_rot
        
        # Magnetic force
        Force_magnet_vmap_fn=partial(Force_magnet_map,mag_ms=mag_ms,new_rots=new_rots,new_poss=new_poss)
        Force_magnet_vmap=jax.vmap(Force_magnet_vmap_fn)   
        F_mag_sys=jnp.sum(Force_magnet_vmap(maglist), axis=0)
        
        return F_mag_sys.flatten()
    return Force_mag

@jax.jit
def Force_mag_steric_repulsion_map(pair, magr, pos, eps=3.0e-2):
    # critical c2c distance between two magnets
    magr1=magr[pair[0]]
    magr2=magr[pair[1]]

    pos1=pos[pair[0]]
    pos2=pos[pair[1]]

    cridis = magr1+magr2 # in cm

    distemp = jnp.linalg.norm(pos1-pos2) # in cm
    n12 = (pos1-pos2)/distemp
    
    # # LJ 12-6
    # # zero-energy position
    # sig = cridis/2**(1/6) # in cm
    # F_rep=jnp.where(distemp < cridis, 24*eps*(-2*sig**12/distemp**13+sig**6/distemp**7)*100*n12, 0*n12)

    # LJ 50-49
    # zero-energy position
    sig = cridis*49/50 # in cm
    F_rep=jnp.where(distemp < cridis, 50*(50/49)**49*eps*(-sig**50/distemp**51+sig**49/distemp**50)*100*n12, 0*n12)
    
    tempspace = jnp.zeros([magnum,3])
    tempspace = tempspace.at[pair[0]].set(-F_rep)
    tempspace = tempspace.at[pair[1]].set(F_rep)

    return tempspace

def init_Force_mag_rep():
    @jax.jit
    def Force_mag_rep(udofs,maglist=None):
        """ A method to calculate part of the total energy of the magneto-elastic network after deformations, udofs.
        bond-bond repulsion and magnetic interaction are not included. 
        (cause that former force is defined externally instead of from the derivative of total energy,
        the latter is calculated parallelly on different cores)

        PARAMETERS 
            udofs: displacement of free DOFs, 1D array [#freedofs,] (x1, y1, theta1, x2, y2, theta2, ...), 
                    1D array [6,] (in cm).
        
        RETURN 
            F_mag_rep_sys: magnetic repulsive forces, 1D array [#freedofs,] (in N, N, 0, N, N, 0, ...).
            
        """
        # update position
        udofs_arr = udofs.reshape([magnum, 3])
        udofs_xyz = jnp.hstack([udofs_arr[:,:-1], jnp.zeros([magnum,1])])
        new_poss = mag_poss + udofs_xyz

        # Steric repulsion
        Force_mag_steric_repulsion_vmap_fn=partial(Force_mag_steric_repulsion_map,magr=mag_rs,pos=new_poss,eps=eps)
        Force_mag_steric_repulsion_vmap=jax.vmap(Force_mag_steric_repulsion_vmap_fn)
        F_mag_rep_sys=jnp.sum(Force_mag_steric_repulsion_vmap(maglist), axis=0)     
                
        return F_mag_rep_sys.flatten()
    return Force_mag_rep

def Force_spring(udofs):
    # update position
    udofs_arr = udofs.reshape([magnum, 3])
    udofs_xyz = jnp.hstack([udofs_arr[:,:-1], jnp.zeros([magnum,1])])
    new_poss = mag_poss + udofs_xyz

    temp = new_poss[bond_connec[:,0]]-new_poss[bond_connec[:,1]]
    bond_l = jnp.linalg.norm(temp,axis=1)
    bond_l = bond_l[:,None] 
    bond_n = temp/bond_l

    bond_dl = bond_l-lo
    F_el = k*bond_dl/100*bond_n # in N, derived from Eang=0.5*k*dl^2

    forces = jnp.zeros([magnum,3])
    forces = forces.at[bond_connec[:,0]].set(-F_el)
    forces = forces.at[bond_connec[:,1]].set(F_el)
    
    return forces.flatten()

def Force_angle(udofs):
    # update position
    udofs_arr = udofs.reshape([magnum, 3])
    udofs_xyz = jnp.hstack([udofs_arr[:,:-1], jnp.zeros([magnum,1])])
    new_poss = mag_poss + udofs_xyz

    # update dipole orientations
    udofs_rot = udofs_arr[:, -1]
    new_rots = mag_phis + udofs_rot
    
    # harmonic angle force
    # part1: end torque
    # calculate new rotation of springs 
    newsrot_spring_vmap_fn=partial(calc_newsrot_spring,poss=new_poss)
    newsrot_spring_vmap=jax.vmap(newsrot_spring_vmap_fn)
    newsrot_spring=newsrot_spring_vmap(bond_connec)
    # NEW!!!
    # relative angle between the spring and the magnetic dipole
    CalcAngle_vmap_fn=partial(CalcAngle_map,srot_spring=newsrot_spring)
    CalcAngle_vmap=jax.vmap(CalcAngle_vmap_fn)
    newrots_re=CalcAngle_vmap(new_rots,bondconn_dic)
    # convert to [-pi, pi] 
    newrots_re = jnp.arctan2(jnp.sin(newrots_re), jnp.cos(newrots_re))
    temp2 = newrots_re - magrots_re
    temp2 = jnp.arctan2(jnp.sin(temp2), jnp.cos(temp2))
    force_ang = -k_ang*temp2*2 # temp2 is now in [-pi, pi], in N*m, derived from Eang=k*ang^2
    
    # part2: Fx, Fy to balance the end torques of each bar
    temp = new_poss[bond_connec[:,0]]-new_poss[bond_connec[:,1]]
    bond_l = jnp.linalg.norm(temp,axis=1)
    bond_n = temp/bond_l[:,None]
    bond_n = jnp.hstack([-bond_n, bond_n]).reshape([magnum,3])
    rot90 = jnp.array([[0, -1, 0],[1, 0, 0],[0,0,1]])
    bond_nv = jnp.matmul(rot90, bond_n.T).T
    endtorques = jnp.sum(force_ang[bond_connec], axis=1)
    resultantF = endtorques/bond_l*100 # in N
    resultantF = resultantF[:,None]
    resultantF = resultantF.repeat(2)
    force_trans = resultantF[:,None]*bond_nv
    
    F_ang_sys = np.concatenate((force_trans[:,:2], force_ang.T[:,None]), axis=1)

    return F_ang_sys.flatten()

def init_Force_bond_cross():
    # hard coded
    # entry i: save a list for atom i of other atoms nonbonded to i
    def create_neigh_bonds():
        neigh_bonds=[]
        for i in range(magnum):
            neigh_bonds_this=[]
            for bondi in bond_connec:
                if i not in bondi:
                    neigh_bonds_this.append(bondi)
            neigh_bonds.append(neigh_bonds_this)
        return jnp.array(neigh_bonds)

    neighbor_bonds=create_neigh_bonds() # lists of bonded pairs

    @jax.jit
    def calc_bond_repulsive_force(bond_connec,particle_index=None,pos=None):
        u, ds, foot = dist(pos[bond_connec[0]], pos[bond_connec[1]], pos[particle_index]) 
        return u,ds,foot

    @jax.jit
    def calc_node_force(idx,neigh_bonds,pos=None):
        calc_bond_repulsive_force_vmap_fn=partial(calc_bond_repulsive_force,particle_index=idx,pos=pos)
        calc_bond_repulsive_force_vmap=jax.vmap(calc_bond_repulsive_force_vmap_fn)
        u, ds, foot=calc_bond_repulsive_force_vmap(neigh_bonds)
        thisf = k_rep*(bondwidth*0.01-ds*0.01)*(u>0)*(u<1)
        thisf = thisf*(thisf>0) ###
        unit_vec = pos[idx] - foot # pointing from foot to new_poss[pt], 1D array [3,]
        norm_vec=jnp.linalg.norm(unit_vec,axis=1)
        unit_vec = unit_vec/norm_vec[:,None]
        thisf_vec = unit_vec*thisf[:,None]
        L = jnp.linalg.norm(pos[neigh_bonds[:,0]]-foot,axis=1)/jnp.linalg.norm(pos[neigh_bonds[:,0]]-pos[neigh_bonds[:,1]],axis=1) ###
        return thisf_vec, L

    @jax.jit
    def calc_bond_force(idx, neighbor_bonds, thisf_vec=None, L=None):
        tempspace = jnp.zeros([magnum, 3])
        tempspace = tempspace.at[neighbor_bonds[:,1]].set(-thisf_vec[idx]*L[idx,:,None])
        tempspace = tempspace.at[neighbor_bonds[:,0]].set(-thisf_vec[idx]*(1-L)[idx,:,None])
        return tempspace
    

    @jax.jit
    def Force_bond_cross(udofs):
        """ A method to calculate the bond repulsive force of the magneto-elastic network after deformations, udofs.

        PARAMETERS 
            udofs: displacement of free DOFs, 1D array [#freedofs,] (x1, y1, theta1, x2, y2, theta2, ...), 
                    (in cm, cm, rad, cm, cm, rad, ...).

        RETURN 
            bond crossing forces on nodes, 1D array [#freedofs,] (in N, N, null, N, N, null, ...).

        """
        # update position
        udofs_arr = udofs.reshape([magnum, 3])
        udofs_xyz = jnp.hstack([udofs_arr[:,:-1], jnp.zeros([magnum,1])])
        new_poss = mag_poss + udofs_xyz

        # node force
        calc_node_force_vmap_fn=partial(calc_node_force,pos=new_poss)
        calc_node_force_vmap=jax.vmap(calc_node_force_vmap_fn)
        node_force, L = calc_node_force_vmap(jnp.arange(magnum,dtype=jnp.int32),neighbor_bonds)        

        # bond force
        calc_bond_force_vmap_fn=partial(calc_bond_force, thisf_vec=node_force, L=L)
        calc_bond_force_vmap=jax.vmap(calc_bond_force_vmap_fn)
        bond_force=calc_bond_force_vmap(jnp.arange(magnum,dtype=jnp.int32), neighbor_bonds)

        return (jnp.sum(node_force,axis=1) + jnp.sum(bond_force, axis=0)).flatten()

    return Force_bond_cross

def init_Force_boundary_cross():
    
    @jax.jit
    def calc_boundary_force(idx, bs=None, pos=None):
        dx_bc1 = (bs[0] - pos[idx][0])*0.01 # in m
        dx_bc2 = (bs[2] - pos[idx][0])*0.01 # in m
        dy_bc1 = (bs[1] - pos[idx][1])*0.01 # in m
        dy_bc2 = (bs[3] - pos[idx][1])*0.01 # in m
        tempfx = k_wallrep*(dx_bc1*(dx_bc1>0) + dx_bc2*(dx_bc2<0)) # in N
        tempfy = k_wallrep*(dy_bc1*(dy_bc1>0) + dy_bc2*(dy_bc2<0)) # in N
        Fb = jnp.array([tempfx, tempfy, 0])
        return Fb

    @jax.jit
    def Force_boundary_cross(udofs, newbs):
        """ A method to calculate the boundary repulsive force of the magneto-elastic network after deformations, 
            udofs.

            PARAMETERS 
                udofs: displacement of free DOFs, 1D array [#freedofs,] (x1, y1, theta1, x2, y2, theta2, ...), 
                       (in cm, cm, rad, cm, cm, rad, ...).

            RETURN 
                boundary_force: boundary crossing forces, 1D array [#freedofs,] (in N, N, 0, N, N, 0, ...).

        """

        # update position
        udofs_arr = udofs.reshape([magnum, 3])
        udofs_xyz = jnp.hstack([udofs_arr[:,:-1], jnp.zeros([magnum,1])])
        new_poss = mag_poss + udofs_xyz

        calc_boundary_force_vmap_fn=partial(calc_boundary_force, bs=newbs, pos=new_poss)
        calc_boundary_force_vmap=jax.vmap(calc_boundary_force_vmap_fn)
        boundary_force=calc_boundary_force_vmap(jnp.arange(magnum,dtype=jnp.int32))
        return boundary_force.flatten()
    return Force_boundary_cross

# #---------------- AUG are defined here - Equilibration ---------------------------[AUG]
dt = 2e-7 # timestep [s]
t_tot = 3.0 # total simulation time [s]

# damping
zeta = 0.5
cc = 50 # Hz

operation = 'eq'
neq = 0 # neq = 0, 1, 2, ...
outdt = dt*1000 # output u, v, a every outdt [s]
savealongprocess=True # if Ture, save .pkl along running; if False, only save one .pkl when the simulation is done
savenum = 20 # total number of trajectory to be saved (overwrite the previously saved one)

# B.C.s
squeezebox = False
ini_box = 20.0 # inital box size 20x20
fin_box = 20.0 # final box size 10x10
duration_decr = 2.0 # time for dreasing from intial box size to final box size (i.e., boundary decreases from 20 to 10 in 1.0s)
starttime_for_decr = 4.0 # time when boundary starts decreasing
Fext = jnp.zeros(3*len(poss)) # define external forces (if any)

kbt = 0 # temperature

# vSMD
vSMD = False
#[SPRING]
sl0 = 0.02 #
pullratexl = 0. # m/s left
pullratexr = 0. # m/s right 
pullrateyt = 0. # m/s top
pullrateyb = 0. # m/s bottom
sk_xr =  0. # N/m
sk_xl = 0. # N/m
sk_yt = 0.
sk_yb = 0.
#[SPRING]

# simulation control (losing particles or not)
stop_in_process = False
checkbound = 100

##----------------Simulation AUG are defined here - Uniaxial Pulling ---------------------------[AUG]
# dt = 2e-7 # timestep [s]
# t_tot = 30.0 # total simulation time [s]

# # damping
# zeta = 0.5
# cc = 50 # Hz

# operation = 'biaxial'
# neq = 0 # neq = 0, 1, 2, ...
# outdt = dt*1000 # output u, v, a every outdt [s]
# savealongprocess=True # if Ture, save .pkl along running; if False, only save one .pkl when the simulation is done
# savenum = 200 # total number of trajectory to be saved (overwrite the previously saved one)

# # B.C.s
# squeezebox = False
# ini_box = 200.0 # inital box size
# fin_box = 200.0 # final box size
# duration_decr = 2.0 # time for dreasing from intial box size to final box size (i.e., boundary decreases from 10 to 7 in 1.0s)
# starttime_for_decr = 0.0 # time when boundary starts decreasing
# Fext = jnp.zeros(3*len(poss)) # define external forces (if any)

# kbt = 0 # temperature

# # vSMD
# vSMD = True
# #[SPRING]
# sl0 = 0.02 #
# pullratexl = -0.002 # m/s left
# pullratexr = 0.002 # m/s right 
# pullrateyt = 0.002 # m/s top
# pullrateyb = 0.002 # m/s bottom
# sk_xr =  5000 # N/m
# sk_xl = -5000 # N/m
# sk_yt = 5000
# sk_yb = -5000
# #[SPRING]

# # simulation control (losing particles or not)
# stop_in_process = True
# checkbound = 100
#----------------------------------------------------------------------------
# (1) squeeze box strategy (harmonic wall potential)
ini_bs = np.array([-1.0, -1.0, 1.0, 1.0])*ini_box
fin_bs = np.array([-1.0, -1.0, 1.0, 1.0])*fin_box
startstep_for_decr = int(starttime_for_decr//dt)
endstep_for_decr = int((starttime_for_decr+duration_decr)//dt)
nstep_for_decr = int(duration_decr//dt)
decrx_per_step = ((ini_bs[2]-ini_bs[0])-(fin_bs[2]-fin_bs[0]))/(nstep_for_decr*2.0)*squeezebox
decry_per_step = ((ini_bs[3]-ini_bs[1])-(fin_bs[3]-fin_bs[1]))/(nstep_for_decr*2.0)*squeezebox
decr_per_step = np.array([decrx_per_step, decry_per_step, -decrx_per_step, -decry_per_step])

# (2) vSMD
##-----pick particles to pull
moveatoms_xr = np.array([183, 184, 134, 135, 136, 76, 77, 78, 19, 20])
moveatoms_xr = moveatoms_xr -1
moveatoms_xl = np.array([165, 166, 107, 108, 109, 49, 50, 51, 2, 1])
moveatoms_xl = moveatoms_xl -1
moveatoms_yt = np.array(list(np.arange(167, 183))+[165, 166, 183, 184])
moveatoms_yt = moveatoms_yt -1
moveatoms_yb = np.array(list(np.arange(3, 19))+[1,2,19,20])
moveatoms_yb = moveatoms_yb -1

#-------based on eq config: right 10 to moveatoms_xr, left 10 to moveatoms_xl
# nbeads = 10
# moveatoms_xl = np.array(np.argsort(poss[:,0])[:nbeads])
# moveatoms_xr = np.array(np.argsort(poss[:,0])[::-1][:nbeads])
# moveatoms_yb = np.array(np.argsort(poss[:,1])[:nbeads])
# moveatoms_yt = np.array(np.argsort(poss[:,1])[::-1][:nbeads])
# moveatoms = np.hstack([moveatoms_xl, moveatoms_xr, moveatoms_yt, moveatoms_yb])

# spring velocities 
spring_vx = np.zeros(len(poss))
spring_vx[moveatoms_xr] = np.array([pullratexr]*len(moveatoms_xr))
spring_vx[moveatoms_xl] = np.array([pullratexl]*len(moveatoms_xl))
spring_vy = np.zeros(len(poss))
spring_vy[moveatoms_yt] = np.array([pullrateyt]*len(moveatoms_yt))
spring_vy[moveatoms_yb] = np.array([pullrateyb]*len(moveatoms_yb))
spring_v = np.vstack([spring_vx, spring_vy, np.zeros(len(poss))]).T # [mag_num, 3]

# spring stiffnesses
spring_kx = np.zeros(len(poss))
spring_kx[moveatoms_xl] = np.array([sk_xl]*len(moveatoms_xl)) 
spring_kx[moveatoms_xr] = np.array([sk_xr]*len(moveatoms_xr))
spring_ky = np.zeros(len(poss))
spring_ky[moveatoms_yt] = np.array([sk_yt]*len(moveatoms_yt)) 
spring_ky[moveatoms_yb] = np.array([sk_yb]*len(moveatoms_yb))
spring_k = np.vstack([spring_kx, spring_ky, np.zeros(len(poss))]).T # [mag_num, 3]

# initial length of springs 
spring_xl0 = np.zeros(len(poss))
spring_xl0[moveatoms_xr] = np.array([sl0]*len(moveatoms_xr))
spring_xl0[moveatoms_xl] = np.array([sl0]*len(moveatoms_xl))
spring_yl0 = np.zeros(len(poss))
spring_yl0[moveatoms_yt] = np.array([sl0]*len(moveatoms_yt))
spring_yl0[moveatoms_yb] = np.array([sl0]*len(moveatoms_yb))
spring_l0 = np.vstack([spring_xl0, spring_yl0, np.zeros(len(poss))]).T # [mag_num, 3]

# initial offset from atom postion to spring (the position of dummy atoms)
spring_xl0_off= np.zeros(len(poss))
spring_xl0_off[moveatoms_xr] = np.array([sl0]*len(moveatoms_xr))
spring_xl0_off[moveatoms_xl] = np.array([-sl0]*len(moveatoms_xl))
spring_yl0_off = np.zeros(len(poss))
spring_yl0_off[moveatoms_yt] = np.array([sl0]*len(moveatoms_yt))
spring_yl0_off[moveatoms_yb] = np.array([-sl0]*len(moveatoms_yb))
spring_l0_off = np.vstack([spring_xl0_off, spring_yl0_off, np.zeros(len(poss))]).T # [mag_num, 3]

#--------------------------------------------------------------------------
# linearly decrease boundary
def update_boundary(nstep):
        if nstep<startstep_for_decr:
            return ini_bs
        elif nstep<=endstep_for_decr:
            return ini_bs+decr_per_step*(nstep-startstep_for_decr)

        else:
            return fin_bs

# system variables calculated from given AUG 
magpro = jnp.array([mag_mass, mag_mass, mag_J]*len(poss)) # 1D array (ndofs,) [kg, kg, kg*m^2]*npts
dampc = jnp.array([mag_mass*(2*zeta)*cc*2*jnp.pi, 
                  mag_mass*(2*zeta)*cc*2*jnp.pi, 
                  mag_J*(2*zeta)*cc*2*jnp.pi]*len(poss))

if neq<1:
    restart=False
    restart_config={}
else:
    restart=True
    restart_config=pkl.load(open(f'dic_in_process_{operation}-{int(neq-1)}','rb'))
    print(f'Continue uniaxial pulling {operation}-{int(neq-1)}.')

nstep = int(1+t_tot/dt)
outfreq = int(outdt/dt)
npts = len(poss)
ndofs = 3*len(poss)

# initial condition (assume no initial velocity, no initial displacement)
if not restart:
    # uold = np.zeros(ndofs) # 1D array (ndofs,) [m, m, rad]*npts
    uold = np.zeros(ndofs) # 1D array (ndofs,) [m, m, rad]*npts (last frame)
    vold = np.zeros(ndofs) # 1D array (ndofs,) [m/s, m/s, rad/s]*npts
    # initialize array for output
    uout = [np.zeros(ndofs)] # 1D array (ndofs,) [m, m, rad]*npts
    vout = [np.zeros(ndofs)] # 1D array (ndofs,) [m/s, m/s, rad/s]*npts
    aout = [np.zeros(ndofs)] # 1D array (ndofs,) [m/s2, m/s2, rad/s2]*npts
    
    emagout = [] # in J
    etotout = [] # in J

    fbonds = []
    fbcs = []
    fints = []
    fsprings = [] 
    
    irange = (0, nstep-1)
else:
    uold = restart_config['u'][-1] # 1D array (ndofs,) [m, m, rad]*npts (last frame)
    vold = restart_config['v'][-1] # 1D array (ndofs,) [m/s, m/s, rad/s]*npts (last frame)
    # initialize array for output
    uout = restart_config['u'] # 2D array (nframes, ndofs)
    vout = restart_config['v'] 
    aout = restart_config['a']
    
    emagout = restart_config['emag'] # 1D array (nframes,)
    etotout = restart_config['etot']

    fbonds = restart_config['fbonds']
    fbcs = restart_config['fbcs']
    fints = restart_config['fints']
    fsprings = restart_config['fsprings']

    irange = ((len(uout)-1)*outfreq, nstep-1) 

# random force/torque level
scalez = (kbt*2*dampc[0]/dt)**0.5
scalep = (kbt*2*dampc[2]/dt)**0.5

def init_poss(poss): # in m, m, rad
    points=origin_net_info['points'].copy()
    points[:, 0]=points[:,0]*0.01
    points[:, 1]=points[:,1]*0.01
    poss = jnp.hstack([points, rots.reshape([rots.shape[0],1])])
    return poss

ini_poss=init_poss(poss)

maglist, update_maglist=init_update_maglist()

Energy_mag=init_Energy_mag()
Energy_part=init_Energy_part()
Energy_boundary=init_Energy_boundary_cross()
Energy_bond=init_Energy_bond_cross()

Fbond = init_Force_bond_cross()
Fboundary = init_Force_boundary_cross()
Fmag = init_Force_mag()
Fmagrep = init_Force_mag_rep()

moveatom_y0 = np.zeros(len(poss))
moveatom_y0[moveatoms_yt] = (ini_poss.flatten())[moveatoms_yt*3+1]
moveatom_y0[moveatoms_yb] = (ini_poss.flatten())[moveatoms_yb*3+1]

moveatom_x0 = np.zeros(len(poss))
moveatom_x0[moveatoms_xr] = (ini_poss.flatten())[moveatoms_xr*3]
moveatom_x0[moveatoms_xl] = (ini_poss.flatten())[moveatoms_xl*3]
moveatoms_xy0 = np.vstack([moveatom_x0, moveatom_y0, np.zeros(len(poss))]).T
spring_xy0 = moveatoms_xy0 + spring_l0_off 

#-------------------------------VV steps-----------------------------
# define constant mats for LINCS to restrain bond lengths
Mmat = np.diag(np.array([mag_mass, mag_mass, mag_J]*len(poss))) # kg, kg, kg*m^2
dmat = lo.flatten().copy()*0.01 # in m
ini_poss_m = np.array(poss.copy()) # otherwise poss would be modified
ini_poss_m[:,0] *= 0.01
ini_poss_m[:,1] *= 0.01
ini_poss_m[:,2] = np.array(rots)

start = time.time()
for i in range(irange[0], irange[1]):
    current_start = time.time()
    newbs = update_boundary(i)

    # VV - step1
    Fran = np.random.normal(size=(npts,2))
    Fran = jnp.hstack([scalez*Fran, scalep*np.random.normal(size=(npts,1))])

    Fran = Fran.flatten()
    du0 = uold*jnp.array([100, 100, 1]*npts) # 1D array (ndofs,) [cm, cm, rad]*npts

    # update maglist to find force function
    maglist=update_maglist(maglist,du0)
    #print(f'update_maglist: {(time.time() - start):.3f} sec.')
    Energy_mag_func_fn=partial(Energy_mag,maglist=maglist)
    grad_func = jax.grad(Energy_mag_func_fn)
    Fint = grad_func(du0)
    #print(f'Energy_mag: {(time.time() - start):.3f} sec.')
    Energy_part_func_fn=partial(Energy_part, maglist=maglist)
    Fint_func = jax.grad(Energy_part_func_fn)
    Fints = Fint + Fint_func(du0)
    #print(f'Energy_part: {(time.time() - start):.3f} sec.')

    if vSMD:
        moveatom_y = np.zeros(len(poss))
        moveatom_y[moveatoms_yt] = (uold+ini_poss.flatten())[moveatoms_yt*3+1]
        moveatom_y[moveatoms_yb] = (uold+ini_poss.flatten())[moveatoms_yb*3+1]
        moveatom_x = np.zeros(len(poss))
        moveatom_x[moveatoms_xr] = (uold+ini_poss.flatten())[moveatoms_xr*3]
        moveatom_x[moveatoms_xl] = (uold+ini_poss.flatten())[moveatoms_xl*3]
        moveatoms_xy = np.vstack([moveatom_x, moveatom_y, np.zeros(len(poss))]).T
        Fspring = (np.abs(spring_v*dt*i + spring_xy0 - moveatoms_xy) - spring_l0)*spring_k
        Fspring = jnp.array(Fspring.flatten()) 
    else:
        Fspring = np.zeros(np.shape(Fext))

    Feold = Fran + Fext + Fspring + Fbond(du0) + Fboundary(du0, newbs) - Fints*jnp.array([100, 100, 1]*npts) # 1D array (ndofs,) [N, N, N*m]*npts
    #print(f'Fbond: {(time.time() - start):.3f} sec.')
    aold = (Feold - dampc*vold)/magpro # 1D array (ndofs,) [m/s2, m/s2, rad/s2]*npts
    unew = uold + vold*dt + 0.5*aold*dt**2 # 1D array (ndofs,) [m, m, rad]*npts

    # VV - step2
    du1 = unew*jnp.array([100, 100, 1]*npts) # 1D array (ndofs,) [cm, cm, rad]*npts
    #print(f'create unew and du1: {(time.time() - start):.3f} sec.')
    maglist=update_maglist(maglist,du1)
    #print(f'update_maglist new: {(time.time() - start):.3f} sec.')
    Energy_mag_func_fn=partial(Energy_mag,maglist=maglist)
    grad_func = jax.grad(Energy_mag_func_fn)
    Fint = grad_func(du1)
    #print(f'Energy_mag: {(time.time() - start):.3f} sec.')
    Energy_part_func_fn=partial(Energy_part,maglist=maglist)
    Fint_func = jax.grad(Energy_part_func_fn)
    Fints = Fint + Fint_func(du1)
    #print(f'Energy_part: {(time.time() - start):.3f} sec.')

    if vSMD:
        moveatom_y = np.zeros(len(poss))
        moveatom_y[moveatoms_yt] = (unew+ini_poss.flatten())[moveatoms_yt*3+1]
        moveatom_y[moveatoms_yb] = (unew+ini_poss.flatten())[moveatoms_yb*3+1]
        moveatom_x = np.zeros(len(poss))
        moveatom_x[moveatoms_xr] = (unew+ini_poss.flatten())[moveatoms_xr*3]
        moveatom_x[moveatoms_xl] = (unew+ini_poss.flatten())[moveatoms_xl*3]
        moveatoms_xy = np.vstack([moveatom_x, moveatom_y, np.zeros(len(poss))]).T
        Fspring = (np.abs(spring_v*dt*i + spring_xy0 - moveatoms_xy) - spring_l0)*spring_k

    fspring = Fspring.copy()
    Fspring = jnp.array(Fspring.flatten()) 
    fbond = Fbond(du1)
    fbc = Fboundary(du1, newbs)
    fint = - Fints*jnp.array([100, 100, 1]*npts)
    Fenew = Fran + Fext + fbond + fbc + fint + Fspring # 1D array (ndofs,) [N, N, N*m]*npts
    # 1D array (ndofs,) [m/s, m/s, rad/s]*npts
    vnew = (vold + 0.5*dt*aold + 0.5*dt*Fenew/magpro)/(1+0.5*dt*dampc/magpro) 
    anew = (Fenew - dampc*vnew)/magpro # 1D array (ndofs,) [m/s2, m/s2, rad/s2, m/s2, m/s2, rad/s2]

    ##------------------------------if rigid bar is used (remove if k is finite)--------------------------------[RIGID]
    u_curr = np.array(unew) + ini_poss_m.flatten() # current position
    Bmat = np.zeros([len(conn), 3*len(poss)])
    inn = 0
    for ij in conn:
        tempp = ((u_curr[ij[0]*3]-u_curr[ij[1]*3])**2 + (u_curr[ij[0]*3+1]-u_curr[ij[1]*3+1])**2)**0.5
        Bmat[inn][3*ij[0]] = (u_curr[ij[0]*3]-u_curr[ij[1]*3])/tempp
        Bmat[inn][3*ij[0]+1] = (u_curr[ij[0]*3+1]-u_curr[ij[1]*3+1])/tempp
        Bmat[inn][3*ij[1]] = -(u_curr[ij[0]*3]-u_curr[ij[1]*3])/tempp
        Bmat[inn][3*ij[1]+1] = -(u_curr[ij[0]*3+1]-u_curr[ij[1]*3+1])/tempp
        inn += 1
    # First projection.
    temp = (np.identity(len(poss)*3) - np.matmul(Bmat.T/2.0, Bmat))
    u_curr_restrained = np.matmul(temp, u_curr) + np.matmul(Bmat.T/2.0, dmat)
    # Second rotation.
    temp2 = u_curr_restrained.reshape([len(poss), 3])[:, :-1]
    pmat = (2*dmat**2-(np.linalg.norm(temp2[conn[:,0]] - temp2[conn[:,1]], axis=1))**2)**0.5
    u_curr_restrained_rot = np.matmul(temp, u_curr_restrained) + np.matmul(Bmat.T/2.0, pmat)
    # go back to jnp array
    unew = jnp.array(u_curr_restrained_rot - ini_poss_m.flatten())
    vnew = np.matmul(temp, vnew)
    anew = np.matmul(temp, anew)  
    ##-----------------------------------------------------------------------------------------------------------[RIGID]

    # check if particle has been pulled out of box, if Truem stop simulation
    if stop_in_process and np.max(np.abs(Fboundary(du1, newbs)))>checkbound:
        print("Atom left the box. Simulation stoped.")
        break
        
    # save data
    if (i+1)%outfreq == 0:
        uout.append(unew)
        vout.append(vnew)
        aout.append(anew)  
        fbonds.append(fbond)
        fbcs.append(fbc)
        fints.append(fint)  
        fsprings.append(fspring)
        
        emagthis = Energy_mag(du1, maglist=maglist)
        etotthis = emagthis + Energy_part(du1, maglist=maglist) + Energy_boundary(du1, newbs) + Energy_bond(du1)
        emagout.append(emagthis)
        etotout.append(etotthis)
    
        if savealongprocess and ((i+1)%(nstep//savenum) == 0):
            pkl.dump({"u":uout, "v":vout, "a":aout, "fbonds":fbonds, "fbcs":fbcs, "fints":fints, "fsprings":fsprings, "emag":emagout, "etot":etotout, "boundaries": newbs}, 
                open(f'dic_in_process_{operation}-{int(neq)}', 'wb'))

            if operation=='eq':
                fig = plt.figure(figsize=[9,4])
                ax = fig.add_subplot(111)
                espacing=1
                ax.plot(np.array(emagout)[::espacing], '-', label=f'emag')             
                plt.xlabel('frame no.')
                plt.ylabel('(J)')
                plt.xlim(0, ) # 8000 frames show 0.4s simulation
                plt.legend(loc='best')
                plt.title(f"Energy history for equilibration run")
                plt.tight_layout()
                plt.savefig(f'ehis.png', dpi=200)
                plt.close()

    # update old values
    uold = unew # (ndofs,)
    vold = vnew # (ndofs,)
    aold = anew # (ndofs,)   
    
    # print(f'loop {i}: {(time.time() - current_start):.3f} sec.')


pkl.dump({"u":uout, "v":vout, "a":aout, "fbonds":fbonds, "fbcs":fbcs, "fints":fints, "fsprings":fsprings, "emag":emagout, "etot":etotout, "boundaries": newbs}, 
    open(f'dic_in_process_{operation}-{int(neq)}', 'wb'))

##-------------------------PLOTTING------------------------------
if operation=='eq':
    fig = plt.figure(figsize=[9,4])
    ax = fig.add_subplot(111)
    espacing=1
    ax.plot(np.array(emagout)[::espacing], '-', label=f'emag')             
    plt.xlabel('frame no.')
    plt.ylabel('(J)')
    plt.xlim(0, ) # 8000 frames show 0.4s simulation
    plt.legend(loc='best')
    plt.title(f"Energy history for equilibration run")
    plt.tight_layout()
    plt.savefig(f'ehis.png', dpi=200)
    plt.close()

if operation=='biaxial':
    f_traj = np.array(fsprings)
    sumforce_his_xl = np.array([np.sum(f_traj[jjj][moveatoms_xl][:,0]) for jjj in range(len(f_traj))])
    sumforce_his_xr = np.array([np.sum(f_traj[jjj][moveatoms_xr][:,0]) for jjj in range(len(f_traj))])
    sumforce_his_yb = np.array([np.sum(f_traj[jjj][moveatoms_yb][:,1]) for jjj in range(len(f_traj))])
    sumforce_his_yt = np.array([np.sum(f_traj[jjj][moveatoms_yt][:,1]) for jjj in range(len(f_traj))])

    du_traj = np.array(pkl.load(open(f'dic_in_process_{operation}-{int(neq)}','rb'))['u'][1:]) # in m

    avedisp_his0_xl = np.sum(du_traj[:, moveatoms_xl*3], axis=1)/len(moveatoms_xl)
    avedisp_his_xl = avedisp_his0_xl-avedisp_his0_xl[0] # negative
    avedisp_his0_xr = np.sum(du_traj[:, moveatoms_xr*3], axis=1)/len(moveatoms_xr)
    avedisp_his_xr = avedisp_his0_xr-avedisp_his0_xr[0] # positive
    avedisp_his0_yt = np.sum(du_traj[:, moveatoms_yt*3+1], axis=1)/len(moveatoms_yt)
    avedisp_his_yt = avedisp_his0_yt-avedisp_his0_yt[0] # positive
    avedisp_his0_yb = np.sum(du_traj[:, moveatoms_yb*3+1], axis=1)/len(moveatoms_yb)
    avedisp_his_yb = avedisp_his0_yb-avedisp_his0_yb[0] # negative

    # plot x-dir and y-dir separately
    fig = plt.figure(figsize=[10,3.5])
    espacing = 1

    ax1 = fig.add_subplot(1,3,1)
    plt.plot(-avedisp_his_xl[::espacing]*100, -sumforce_his_xl[::espacing], label='xl')
    plt.plot(avedisp_his_xr[::espacing]*100, sumforce_his_xr[::espacing], label='xr')
    plt.plot(avedisp_his_yt[::espacing]*100, sumforce_his_yt[::espacing], label='yt')
    plt.plot(-avedisp_his_yb[::espacing]*100, -sumforce_his_yb[::espacing], label='yb')
    plt.legend(loc='best')
    plt.xlabel('Average displacement (cm)')
    plt.ylabel('Spring force (N)')
    plt.xlim(0, )

    ax2 = fig.add_subplot(1,3,2)
    ax2.plot(-avedisp_his_xl[::espacing]*100, label='xl')
    ax2.plot(avedisp_his_xr[::espacing]*100, label='xr')
    ax2.plot(avedisp_his_yt[::espacing]*100, label='yt')
    ax2.plot(-avedisp_his_yb[::espacing]*100, label='yb')
    plt.legend(loc='best')
    ax2.set_ylabel('Average displacement (cm)')
    ax2.set_xlabel('#frame')

    ax3 = fig.add_subplot(1,3,3)
    ax3.plot(-sumforce_his_xl[::espacing], label='xl')
    ax3.plot(sumforce_his_xr[::espacing], label='xr')
    ax3.plot(sumforce_his_yt[::espacing], label='yt')
    ax3.plot(-sumforce_his_yb[::espacing], label='yb')
    plt.legend(loc='best')
    ax3.set_ylabel('Spring force (N)')
    ax3.set_xlabel('#frame')

    plt.tight_layout()
    plt.savefig(f'F-d.png', dpi=200)
    plt.close()

    # plot x- and y-dir. average
    fig2 = plt.figure(figsize=[5, 3.5])
    ax = fig2.add_subplot(111)
    plt.plot((avedisp_his_xr-avedisp_his_xl)[::espacing]*100, 0.5*(sumforce_his_xr-sumforce_his_xl)[::espacing], label='xx')
    plt.plot((avedisp_his_yt-avedisp_his_yb)[::espacing]*100, 0.5*(sumforce_his_yt-sumforce_his_yb)[::espacing], label='yy')
    plt.legend(loc='best')
    plt.xlabel('Network edge length change (cm)')
    plt.ylabel('Average spring force on egdes (N)')
    plt.xlim(0, )
    plt.tight_layout()
    plt.savefig(f'F-d_aveXY.png', dpi=200)
    plt.close()