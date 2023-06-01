import sys
import os
import glob
import os.path as path
import time

# import statement for autograd wrapped numpy
import autograd.numpy as np   
# import statment for gradient calculator
from autograd.misc.flatten import flatten_func
from autograd import grad as compute_grad
# import numpy as np
import sympy as sym
from sympy import *
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

mu0 = 4*np.pi*1e-7    # permeability of vacuum (H/m), H = J/A2


def Energy_magnet_single(mag_m1, mag_m2, mag_phi1, mag_phi2, pos1, pos2):
    """Get magnetic energy from magnets orientation and distance. Follow point dipole approximation, length of
       cylindrical magnet much smaller than their distance.
       
    PARAMETERS
        mag_m1: magnitude of magnetic dipole, float (in A*m2).
        mag_m2: magnitude of magnetic dipole, float (in A*m2).
        mag_phi1: angle of m1 dipole moment direction vector from pos-x, float (in rad).
        mag_phi2: angle of m2 dipole moment direction vector from pos-x, float (in rad).
        pos1: m1 dipole moment position, 1D array [3,] (in cm).
        pos2: m2 dipole moment position, 1D array [3,] (in cm).

        r_vec12: distance vector of two magnets, from point dipole 1 to point dipole 2 (pt2_vec - pt1_vej), 
               1D array [3,] (in cm).
       
    RETURN
        E_from_vec: magnet interaction energy, float (in J).
    """
    mu0 = 4*np.pi*1e-7 
    r_vec12 = pos2 - pos1 #
    r_vec = r_vec12/100
    m1 = mag_m1*np.array([np.cos(mag_phi1), np.sin(mag_phi1), 0])
    m2 = mag_m2*np.array([np.cos(mag_phi2), np.sin(mag_phi2), 0])
    
    E_from_vec = mu0/(np.pi*4)*(np.inner(m1, m2)/np.linalg.norm(r_vec)**3 - 3*np.inner(m1, r_vec)*np.inner(m2, r_vec)/np.linalg.norm(r_vec)**5) # in J    
    return E_from_vec

def Energy_steric_repulsion(magr1, magr2, pos1, pos2, eps=3.0e-2):
    """Get steric repulsive energy due to the exclusive volume of magnets.
       
    PARAMETERS
        magr1: m1 radius, float (in cm).
        magr2: m2 radius, float (in cm).
        pos1: m1 dipole moment position, 1D array [3,] (in cm).
        pos2: m2 dipole moment position, 1D array [3,] (in cm).
       
    RETURN
        E_rep: teric repulsive energy, float (in J).
    """
    # critical c2c distance between two magnets
    cridis = magr1+magr2 # in cm
    # energy well depth, zero-energy position
    sig = cridis/2**(1/6) # in cm
    shiftedLJ = 4*eps*((sig/cridis)**12-(sig/cridis)**6)
    
    distemp = np.linalg.norm(pos1-pos2) # in cm
    if distemp < cridis: 
        E_rep = 4*eps*((sig/distemp)**12-(sig/distemp)**6) - shiftedLJ
    else:
        E_rep = 0

    return E_rep

def Energy_bond_single_dynamic(pos1, pos2, newpos1, newpos2, k=1.0, maxelong=0.3):
    """Get steric repulsive energy due to the exclusive volume of magnets.
       
    PARAMETERS
        l: deformed length, float (in cm).
        lo: original length, float (in cm).
        k: spring stiffness, float (in N/m)
       
    RETURN
        E_el: elastic bond energy, float (in J).
    """
    lo = np.linalg.norm(pos1-pos2)
    l = np.linalg.norm(newpos1-newpos2)
    E_el = 0.5*k*(l/100-lo/100)**2*((l/lo)<=(maxelong+1.0))

    return E_el

def Energy_bond_single_static(lo, newpos1, newpos2, k=1.0):
    """Get steric repulsive energy due to the exclusive volume of magnets.
       
    PARAMETERS
        l: deformed length, float (in cm).
        lo: original length, float (in cm).
        k: spring stiffness, float (in N/m)
       
    RETURN
        E_el: elastic bond energy, float (in J).
    """
    l = np.linalg.norm(newpos1-newpos2)
    E_el = 0.5*k*(l/100-lo/100)**2

    return E_el

def torque_calc(mag_m1, mag_m2, mag_phi1, mag_phi2, pos1, pos2):
    """Get magnet torque from magnets orientation and distance. Follow point dipole approximation, length of
       cylindrical magnet much smaller than their distance.
       
    PARAMETERS
        mag_m1: magnitude of magnetic dipole, float (in A*m2).
        mag_m2: magnitude of magnetic dipole, float (in A*m2).
        mag_phi1: angle of m1 dipole moment direction vector from pos-x, float (in rad).
        mag_phi2: angle of m2 dipole moment direction vector from pos-x, float (in rad).
        pos1: m1 dipole moment position, 1D array [3,] (in cm).
        pos2: m2 dipole moment position, 1D array [3,] (in cm).
    
    RETURN 
        T1, T2: torques felt by magnet 1 and 2, float (in N*m).
    """
    mu0 = 4*np.pi*1e-7 
    r_vec12 = pos2 - pos1 #
    r_vec = r_vec12/100
    dr = np.linalg.norm(r_vec)
    phi_align = np.arctan2(r_vec12[1], r_vec12[0])
    T1 = mu0*mag_m1*mag_m2/(np.pi*4*dr**3)*(np.sin(mag_phi1 - mag_phi2)+3*np.sin(phi_align-mag_phi1)*np.cos(phi_align-mag_phi2))
    T2 = mu0*mag_m1*mag_m2/(np.pi*4*dr**3)*(np.sin(mag_phi2 - mag_phi1)+3*np.cos(phi_align-mag_phi1)*np.sin(phi_align-mag_phi2))
    return T1, T2

def plot_discs(pos1, phi1, magr1, pos2, phi2, magr2, lo, bondbreak=True, maxelong=0.3):
    """Visualize the two-magnet system with draw-to-scale disc diameter, position and orientation.
       
    PARAMETERS
        pos1: m1 dipole moment position, 1D array [3,] (in cm).
        mag_phi1: angle of m1 dipole moment direction vector from pos-x, float (in rad).
        magr1: m1 radius, float (in cm).
        pos2: m2 dipole moment position, 1D array [3,] (in cm).
        mag_phi2: angle of m2 dipole moment direction vector from pos-x, float (in rad).
        magr2: m2 radius, float (in cm).
       
    RETURN NONE
    """

    fig = plt.figure(figsize=[4,4])
    ax = fig.add_subplot(111, aspect=1.0)

    plt.scatter(pos1[0], pos1[1], zorder=10, color='black')
    plt.scatter(pos2[0], pos2[1], zorder=11, color='black')

    circles = []
    # magnetic disc 1
    circles.append(plt.Circle(pos1[:-1], magr1, clip_on=False, color='orange'))
    ax.add_patch(circles[0])
    ax.text(pos1[0]+magr1, pos1[1]+magr1, '1', fontsize=14, color='b')
    ax.quiver(pos1[0], pos1[1], # <-- starting point of vector
              np.cos(phi1), np.sin(phi1), # <-- directions of vector
              color = 'k', alpha = 1.0, lw = 10, zorder=12, scale=3.5, headwidth=15, headlength=16)
    # magnetic disc 2
    circles.append(plt.Circle(pos2[:-1], magr2, clip_on=False, color='orange'))
    ax.add_patch(circles[1])
    ax.text(pos2[0]+magr2, pos2[1]+magr2, '2', fontsize=14, color='b')
    ax.quiver(pos2[0], pos2[1], # <-- starting point of vector
              np.cos(phi2), np.sin(phi2), # <-- directions of vector
              color = 'k', alpha = 1.0, lw = 10, zorder=12, scale=3.5, headwidth=15, headlength=16)

    l = np.linalg.norm(pos2 - pos1)
    if bondbreak and (l/lo)<=(1+maxelong):
        plt.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], '--', linewidth=1.0, c='gray')
    
    plt.xlim(ax.get_xlim()[0]-0.5, ax.get_xlim()[1]+0.5)
    plt.ylim(ax.get_ylim()[0]-0.5, ax.get_ylim()[1]+0.5)
    plt.xlabel('x (cm)')
    plt.ylabel('y (cm)')
    plt.tight_layout()
    plt.show()
    
    pass

def dist(pt1, pt2, pt3): # x3,y3 is the point
    
    x1, y1 = pt1[:-1]
    x2, y2 = pt2[:-1]
    x3, y3 = pt3[:-1]
    
    px = x2-x1
    py = y2-y1

    norm = px*px + py*py

    u =  ((x3 - x1) * px + (y3 - y1) * py) / float(norm)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0
        
    x = x1 + u * px
    y = y1 + u * py

    dx = x - x3
    dy = y - y3
    
    foot = np.array([x, y, 0])
    dist = (dx*dx + dy*dy)**.5

    return u, dist, foot

class multi_disc_system:
    
    mu0 = 4*np.pi*1e-7    # permeability of vacuum (H/m), H = J/A2
    
    def __init__(self, magrs, magts, Brs, mag_phis, poss, connec, lo, eps=3.0e-2, k=1.0, krep=5000, dc=0, kang=1.0):
        
        self.ptnum = len(magrs)
        self.edgenum = len(connec)
        self.magrs = magrs # magnet radius and thickness are in cm, 1D array [ptnum,]
        self.mag_ms = Brs/mu0 * (magts/100)*np.pi*(magrs/100)**2 # in A*m2
        self.mag_phis = mag_phis # in rad
        self.connec = connec # connectivity matrix, 2D array, [edgenum, 2]
        coss = np.cos(mag_phis) # 1D array [ptnum,]
        sins = np.sin(mag_phis) # 1D array [ptnum,]
        self.mag_ns = np.vstack([coss, sins, np.zeros(self.ptnum)]).T # stacked unit vector, 2D array [ptnum, 3]
        self.poss = poss # shape 2D array [ptnum, 3], in cm
        
        self.lo = lo # in cm, 1D array, [edgenum,]
        self.eps = eps # in J (for repulsion calc)
        self.k = k # in N/m (elastic bond stiffness)
        self.krep = krep # in N/m (elastic repulsive bond stiffness)
        self.dc = dc # in cm (bond critical distance = repulsive radius for bond + magnet radius)
        self.kang= kang # in J, anglr stiffness (fixed-end) 

        self.conn_dic = {}
        for jj in range(self.edgenum):
            self.conn_dic[self.connec[jj][0]] = jj
            self.conn_dic[self.connec[jj][1]] = jj

        self.rot_re = np.zeros(np.shape(self.mag_phis))
        for thisconn in self.connec:
            temp = self.poss[thisconn[1]] - self.poss[thisconn[0]]
            temp = temp[:2]/np.linalg.norm(temp[:2])
            thisrot = np.arctan2(temp[1], temp[0])
            self.rot_re[thisconn[1]] = self.mag_phis[thisconn[1]] - thisrot
            self.rot_re[thisconn[0]] = self.mag_phis[thisconn[0]] - thisrot
        # convert to [-pi, pi] 
        self.rot_re = np.arctan2(np.sin(self.rot_re), np.cos(self.rot_re))
        
    def force_bond_cross(self, udofs):
        """ A method to calculate the bond repulsive force of the magneto-elastic network after deformations, udofs.

            PARAMETERS 
                udofs: displacement of free DOFs, 1D array [#freedofs,] (x1, y1, theta1, x2, y2, theta2, ...), 
                       (in cm, cm, rad, cm, cm, rad, ...).

            RETURN 
                Fbond: total energy, 1D array [#freedofs,] (in N, N, null, N, N, null, ...).

        """
        # update normal and position
        udofs_arr = udofs.reshape([self.ptnum, 3])
        udofs_xyz = udofs_arr[:,:-1]
        udofs_xyz = np.hstack([udofs_xyz, np.zeros([self.ptnum,1])])
        new_poss = self.poss + udofs_xyz

        Fbond = np.zeros(len(udofs)).reshape([int(len(udofs)//3),3])

        for pp in range(len(self.connec)):
            ptno = np.arange(self.ptnum)
            for pt in ptno[np.where((ptno!=self.connec[pp][0]) & (ptno!=self.connec[pp][1]))]:
                # shortest distance between pt and bond (line segment) pp
                u, ds, foot = dist(new_poss[self.connec[pp][0]], new_poss[self.connec[pp][1]], new_poss[pt]) 
                thisf = self.krep*(self.dc*0.01-ds*0.01)*(u>0)*(u<1)
                if thisf > 0:
                    unit_vec = new_poss[pt] - foot # pointing from foot to new_poss[pt], 1D array, [3,]
                    unit_vec = unit_vec/np.linalg.norm(unit_vec)
                    thisf = thisf*unit_vec
                    Fbond[pt] += thisf
                    L = np.linalg.norm(new_poss[self.connec[pp][0]]-foot)/np.linalg.norm(new_poss[self.connec[pp][0]]-new_poss[self.connec[pp][1]])
                    Fbond[self.connec[pp][1]] += -thisf*L
                    Fbond[self.connec[pp][0]] += -thisf*(1-L)
                else:
                    continue

        return Fbond.flatten()   

    def Energy_bond_cross(self, udofs):
        """ A method to calculate the bond repulsive force of the magneto-elastic network after deformations, udofs.

            PARAMETERS 
                udofs: displacement of free DOFs, 1D array [#freedofs,] (x1, y1, theta1, x2, y2, theta2, ...), 
                       (in cm, cm, rad, cm, cm, rad, ...).

            RETURN 
                Fbond: total energy, 1D array [#freedofs,] (in N, N, null, N, N, null, ...).

        """
        # update normal and position
        udofs_arr = udofs.reshape([self.ptnum, 3])
        udofs_xyz = udofs_arr[:,:-1]
        udofs_xyz = np.hstack([udofs_xyz, np.zeros([self.ptnum,1])])
        new_poss = self.poss + udofs_xyz

        ebondcross = 0
        for pp in range(len(self.connec)):
            ptno = np.arange(self.ptnum)
            for pt in ptno[np.where((ptno!=self.connec[pp][0]) & (ptno!=self.connec[pp][1]))]:
                # shortest distance between pt and bond (line segment) pp
                u, ds, foot = dist(new_poss[self.connec[pp][0]], new_poss[self.connec[pp][1]], new_poss[pt]) 
                ebondcross += 0.5*self.krep*(self.dc*0.01-ds*0.01)**2*(u>0)*(u<1)*(ds<self.dc)

        return ebondcross  

    def Energy_angle(self, udofs):
        
        udofs_arr = udofs.reshape([self.ptnum, 3])
        udofs_rot = udofs_arr[:, -1]
        udofs_xyz = udofs_arr[:,:-1]
        udofs_xyz = np.hstack([udofs_xyz, np.zeros([self.ptnum,1])])
        new_poss = self.poss + udofs_xyz
        new_rots = self.mag_phis + udofs_rot
        
        # calculate new rotation of springs 
        newsrot_spring = []
        for thisconn in self.connec:
            temp = new_poss[thisconn[1]] - new_poss[thisconn[0]]
            temp = temp/np.linalg.norm(temp)
            newsrot_spring.append(np.arctan2(temp[1], temp[0]))
        newsrot_spring = np.array(newsrot_spring)

        # relative angle between the spring and the magnetic dipole
        newrot_re = np.array([new_rots[nn] - newsrot_spring[self.conn_dic[nn]] for nn in range(self.ptnum)])
        # convert to [-pi, pi] 
        newrot_re = np.arctan2(np.sin(newrot_re), np.cos(newrot_re))
        temp2 = newrot_re - self.rot_re
        temp2 = np.arctan2(np.sin(temp2), np.cos(temp2))
        energy_ang = self.kang*temp2**2 # temp2 is now in [-pi, pi]
        
        return np.sum(energy_ang)

    def relative_angle(self, udofs):
        
        udofs_arr = udofs.reshape([self.ptnum, 3])
        udofs_rot = udofs_arr[:, -1]
        udofs_xyz = udofs_arr[:,:-1]
        udofs_xyz = np.hstack([udofs_xyz, np.zeros([self.ptnum,1])])
        new_poss = self.poss + udofs_xyz
        new_rots = self.mag_phis + udofs_rot
        
        # calculate new rotation of springs 
        newsrot_spring = []
        for thisconn in self.connec:
            temp = new_poss[thisconn[1]] - new_poss[thisconn[0]]
            temp = temp/np.linalg.norm(temp)
            newsrot_spring.append(np.arctan2(temp[1], temp[0]))
        newsrot_spring = np.array(newsrot_spring)

        # relative angle between the spring and the magnetic dipole
        newrot_re = np.array([new_rots[nn] - newsrot_spring[self.conn_dic[nn]] for nn in range(self.ptnum)])
        # convert to [-pi, pi] 
        newrot_re = np.arctan2(np.sin(newrot_re), np.cos(newrot_re))
        temp2 = newrot_re - self.rot_re
        temp2 = np.arctan2(np.sin(temp2), np.cos(temp2))
        
        return temp2 

    def Energy_decomp(self, udofs):
        """ A method to calculate the total energy and the components of the magneto-elastic network after 
            deformations, udofs.

            PARAMETERS 
                udofs: displacement of free DOFs, 1D array [#freedofs,] (x1, y1, theta1, x2, y2, theta2, ...), 
                       1D array [6,] (in cm).
            
            RETURN 
                (E_tot, E_mag_sys, E_rep_sys): total, magnetic, and repulsive energy, tuple [3,] (in J).

        """
        # update normal and position
        udofs_arr = udofs.reshape([self.ptnum, 3])
        udofs_rot = udofs_arr[:, -1]
        udofs_xyz = udofs_arr[:,:-1]
        udofs_xyz = np.hstack([udofs_xyz, np.zeros([self.ptnum,1])])
        
        new_poss = self.poss + udofs_xyz
        new_rots = self.mag_phis + udofs_rot
        
        # Magnetic energy
        E_mag_sys = 0
        for ii in range(self.ptnum):
            for jj in range(ii+1, self.ptnum):
                E_mag_sys += Energy_magnet_single(self.mag_ms[ii], self.mag_ms[jj], new_rots[ii], new_rots[jj], new_poss[ii], new_poss[jj])
        # Steric repulsion
        E_rep_sys = 0
        for ii in range(self.ptnum):
            for jj in range(ii+1, self.ptnum):
                E_rep_sys += Energy_steric_repulsion(self.magrs[ii], self.magrs[jj], new_poss[ii], new_poss[jj])
                        
        # Elastic bond  
        E_el_sys = np.sum([Energy_bond_single_static(self.lo[pp], new_poss[self.connec[pp][0]], new_poss[self.connec[pp][1]], self.k) for pp in range(len(self.connec))])
        
        # exclusive bond repulsion
        ebondcross = 0
        for pp in range(len(self.connec)):
            ptno = np.arange(self.ptnum)
            for pt in ptno[np.where((ptno!=self.connec[pp][0]) & (ptno!=self.connec[pp][1]))]:
                # shortest distance between pt and bond (line segment) pp
                u, ds, foot = dist(new_poss[self.connec[pp][0]], new_poss[self.connec[pp][1]], new_poss[pt]) 
                ebondcross += 0.5*self.krep*(self.dc*0.01-ds*0.01)**2*(u>0)*(u<1)*(ds<self.dc)
        
        eangle = self.Energy_angle(udofs)

        E_tot = E_mag_sys + E_rep_sys + E_el_sys + ebondcross + eangle
        print(f'Total: {E_tot:.3e}, magnetic: {E_mag_sys:.3e}, elastic:{E_el_sys:.3e}, magnet-magnet LJ repulsion: {E_rep_sys:.3e},  bond-magnet el repulsion: {ebondcross:.3e}, angle energy: {eangle:.3e}')
        
        return (E_tot, E_mag_sys, E_el_sys, E_rep_sys, ebondcross, eangle)

    def Energy_tot(self, udofs):
        """ A method to calculate the total energy of the magneto-elastic network after deformations, udofs.
            bond-bond repulsion is not included. (cause that force is defined externally instead of from the 
            derivative of total energy)

            PARAMETERS 
                udofs: displacement of free DOFs, 1D array [#freedofs,] (x1, y1, theta1, x2, y2, theta2, ...), 
                       1D array [6,] (in cm).
            
            RETURN 
                E_tot: total energy, float (in J).

        """
        # update normal and position
        udofs_arr = udofs.reshape([self.ptnum, 3])
        udofs_rot = udofs_arr[:, -1]
        udofs_xyz = udofs_arr[:,:-1]
        udofs_xyz = np.hstack([udofs_xyz, np.zeros([self.ptnum,1])])
        
        new_poss = self.poss + udofs_xyz
        new_rots = self.mag_phis + udofs_rot
        
        # Magnetic energy
        E_mag_sys = 0
        for ii in range(self.ptnum):
            for jj in range(ii+1, self.ptnum):
                E_mag_sys += Energy_magnet_single(self.mag_ms[ii], self.mag_ms[jj], new_rots[ii], new_rots[jj], new_poss[ii], new_poss[jj])
        # Steric repulsion
        E_rep_sys = 0
        for ii in range(self.ptnum):
            for jj in range(ii+1, self.ptnum):
                E_rep_sys += Energy_steric_repulsion(self.magrs[ii], self.magrs[jj], new_poss[ii], new_poss[jj])
                        
        # Elastic bond  
        E_el_sys = np.sum([Energy_bond_single_static(self.lo[pp], new_poss[self.connec[pp][0]], new_poss[self.connec[pp][1]], self.k) for pp in range(len(self.connec))])
        
        eangle = self.Energy_angle(udofs)

        E_tot = E_mag_sys + E_rep_sys + E_el_sys + eangle
        
        return E_tot
    
    def Energy_rep(self, udofs):
        """ A method to calculate the total energy of the magneto-elastic network after deformations, udofs.

            PARAMETERS 
                udofs: displacement of free DOFs, 1D array [#freedofs,] (x1, y1, theta1, x2, y2, theta2, ...), 
                       1D array [6,] (in cm).
            
            RETURN 
                E_tot: total energy, float (in J).

        """
        # update normal and position
        udofs_arr = udofs.reshape([self.ptnum, 3])
        udofs_xyz = udofs_arr[:,:-1]
        udofs_xyz = np.hstack([udofs_xyz, np.zeros([self.ptnum,1])])
        
        new_poss = self.poss + udofs_xyz
        
         # Steric repulsion
        E_rep_sys = 0
        for ii in range(self.ptnum):
            for jj in range(ii+1, self.ptnum):
                E_rep_sys += Energy_steric_repulsion(self.magrs[ii], self.magrs[jj], new_poss[ii], new_poss[jj])
                        
        return E_rep_sys 

    def Energy_el(self, udofs):
        """ A method to calculate the total energy of the magneto-elastic network after deformations, udofs.

            PARAMETERS 
                udofs: displacement of free DOFs, 1D array [#freedofs,] (x1, y1, theta1, x2, y2, theta2, ...), 
                       1D array [6,] (in cm).
            
            RETURN 
                E_tot: total energy, float (in J).

        """
        # update normal and position
        udofs_arr = udofs.reshape([self.ptnum, 3])
        udofs_xyz = udofs_arr[:,:-1]
        udofs_xyz = np.hstack([udofs_xyz, np.zeros([self.ptnum,1])])
        new_poss = self.poss + udofs_xyz
        
         # Elastic bond  
        E_el_sys = np.sum([Energy_bond_single_static(self.lo[pp], new_poss[self.connec[pp][0]], new_poss[self.connec[pp][1]], self.k) for pp in range(len(self.connec))])

        
        return E_el_sys
    
    def Energy_mag_only(self, udofs):
        """ A method to calculate the magnetic energy of the magneto-elastic network after deformations, udofs.

            PARAMETERS 
                udofs: displacement of free DOFs, 1D array [#freedofs,] (x1, y1, theta1, x2, y2, theta2), 
                       1D array [6,] (in cm).
            
            RETURN 
                E_mag_sys: magnetic energy, float (in J).

        """
        # update normal and position
        udofs_arr = udofs.reshape([self.ptnum, 3])
        udofs_rot = udofs_arr[:, -1]
        udofs_xyz = udofs_arr[:,:-1]
        udofs_xyz = np.hstack([udofs_xyz, np.zeros([self.ptnum,1])])
        
        new_poss = self.poss + udofs_xyz
        new_rots = self.mag_phis + udofs_rot
        
        # Magnetic energy
        E_mag_sys = 0
        for ii in range(self.ptnum):
            for jj in range(ii+1, self.ptnum):
                E_mag_sys += Energy_magnet_single(self.mag_ms[ii], self.mag_ms[jj], new_rots[ii], new_rots[jj], new_poss[ii], new_poss[jj])
        
        return E_mag_sys
    
    def gradEnergy(self):
        udofs = np.zeros(self.ptnum*3)
        g_flat, unflatten, uu = flatten_func(self.Energy_tot, udofs)
        grad_mag = compute_grad(g_flat) 
        return grad_mag
    
    def gradEnergy_mag_only(self):
        udofs = np.zeros(self.ptnum*3)
        g_flat, unflatten, uu = flatten_func(self.Energy_mag_only, udofs)
        grad_mag = compute_grad(g_flat) 
        return grad_mag
    
    def gradEnergy_el_only(self):
        udofs = np.zeros(self.ptnum*3)
        g_flat, unflatten, uu = flatten_func(self.Energy_el, udofs)
        grad_mag = compute_grad(g_flat) 
        return grad_mag
    
    def gradEnergy_rep_only(self):
        udofs = np.zeros(self.ptnum*3)
        g_flat, unflatten, uu = flatten_func(self.Energy_rep, udofs)
        grad_mag = compute_grad(g_flat) 
        return grad_mag

    def gradEnergy_angle_only(self):
        udofs = np.zeros(self.ptnum*3)
        g_flat, unflatten, uu = flatten_func(self.Energy_angle, udofs)
        grad_mag = compute_grad(g_flat) 
        return grad_mag

def plot_multi_discs(poss, phis, magrs, connec, circlescale=1.0, shiftscale=1, wbold=14, size=[4,4],
    arrowlw=10, arrowscale=7.5, arrowheadwidth=6, arrowheadlength=6, arrowalpha=1.0, arrowc='k', shownodenum=True, nodesize=5,
    nodefontsize=12, save=False, filename=None, startfrom1=True, show=False, color_by_deform=False, color_by_rot=False, dup=None):
    """Visualize the two-magnet system with draw-to-scale disc diameter, position and orientation.
       
    PARAMETERS
        pos1: m1 dipole moment position, 1D array [3,] (in cm).
        mag_phi1: angle of m1 dipole moment direction vector from pos-x, float (in rad).
        magr1: m1 radius, float (in cm).
        pos2: m2 dipole moment position, 1D array [3,] (in cm).
        mag_phi2: angle of m2 dipole moment direction vector from pos-x, float (in rad).
        magr2: m2 radius, float (in cm).
        dup: a frame of output displacements and rotation from simulation, [# of atoms, 3], (in m, m, rad).
    RETURN NONE
    """

    fig = plt.figure(figsize=size)
    ax = fig.add_subplot(111, aspect=1.0)

    plt.scatter(poss[:, 0], poss[:, 1], zorder=10, color='black',s=nodesize)

    circles = []
    # magnetic discs
    if color_by_deform:
        deltdist = np.linalg.norm(dup[:,:2], axis=1)
        deltrot = dup[:,2]
        cax = plt.scatter(poss[:,0],poss[:,1], s=10*circlescale, alpha=1, clip_on=False, c=deltdist, cmap='copper')
        cbar=fig.colorbar(cax, label='Displacement [cm]', orientation='horizontal', location='top', 
            shrink=0.75, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=13) 

    if color_by_rot:
        deltdist = np.linalg.norm(dup[:,:2], axis=1)
        deltrot = dup[:,2]
        cax = plt.scatter(poss[:,0],poss[:,1], s=10*circlescale, alpha=1, clip_on=False, c=deltrot, cmap='coolwarm',
            vmin=-np.max(np.abs(deltrot)), vmax=np.max(np.abs(deltrot)))
        cbar=fig.colorbar(cax, label='Rotation [rad]', orientation='horizontal', location='top', 
            shrink=0.75, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=13) 

    for i in range(len(phis)):
            # circles.append(plt.Circle(poss[i][:-1], magrs[i]*circlescale, clip_on=False, 
            #     color=deltdist[i], cmap='copper'))
        if (not color_by_deform) and (not color_by_rot):
            circles.append(plt.Circle(poss[i][:-1], magrs[i]*circlescale, clip_on=False, color='orange'))
            ax.add_patch(circles[i])
        if shownodenum:
            if startfrom1:
                ax.text(poss[i][0]+magrs[i]*shiftscale, poss[i][1]+magrs[i]*shiftscale, f'{i+1}', fontsize=nodefontsize, color='b')
            else:
                ax.text(poss[i][0]+magrs[i]*shiftscale, poss[i][1]+magrs[i]*shiftscale, f'{i}', fontsize=nodefontsize, color='b')
        ax.quiver(poss[i][0], poss[i][1], # <-- starting point of vector
                  np.cos(phis[i]), np.sin(phis[i]), # <-- directions of vector
                  color = arrowc, alpha = arrowalpha, lw = arrowlw, zorder=12, scale=arrowscale, headwidth=arrowheadwidth, headlength=arrowheadlength)

    # springs
    for pair in connec:
        ni = pair[0]
        nj = pair[1]
        plt.plot([poss[ni][0], poss[nj][0]], [poss[ni][1], poss[nj][1]], '--', linewidth=1.0, c='gray')
        plt.plot([poss[ni][0], poss[nj][0]], [poss[ni][1], poss[nj][1]], '-', linewidth=wbold, c='turquoise', alpha=0.5, zorder=0)
    
    plt.xlim(ax.get_xlim()[0]-0.5, ax.get_xlim()[1]+0.5)
    plt.ylim(ax.get_ylim()[0]-0.5, ax.get_ylim()[1]+0.5)
    plt.xlabel('$x$ [cm]', fontsize=17)
    plt.ylabel('$y$ [cm]', fontsize=17)
    ax.tick_params(axis='both', which='both', labelsize=16)
    plt.tight_layout()
    if save:
        plt.savefig(filename, dpi=200)
    if show:
        plt.show()
    
    pass


def animate_traj(test_sys, u_traj, save=False, name=None, xbound=(-2,2), ybound=(-2, 2), sizex=7, sizey=6.5,
                 arrowlw=10, arrowscale=20, arrowheadwidth=1., arrowheadlength=1., arrowalpha=0.5,
                 wbold=14, dpi=150, interval=10, showtime=False, tx=5, ty=5, nodesize=3,
                 duration=9.0, tstart=0, timersize=12, writer=plt.rcParams['animation.writer']): 
    
    poss = test_sys.poss
    magrs = test_sys.magrs
    rots = test_sys.mag_phis
    connec = test_sys.connec
    
    fig = plt.figure(figsize=[sizex, sizey])
    fig.set_dpi(dpi)
    # fig.set_size_inches(sizex, sizey)

    ax = plt.axes(xlim=xbound, ylim=ybound, aspect=1.0)
    patches = []
    patches2 = []
    qvs = []

    for ii in range(test_sys.ptnum):
        patches.append(plt.Circle(poss[ii][:-1], magrs[ii], clip_on=True, color='orange'))
        patches2.append(plt.Circle(poss[ii][:-1], nodesize, clip_on=True, color='k', zorder=500))
        qvs.append(ax.quiver(poss[ii][0], poss[ii][1], # <-- starting point of vector
                np.cos(rots[ii]), np.sin(rots[ii]), # <-- directions of vector
                color = 'k', alpha=arrowalpha, lw = arrowlw, zorder=12, scale=arrowscale, 
                                headwidth=arrowheadwidth, headlength=arrowheadlength))


    bonds = []
    boldbonds = []
    for pp in connec:
            boldbonds.append(plt.plot([poss[pp[0]][0], poss[pp[1]][0]], [poss[pp[0]][1], poss[pp[1]][1]], '-', linewidth=wbold, c='turquoise', alpha=0.5, zorder=0))
            bonds.append(plt.plot([poss[pp[0]][0], poss[pp[1]][0]], [poss[pp[0]][1], poss[pp[1]][1]], '--', linewidth=1.0, color='gray'))

    def init():
        for ii in range(test_sys.ptnum):
            patches[ii].center = (poss[ii][0], poss[ii][1])
            patches2[ii].center = (poss[ii][0], poss[ii][1])
            ax.add_patch(patches[ii])
            ax.add_patch(patches2[ii])
        return patches, patches2, qvs, bonds, boldbonds,

    def animate(i):
        for jj in range(test_sys.ptnum):
            x, y, rot = (u_traj[i][3*jj]*100, u_traj[i][3*jj+1]*100, u_traj[i][3*jj+2])
            # update magnet position
            patches[jj].center = (x,y)
            patches2[jj].center = (x,y)
            # update magnet orientation
            qvs[jj].set_offsets([x, y])
            qvs[jj].set_UVC(np.cos(rot), np.sin(rot))
            if showtime:
                timer = tstart + i/len(u_traj)*duration
                # ax.text(tx,ty,f'{(i/len(u_traj)*duration):.2f} s')
                ax.text(tx,ty,f'{timer:.1f} s', fontsize=timersize, color='blue', 
                    bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.1'))

        # update dynamic bond
        ind = 0
        for pp in connec:
            bonds[ind][0].set_xdata([u_traj[i][3*pp[0]]*100, u_traj[i][3*pp[1]]*100])
            bonds[ind][0].set_ydata([u_traj[i][3*pp[0]+1]*100, u_traj[i][3*pp[1]+1]*100])
            boldbonds[ind][0].set_xdata([u_traj[i][3*pp[0]]*100, u_traj[i][3*pp[1]]*100])
            boldbonds[ind][0].set_ydata([u_traj[i][3*pp[0]+1]*100, u_traj[i][3*pp[1]+1]*100])

            ind += 1

        return patches, patches2, qvs, bonds, boldbonds,

    anim = animation.FuncAnimation(fig, animate, 
                                   init_func=init, 
                                   frames=len(u_traj)-1, 
                                   interval=interval,
                                   blit=True,
                                   repeat=False)

    ax.set_xlim(xbound)
    ax.set_ylim(ybound)
    # plt.title('Trajectory (2D)')
    plt.xlabel('$x$ [cm]', fontsize=17)
    plt.ylabel('$y$ [cm]', fontsize=17)
    ax.tick_params(axis='both', which='both', labelsize=16)
    plt.show()

    if save:
        anim.save(name, fps=30, dpi=dpi, writer=writer)
        
    pass

    # Saved movie and Jupyter notebook visualization may be different,
    # since several frames may be lost due to input animation parameters