#!/usr/bin/python
# Copyright (c) 2014, Floris van Breugel
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Author: Floris van Breugel
#
from precomputed_buchner71 import receptor_dirs, triangles, hex_faces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from optparse import OptionParser
import os
import pickle
import time
import scipy.spatial
import copy

#######################################################################
# Helper Functions
#######################################################################
def get_filenames_from_directory(path, match='.png'):    
    cmd = 'ls ' + path
    ls = os.popen(cmd).read()
    all_filelist = ls.split('\n')
    try:
        all_filelist.remove('')
    except:
        pass
        
    filelist = []
    for i, filename in enumerate(all_filelist):
        if match in filename:
            filelist.append(os.path.join(path, filename))
    return filelist

def get_simple_intrinsic_camera_matrix(focal_length, sensor_width, sensor_height):
    M = np.zeros([3,3])
    M = np.zeros([3,3])
    M[0,0] = focal_length
    M[1,1] = focal_length
    M[0,2] = sensor_width/2.
    M[1,2] = sensor_height/2.
    M[2,2] = 1
    return M
    
def load_intrinsic_camera_matrix_from_csv_file(filename):
    f = open(filename)
    line = f.readlines()[0].rstrip(',')
    vals = line.split(',')
    mat = [float(val) for val in vals]        
    mat = np.array(mat).reshape(3,3)
    return mat

#######################################################################
# Equirectangular Math
#######################################################################
def get_equirectangular_hexagon(n, hex_face):
    new_face = []
    for v in hex_face:
        a = np.arctan(v[0]/v[1])
        
        if n > 698:
            if v[2] > 0:
                if a>1.4:
                    a -= np.pi
            if v[2] <= 0:
                if a> 1.26:
                    a -= np.pi
            a += np.pi/2.
            new_face.append([a,v[2]*np.pi/2.])
        
        if n <= 698:
            if v[2] > 0:
                if a<-1.4:
                    a += np.pi
            if v[2] <= 0:
                if a< -1.26:
                    a += np.pi
            a -= np.pi/2.
            new_face.append([a,v[2]*np.pi/2.])
    return np.array(new_face)

#######################################################################
# Ommatidia Calculations
#######################################################################

def get_lat_long_of_ommatidial_center(om, ommatidia_directions=None):
    if ommatidia_directions is None:
        ommatidia_directions = get_ommatidia_directions()
    pt = ommatidia_directions[om]
    lat = np.arccos(pt[2])-np.pi/2.
    lon = -1*np.arctan2(pt[1], pt[0])
    return [lat, lon]
def get_all_ommatidia_as_lat_long(ommatidia_directions=None):
    if ommatidia_directions is None:
        ommatidia_directions = get_ommatidia_directions()
    pts = []
    for pt in ommatidia_directions:
        lat = np.arccos(pt[2])-np.pi/2.
        lon = -1*np.arctan2(pt[1], pt[0])
        pts.append([lat, lon])
    return pts

def molleweide_newton_raphson(lat, eps=0.01):
    theta = lat
    if np.abs(theta) > 1.56:
        return theta
    err = 100
    while err > eps:
        theta_new = theta - (2*theta+np.sin(2*theta)-np.pi*np.sin(lat))/(2+2*np.cos(2*theta))
        err = np.abs(theta_new - theta)
        theta = theta_new
    return theta        
        
def get_xy_projection_of_ommatidial_center(om, ommatidia_directions=None, projection='equirectangular'):
    if ommatidia_directions is None:
        ommatidia_directions = get_ommatidia_directions()
    center = ommatidia_directions[om]
    
    if projection == 'equirectangular':
        a = np.arctan(center[0]/center[1])
            
        if om > 698:
            if center[2] > 0:
                if a>1.4:
                    a -= np.pi
            if center[2] <= 0:
                if a> 1.26:
                    a -= np.pi
            a += np.pi/2.
        
        if om <= 698:
            if center[2] > 0:
                if a<-1.4:
                    a += np.pi
            if center[2] <= 0:
                if a< -1.26:
                    a += np.pi
            a -= np.pi/2.
            
        center_xy = np.array([a, center[2]*np.pi/2.])
        return center_xy
    
    if projection == 'molleweide':
        if om <= 698:
            central_meridian = -np.pi/2.
        else:
            central_meridian = np.pi/2.
        lat, lon = get_lat_long_of_ommatidial_center(om, ommatidia_directions)
        theta = molleweide_newton_raphson(lat)
        x = 2*np.sqrt(2)/np.pi * (lon - central_meridian)*np.cos(theta)
        y = np.sqrt(2)*np.sin(theta)
        if om <= 698:
            x += -np.pi/2.
        else:
            x += np.pi/2.
        center_xy = np.array([x, y])
        return center_xy
        
    if projection == 'latlong':
        lat, lon = get_lat_long_of_ommatidial_center(om, ommatidia_directions)
        center_xy = np.array([lon, lat])
        return center_xy

def plot_eye_map(ommatidia_directions=None, projection='molleweide'):
    if ommatidia_directions is None:
        ommatidia_directions = get_ommatidia_directions()
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_axes([0,0,1,1])
    fig.set_frameon(False)
    
    ax.set_axis_bgcolor('white')
    ax.set_xlim(-np.pi,np.pi)
    ax.set_ylim(-np.pi/2.,np.pi/2.)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    
    triangles = get_delaunay_triangles()
    
    for om in range(ommatidia_directions.shape[0]):
        print om
        center = get_xy_projection_of_ommatidial_center(om, ommatidia_directions, projection)
        ax.text(center[0], center[1], str(om), horizontalalignment='center', verticalalignment='center', fontsize=3, color='black', zorder=1000)
        
        neighbors = get_delaunay_neighbors(om, triangles)
        for neighbor in neighbors:
            neighbor_center = get_xy_projection_of_ommatidial_center(neighbor, ommatidia_directions, projection)
            ax.plot([center[0], neighbor_center[0]], [center[1], neighbor_center[1]], 'red', linewidth=0.5) 
        
    fig.savefig('eye_map.pdf', format='pdf')


def get_angle_between_ommatidia(om1, om2, ommatidia_directions=None):
    if ommatidia_directions is None:
        ommatidia_directions = get_ommatidia_directions()

    # check if they are on the same eye
    n_ommatidia = len(ommatidia_directions)
    side1 = np.sign(om1 - int(n_ommatidia/2.))
    side2 = np.sign(om1 - int(n_ommatidia/2.))
    
    if side1 != side2:
        return np.pi
    
    else:
        dir1 = ommatidia_directions[om1]
        dir2 = ommatidia_directions[om2]
        s = np.sum(dir1*dir2)
        if np.abs(s) <= 1:
            angle = np.arccos(s)
        else:
            angle = 0
        return angle
        
def ommatidia_on_same_eye(om1, om2, ommatidia_directions=None):
    if ommatidia_directions is None:
        ommatidia_directions = get_ommatidia_directions()
    if np.sign(om1-698.5) == np.sign(om2-698.5):
        return True
    else:
        return False

## calculate connectivity of ommatidia using delaunay triangles

def get_delaunay_triangles(ommatidia_directions=None):
    if ommatidia_directions is None:
        ommatidia_directions = get_ommatidia_directions()
    centers = []
    for om in range(1398):
        center = get_xy_projection_of_ommatidial_center(om, ommatidia_directions, projection='molleweide')
        centers.append(center)
    triangles = scipy.spatial.Delaunay(centers)
    return triangles

def get_delaunay_neighbors(om, triangles):
    ommatidia_directions = get_ommatidia_directions()
    
    def find_neighbors(pindex, triang):
        neighbors = list()
        for simplex in triang.vertices:
            if pindex in simplex:
                neighbors.extend([simplex[i] for i in range(len(simplex)) if simplex[i] != pindex])
                '''
                this is a one liner for if a simplex contains the point we`re interested in,
                extend the neighbors list by appending all the *other* point indices in the simplex
                '''
        #now we just have to strip out all the dulicate indices and return the neighbors list:
        return list(set(neighbors))
        
        
    neighbors_tmp = find_neighbors(om, triangles)
    neighbors = []
    for i, neighbor in enumerate(neighbors_tmp):
        if ommatidia_on_same_eye(om, neighbor, ommatidia_directions):
            neighbors.append(neighbor)
    return neighbors
    
def get_second_order_delaunay_neighbors(om, triangles):
    first_order_neighbors = get_delaunay_neighbors(om, triangles)
    neighbors = []#copy.copy(first_order_neighbors)
    for neighbor in first_order_neighbors:
        second_order_neighbors = get_delaunay_neighbors(neighbor, triangles)
        neighbors.extend([n for n in second_order_neighbors if n not in neighbors and n not in first_order_neighbors])
    return neighbors
    
def precompute_delaunay_neighbors():
    triangles = get_delaunay_triangles()
    neighbors = {}
    for om in range(1398):
        n = get_second_order_delaunay_neighbors(om, triangles)
        neighbors.setdefault(om, n)
    return neighbors
    
def get_ommatidia_directions(buchner71 = 'receptor_directions_buchner71.csv', aslatlong=False):
    buchner_file = open(buchner71)
    ommatidia_directions = []
    for line in buchner_file.readlines():
        linesplit = line.split(',')
        d = [float(linesplit[i]) for i in range(len(linesplit))]
        ommatidia_directions.append( d )
    ommatidia_directions = np.array(ommatidia_directions)

    if aslatlong:
        ommatidia_directions = get_all_ommatidia_as_lat_long(ommatidia_directions)
        
    return ommatidia_directions
    
def get_ommatidia_to_pixel_map_for_camera(image_size, intrinsic_camera_matrix, buchner71=None, ommatidial_acceptance_angle=5, heading_shift=0):
    '''
    size - (int, int) corresponding to (width, height)
           units: mm
    focal length - (float)
           units: mm
    intrinsic_camera_matrix - np.array, or np.matrix
                                3x3 [[fx, 0 , cx],
                                    [0 , fy, cy],
                                    [0 , 0 , 0 ]]
                                where fx and fy are focal lengths
                                      cx and cy are the image center
    buchner71 - path to csv file containing normalized directions of each ommatidia
    ommatidial_acceptance_angle - degrees, default=5
    '''
    print 'Calculating ommatidia to pixel map\nThis will take some time.'
    
    
    if buchner71 is None:
        buchner71 = 'receptor_directions_buchner71.csv'
    ommatidia_directions = get_ommatidia_directions(buchner71)
    
    ommatidia_to_pixels = dict((om, []) for om in range(len(ommatidia_directions)))
    
    M = np.matrix(intrinsic_camera_matrix)
    MI = M.I
    rs = []
    
    theta = np.pi/2. + heading_shift
    Ry = np.matrix([[np.cos(theta),  0, np.sin(theta)],
                    [0,  1, 0],
                    [-1*np.sin(theta), 0, np.cos(theta)]])
                    
    theta = -np.pi/2.
    Rx = np.matrix([[1,  0, 0],
                    [0,  np.cos(theta), -1*np.sin(theta)],
                    [0, np.sin(theta), np.cos(theta)]])
                    
    
    for i in range(image_size[1]):
        for j in range(image_size[0]):
            c = np.matrix([[float(j)], [float(i)], [float(1)]])
            r = MI*c
            r /= np.linalg.norm(r)
            r = Rx*Ry*r
            
            r = np.array(r).reshape(3)
            rs.append(r) 
            # take dot product between r and each ommatidia
            d = ommatidia_directions*r
            s = np.sum(d, axis=1)
            angle_deg = np.arccos(s)*180/np.pi
            oms = np.where(angle_deg<ommatidial_acceptance_angle)[0]
            for om in oms:
                ommatidia_to_pixels[om].append([i,j])
                
    return ommatidia_to_pixels
    
def get_stereo_ommaps(image_size, intrinsic_camera_matrix, shift=np.pi/2.1):
    ommap_left = get_ommatidia_to_pixel_map_for_camera(image_size, intrinsic_camera_matrix, heading_shift=shift)
    ommap_right = get_ommatidia_to_pixel_map_for_camera(image_size, intrinsic_camera_matrix, heading_shift=-shift)
    return ommap_left, ommap_right

def get_all_ommatidia_values_from_rectilinear_image(img, ommap):
    '''
    img   - rectilinear image to convert
    ommap - ommatidia to pixel map for the given camera matrix and image size. 
                If None, it will be calculated (takes a few minutes)
              - see get_ommatidia_to_pixel_map_for_camera(
                                    image_size, 
                                    intrinsic_camera_matrix, 
                                    buchner71=None, 
                                    ommatidial_acceptance_angle=5)
    '''
    ommatidia_values = dict((om, np.array([0,0,0,1])) for om in ommap.keys())
    for om, pixels in ommap.items():
        if len(pixels) > 0:
            v = [img[pixels[i][0], pixels[i][1]] for i in range(len(pixels))]
            ommatidia_values[om] = np.mean(v, axis=0)
    return ommatidia_values
    
def get_ommatidia_value_from_rectilinear_image(img, pixels):
    if len(pixels) > 0:
        v = [img[pixels[i][0], pixels[i][1]] for i in range(len(pixels))]
        return np.mean(v, axis=0)
    else:
        return np.array([np.nan,np.nan,np.nan,1])

#######################################################################
# Speed up class
#######################################################################
class EquiRectangularFlyView:
    def __init__(self,  image_size=None, 
                        intrinsic_camera_matrix=None, 
                        ommap_left=None,
                        ommap_right=None, 
                        save_ommap='',
                        zombiemode=False,
                        SHOW_OMMATIDIA_NUMBERS=False,
                        SHOW_OMMATIDIA_EDGES=False):
        '''
        image_size - [source image width, source image height] 
                     equivalent to [source image columns, source image rows]
        intrinsic_camera_matrix - 3x3 np.array, or 3x3 np.matrix
                                - see get_simple_intrinsic_camera_matrix(
                                    focal_length, 
                                    sensor_width, 
                                    sensor_height)
        ommap - ommatidia to pixel map for the given camera matrix and image size. 
                If None, it will be calculated (takes a few minutes)
              - see get_ommatidia_to_pixel_map_for_camera(
                                    image_size, 
                                    intrinsic_camera_matrix, 
                                    buchner71=None, 
                                    ommatidial_acceptance_angle=5)
              - for stereo implementation, need left and right camera views, see get_stereo_ommaps
        zombiemode - allows use of this class simply for displaying and saving an image given all the pixel values directly
        
        Usage:
        
        Initialize as:
        flyview = EquiRectangularFlyView(image_size, intrinsic_camera_matrix)
        
        render images with:
        flyview.calc_fly_view_for_image_sequence(directory, destination, image_type='.png', output_height=720)        
        
        Notes:
        
        1. This class produces equirectangular projections of the fly view.
        
        2. It assumes that the flies eyes have the same center point, rather than the correct ~0.3mm distance between them. This limitation should only be a problem if objects are closer than a few mm away, and you wish to have a correct stereo view.
         
        3. All pixels that fall within the acceptance angle are treated equally. For sufficiently high resolution images, this should not be a problem.
        
        
        
        '''
        self.SHOW_OMMATIDIA_NUMBERS = SHOW_OMMATIDIA_NUMBERS
        self.SHOW_OMMATIDIA_EDGES = SHOW_OMMATIDIA_EDGES
        
        self.fig = plt.figure(figsize=(2,1))
        self.ax = self.fig.add_axes([0,0,1,1])
        self.fig.set_frameon(False)
        
        self.ax.set_axis_bgcolor('black')
        self.ax.set_xlim(-np.pi,np.pi)
        self.ax.set_ylim(-np.pi/2.,np.pi/2.)
        self.ax.set_aspect('equal')
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        self.calc_polygons()
        if zombiemode:
            return
            
        self.intrinsic_camera_matrix = intrinsic_camera_matrix
        if ommap_left is None:
            ommap_left, ommap_right = get_stereo_ommaps(image_size, intrinsic_camera_matrix, shift=np.pi/2.1)
            
        if len(save_ommap) > 0:
            save_ommap_left = os.path.join(save_ommap, 'ommap_left.pickle')
            save_ommap_right = os.path.join(save_ommap, 'ommap_right.pickle')
            f_left = open(save_ommap_left, 'w')
            f_right = open(save_ommap_right, 'w')
            pickle.dump(ommap_left, f_left)
            pickle.dump(ommap_right, f_right)
            f_left.close()
            f_right.close()
            
        self.ommap_left = ommap_left
        self.ommap_right = ommap_right
        
    def calc_polygons(self):                
        self.polygons = []
        self.centers = []
        if self.SHOW_OMMATIDIA_EDGES:
            edgecolor = 'black'
            linewidth = 0.1
        else:
            edgecolor = 'none'
            linewidth = 1
        for n, face in enumerate(hex_faces):
            new_face = get_equirectangular_hexagon(n, face)
            polygon = patches.Polygon(new_face, facecolor='black', edgecolor=edgecolor, linewidth=linewidth)
            self.polygons.append(polygon)
            self.ax.add_artist(polygon)
            
            center = np.mean(new_face, axis=0)
            self.centers.append(center)
            if self.SHOW_OMMATIDIA_NUMBERS:
                self.ax.text(center[0], center[1], str(n), horizontalalignment='center', verticalalignment='center', fontsize=1, color='red', zorder=1000)
            
    def calc_fly_view_for_image(self, img_left, img_right, filename, output_height=720):
        for n in range(len(self.ommap_left.keys())):
            left_pixels = self.ommap_left[n]
            right_pixels = self.ommap_right[n]
            
            color_left = get_ommatidia_value_from_rectilinear_image(img_left, left_pixels)
            color_right = get_ommatidia_value_from_rectilinear_image(img_right, right_pixels)
            
            print 'colors: ', color_left, color_right
            if np.sum(np.isnan(color_left)) > 0 and np.sum(np.isnan(color_right)) == 0:
                color = color_right
            elif np.sum(np.isnan(color_right)) > 0 and np.sum(np.isnan(color_left)) == 0:
                color = color_left
            elif np.sum(np.isnan(color_right)) > 0 and np.sum(np.isnan(color_left)) > 0:
                color = (0,0,0,1)
            else:
                color = np.mean([color_right, color_left], axis=0)
            print color
            
            if np.max(color) > 1:
                color = [c/255. for c in color]
            self.polygons[n].set_facecolor(color)
        self.fig.savefig(filename, format='png', dpi=int(output_height))
    
    def calc_fly_view_for_image_sequence(self, directory, destination, image_type='.png', output_height=720):
        left_match = 'left' + image_type
        right_match = 'right' + image_type
        imgfiles_left = get_filenames_from_directory(directory, match=left_match)
        imgfiles_right = get_filenames_from_directory(directory, match=right_match)
    
        ndigits = int(np.ceil(np.log(len(imgfiles_left))/np.log(10)))
        for i, imgfile_left in enumerate(imgfiles_left):
            imgfile_right = imgfiles_right[i]
            img_left = plt.imread(imgfile_left)
            img_right = plt.imread(imgfile_right)
            fname = str(i).zfill(ndigits) + '.png'
            filename = os.path.join(destination, fname) 
            self.calc_fly_view_for_image(img_left, img_right, filename, output_height=output_height)
            
    def save_fly_view_for_images(self, directory, destination, image_type='.png'):
        left_match = 'left' + image_type
        right_match = 'right' + image_type
        imgfiles_left = get_filenames_from_directory(directory, match=left_match)
        imgfiles_right = get_filenames_from_directory(directory, match=right_match)
        
        ndigits = int(np.ceil(np.log(len(imgfiles_left))/np.log(10)))
        flyviews = dict([(f,None) for f in range(len(imgfiles_left))])
        for i, imgfile_left in enumerate(imgfiles_left):
            imgfile_right = imgfiles_right[i]
            img_left = plt.imread(imgfile_left)
            img_right = plt.imread(imgfile_right)
            flyview = dict([(om,None) for om in range(len(self.ommap.keys()))])
            
            for n in range(len(self.ommap_left.keys())):
                left_pixels = self.ommap_left[n]
                right_pixels = self.ommap_right[n]
                
                color_left = get_ommatidia_value_from_rectilinear_image(img_left, left_pixels)
                color_right = get_ommatidia_value_from_rectilinear_image(img_right, right_pixels)
                
                if np.nan in color_left:
                    color = color_right
                elif np.nan in color_right:
                    color = color_left
                else:
                    color = np.mean([color_right, color_left])
                    
                flyview[n] = np.asscalar(color[0])
            flyviews[i] = flyview
        
        fname = 'flyview.pickle'
        fname = os.path.join(destination, fname)
        f = open(fname, 'w')
        pickle.dump(flyviews, f)
        f.close()        
        
    def display_fly_view_given_colors(self, colors, filename, output_height=720):
        '''
        useful for zombiemode=True
        colors - dictionary relating ommatidia number to grayscale color value
        '''
        if type(colors) is dict:
            rgb_colors = [[colors[i],colors[i],colors[i]] for i in range(len(colors.keys()))]
            rgb_colors = np.array(rgb_colors)
        else:
            rgb_colors = [[colors[i],colors[i],colors[i]] for i in range(len(colors))]
            rgb_colors = np.array(rgb_colors)
        
        ind_not_nan = np.where(np.isnan(rgb_colors)==False)
            
        rgb_colors -= np.min(rgb_colors[ind_not_nan])
        rgb_colors /= np.max(rgb_colors[ind_not_nan])
        
        ind = np.where(np.isnan(rgb_colors))
        rgb_colors[ind] = 0
        
        for n in range(rgb_colors.shape[0]):
            self.polygons[n].set_facecolor(rgb_colors[n,:])
        self.fig.savefig(filename, format='png', dpi=int(output_height))
        
if __name__ == '__main__':
    
    parser = OptionParser()
    parser.add_option("--directory", type="str", dest="directory", default='',
                        help="directory of rectilinear images you would like to turn into fly views")
    parser.add_option("--destination", type="str", dest="destination", default='',
                        help="destination where to save fly views")
    parser.add_option("--image", type="str", dest="image", default='',
                        help="rectilinear image to turn into a fly view. For an image sequence, use directory and destination instead")
    parser.add_option("--output", type="str", dest="output", default='',
                        help="path and filename where to save fly view. For an image sequence, use directory and destination instead")                  
    parser.add_option("--focal-length", type="float", dest="focal_length", default=20,
                        help="camera focal length used to create the rectilinear image, mm")
    parser.add_option("--sensor-width", type="float", dest="sensor_width", default=1024,
                        help="sensor width of the camera used to create the rectilinear image, mm")
    parser.add_option("--sensor-height", type="float", dest="sensor_height", default=720,
                        help="sensor height of the camera used to create the rectilinear image, mm")
    parser.add_option("--output-height", type="int", dest="output_height", default=720,
                        help="output height in pixels for fly view")
    parser.add_option("--input-type", type="str", dest="input_type", default='png',
                        help="input image type, e.g. png or jpeg")
    parser.add_option("--mat", type="str", dest="mat", default='',
                        help="filename of a csv file describing the intrinsic camera matrix (single line, 9 entries, comma seperated)")
    parser.add_option("--save-ommap", type="str", dest="save_ommap", default='',
                        help="destination where ommap will be saved, default is empty string, which does NOT save ommap")
    parser.add_option("--ommap", type="str", dest="ommap", default='',
                        help="directory where files ommap_left.pickle and ommap_right.pickle can be found")
    parser.add_option("--save-data", type="int", dest="save_data", default=0,
                        help="save ommatidia maps, but not images")
    parser.add_option("--show-ommatidia-numbers", type="int", dest="show_ommatidia_numbers", default=0,
                        help="write ommatidia numbers inside each hexagon, default is 0 aka False")
    parser.add_option("--show-ommatidia-edges", type="int", dest="show_ommatidia_edges", default=0,
                        help="draw edges of each hexagon in black, default is 0 aka False")
    (options, args) = parser.parse_args()
    
    def get_ommap(ommap, side):
        match = 'ommap_' + side + '.pickle'
        ommap_left_filename = get_filenames_from_directory(ommap, match=match)
        if len(ommap_left_filename) > 0:
            f = open(ommap_left_filename[0])
            ommap_left = pickle.load(f)
            f.close()
        else:
            s = 'Could not find ommap_' + side + '.pickle'
            raise ValueError(s)
        return ommap_left
            
    if len(options.ommap) > 0:
        ommap_left = get_ommap(options.ommap, 'left')
        ommap_right = get_ommap(options.ommap, 'right')
    else:
        ommap_left = None
        ommap_right = None
        
    if len(options.mat) > 0:
        intrinsic_camera_matrix = load_intrinsic_camera_matrix_from_csv_file(options.mat)
    else:
        intrinsic_camera_matrix = get_simple_intrinsic_camera_matrix(options.focal_length, options.sensor_width, options.sensor_height)
    
    if len(options.image) > 0:
        img = plt.imread(options.image)
        flyview = EquiRectangularFlyView([img.shape[1], img.shape[0]], intrinsic_camera_matrix, save_ommap=options.save_ommap, ommap_left=ommap_left, ommap_right=ommap_right, SHOW_OMMATIDIA_NUMBERS=options.show_ommatidia_numbers, SHOW_OMMATIDIA_EDGES=options.show_ommatidia_edges)
        flyview.calc_fly_view_for_image(img, img, options.output, options.output_height)
    
    elif len(options.directory) > 0 and len(options.destination) > 0:
        imgfiles = get_filenames_from_directory(options.directory, match=options.input_type)
        img = plt.imread(imgfiles[0])
        flyview = EquiRectangularFlyView([img.shape[1], img.shape[0]], intrinsic_camera_matrix, save_ommap=options.save_ommap, ommap_left=ommap_left, ommap_right=ommap_right, SHOW_OMMATIDIA_NUMBERS=options.show_ommatidia_numbers, SHOW_OMMATIDIA_EDGES=options.show_ommatidia_edges)
        
        if options.save_data:
            flyview.save_fly_view_for_images(options.directory, options.destination)
        else:
            flyview.calc_fly_view_for_image_sequence(options.directory, options.destination, output_height=options.output_height)
    
    
    
        
