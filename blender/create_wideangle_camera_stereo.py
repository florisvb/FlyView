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
import numpy as np
import time
import os

class Camera:
    def __init__(self, location=[2,0,0], camera_name='camera1'):
        self.camera_name = camera_name
        bpy.ops.object.camera_add(location=location)
        self.name = bpy.context.scene.objects.active.name
        camera = bpy.data.objects[self.name]
        camera.scale = (.2,.2,.2) # make the cameras look small
        bpy.context.scene.objects.active = bpy.data.objects[self.name]
        bpy.data.cameras[self.name].lens_unit = 'FOV'
        bpy.data.cameras[self.name].angle = 170*np.pi/180.
        bpy.data.cameras[self.name].clip_start = 0.001
        bpy.data.cameras[self.name].clip_end = 5000
        bpy.data.objects[self.name].rotation_euler = Euler((np.pi/2.,0,np.pi/2.), 'XYZ')
        #bpy.data.objects[self.name].rotation_euler = Euler((-np.pi/2.,0,0), 'XYZ')
        
    def save_camera_view(self, resolution=[1024,720], destination=None, n=0):
        if destination is None:
            destination = '~/blender_tmps/'
        tmp_filename = 'tmp_blender_' + str(n) + '_' + self.camera_name + '.png'
        filename = os.path.join(destination, tmp_filename)
        #print(tmp_filename)
        bpy.context.scene.render.filepath = filename
        bpy.context.scene.render.resolution_x = resolution[0]
        bpy.context.scene.render.resolution_y = resolution[1]
        bpy.context.scene.camera = bpy.data.objects[self.name]
        bpy.ops.render.render(write_still=True)
        
    def set_position(self, location):
        bpy.data.objects[self.name].location = location
        
    def set_xy_heading(self, angle):
        bpy.data.objects[self.name].rotation_euler = Euler((np.pi/2.,0,angle), 'XYZ')
            
    def get_intrinsic_camera_matrix(self):
        '''
        returns - the intrinsic camera matrix
        '''
        # init stuff
        scn    = bpy.context.scene
        width  = scn.render.resolution_x * scn.render.resolution_percentage / 100.0
        height = scn.render.resolution_y * scn.render.resolution_percentage / 100.0
        camData = bpy.data.objects[ self.name ].data
        ratio = width/height
        # assemble intrinsic matrix
        K = Matrix().to_3x3()
        K[0][0] = (width/2.0) / tan(camData.angle/2.0)
        K[1][1] = (height/2.0) / tan(camData.angle/2.0) * ratio
        K[0][2] = width  / 2.0
        K[1][2] = height / 2.0
        K[2][2] = 1.0
        return np.array(K)

    def render_trajectory(self, filename, destination, nskip=0, scale_factor=1, heading_shift=0):
        '''
        filename - path and name for comma seperated text file where each line describes the 3D positions of the camera
        destination - where to save the rendered images to
        nskip - how many frames to skip between rendered frames (default is none)
        heading_shift - used for StereoCamera, ignore unless using StereoCamera, in which case it is taken care of automatically
        '''
        f = open(filename, 'r')
        positions = []
        angles = []
        for line in f.readlines():
            line = line.strip().rstrip(',').split(',')
            val = [float(v)*scale_factor for v in line[0:3]]
            positions.append(val)
            angles.append(float(line[3]))
        f.close()
        ndigits = int(np.ceil(np.log(len(positions)) / np.log(10)))
        for i in range(0, len(positions), nskip+1):
            self.set_position(positions[i])
            self.set_xy_heading(angles[i]+heading_shift)
            self.save_camera_view(destination=destination, n=str(i).zfill(ndigits))
            
        K_file_name = 'intrinsic_camera_matrix' + '_' + self.camera_name + '.csv'
        K_file = os.path.join(destination, K_file_name)
        f = open(K_file, 'w')
        K = self.get_intrinsic_camera_matrix()
        K = K.reshape(9)
        s = ''
        for k in K:
            s += str(k) + ','
        s.strip()
        f.writelines(s)
        f.close()
    
class StereoCamera:
    def __init__(self, shift=np.pi/2.1):
        self.camera1 = Camera(camera_name='left')
        self.camera1.set_xy_heading(np.pi/2.-shift)
        self.camera2 = Camera(camera_name='right')
        self.camera2.set_xy_heading(np.pi/2.+shift)
        self.shift = shift
    def render_trajectory(self, filename, destination, nskip=0, scale_factor=1):
        self.camera1.render_trajectory(filename, destination, nskip=0, scale_factor=1, heading_shift=-self.shift)
        self.camera2.render_trajectory(filename, destination, nskip=0, scale_factor=1, heading_shift=self.shift)
    
def delete_all_objects():
    for key in bpy.data.objects.keys():
        bpy.ops.object.select_pattern(pattern=key)
        bpy.ops.object.delete()

def delete_all_cameras():
    for key in bpy.data.cameras.keys():
        bpy.data.cameras.remove(bpy.data.cameras[key])
        
        
def generate_trajectory_to_scan_world(filename='flyview_scan_1cm.csv', resolution=0.01):
    f = open(filename, 'w')
    xvals = np.arange(-.2, 1.3, resolution)
    yvals = np.arange(-.15, .15, resolution)
    zvals = np.arange(-.15, .15, resolution)
    n = 0
    for x in xvals:
        for y in yvals:
            for z in zvals:
                line = str(x) + ',' + str(y) + ',' + str(z) + ',' + '1.57' + '\n'
                f.writelines(line)
                n += 1
    
    print(n)
    print('N hours to process: ', n*2./3600.)
    f.close()
    
if __name__ == '__main__':
    '''
    to run:
    filename = "/home/caveman/BDocuments/src/python/FlyView/blender_scripts/create_wideangle_camera_stereo.py"
    exec(compile(open(filename).read(), filename, 'exec'))
    
    to render a trajectory:
    stereocamera.render_trajectory(filename, destination)
    '''
    
    stereocamera = StereoCamera()
    
