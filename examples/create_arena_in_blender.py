import numpy as np
 
def delete_all_objects():
    for key in bpy.data.objects.keys():
        bpy.ops.object.select_pattern(pattern=key)
        bpy.ops.object.delete()

def delete_all_cameras():
    for key in bpy.data.cameras.keys():
        bpy.data.cameras.remove(bpy.data.cameras[key])
 
def make_material(name, emit, diffuse=(1,1,1), specular=(1,1,1), alpha=1):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = diffuse
    mat.diffuse_shader = 'LAMBERT' 
    mat.diffuse_intensity = 1.0 
    mat.specular_color = specular
    mat.specular_shader = 'COOKTORR'
    mat.specular_intensity = 0.5
    mat.alpha = alpha
    mat.ambient = 1
    mat.emit = emit
    return mat

def make_reflective_material(name):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (1,1,1)
    mat.diffuse_shader = 'LAMBERT' 
    mat.diffuse_intensity = 1.0 
    mat.specular_color = (1,1,1)
    mat.specular_shader = 'COOKTORR'
    mat.specular_intensity = 0.5
    mat.alpha = 0.2
    mat.ambient = 1
    mat.emit = 0
    mat.use_transparency = 1
    mat.raytrace_mirror.use = 1
    mat.raytrace_mirror.reflect_factor = 0.8
    return mat
    
def make_metal_mesh_material(name):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (.8,.8,.8)
    mat.diffuse_shader = 'LAMBERT' 
    mat.diffuse_intensity = 0.8
    mat.specular_color = (1,1,1)
    mat.specular_shader = 'COOKTORR'
    mat.specular_intensity = 1
    mat.alpha = 1
    mat.ambient = 1
    mat.emit = 0
    mat.use_transparency = 0
    mat.raytrace_mirror.use = 1
    mat.raytrace_mirror.reflect_factor = 0.5
    mat.raytrace_mirror.gloss_factor = 0.3
    mat.specular_hardness = 16
    return mat

def create_rectangle(location, dimensions, material):
    bpy.ops.mesh.primitive_cube_add()   
    rectangle_name = bpy.context.scene.objects.active.name
    bpy.data.objects[rectangle_name].location = location
    bpy.data.objects[rectangle_name].dimensions = dimensions
    bpy.context.object.data.materials.append(material)    

def create_downwind_dot_rectangle(material, scale_factor):
    print('making dot')
    location_xy = [(0.31+.025/2.)*scale_factor, 0*scale_factor, -.16*scale_factor]
    dimension_xy = [0.025*scale_factor, 0.025*scale_factor, 0.001]
    create_rectangle(location_xy, dimension_xy, material)
    
def create_windtunnel(scale_factor):
    gray = make_material('gray', 1, diffuse=(0.5,0.5,0.5), specular=(0.5,0.5,0.5))
    floor = create_rectangle([0.6*scale_factor, 0, -0.17*scale_factor-.001], [1.4*scale_factor, 0.34*scale_factor, 0.001*scale_factor], gray)
    wall1 = create_rectangle([0.6*scale_factor, -0.17*scale_factor-.001, 0*scale_factor], [1.4*scale_factor, 0.001*scale_factor, 0.34*scale_factor], gray)
    wall2 = create_rectangle([0.6*scale_factor, 0.17*scale_factor+.001, 0*scale_factor], [1.4*scale_factor, 0.001*scale_factor, 0.34*scale_factor], gray)
    glass = make_reflective_material('glass')
    ceiling = create_rectangle([0.6*scale_factor, 0, 0.17*scale_factor+.001], [1.4*scale_factor, 0.34*scale_factor, 0.001*scale_factor], glass)
    
    mesh = make_metal_mesh_material('mesh')
    upwind_wall = create_rectangle([-.1*scale_factor, 0, 0], [.001*scale_factor, 0.34*scale_factor, 0.34*scale_factor], mesh)
    
def create_singledot(scale_factor=1):
    create_windtunnel(scale_factor)
    black = make_material('black', 0, diffuse=(0,0,0), specular=(0,0,0))
    create_downwind_dot_rectangle(black, scale_factor)
    
if __name__ == '__main__':
    
    delete_all_objects()
    delete_all_cameras()
    
    create_singledot(scale_factor=1)
    
