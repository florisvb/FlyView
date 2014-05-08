Instructions
==============

The scripts in this package use blender, a free open source 3D modelling program, to render fly-view stills and movies. It is a bit roundabout and inelegant, but it works. The code uses blender to render the scene from two cameras for every desired point along a trajectory. These images are then converted to fly-view images. The entire process does take quite some time. 
 
Download and setup blender
--------------
1. Download blender: http://www.blender.org/
2. Consider learning some of the basics
3. Open blender, and in the upper right corner of the main window, right click and drag left to create a second window
4. In the bottom left corner of that new window, click on the small 3D cube icon, and select "python console" from the pop up list

Creating the scene in blender
--------------
1. Enter the following at the python command prompt, which assigns the correct file to the variable filename:
    <blockquote>
        <p>filename = "~/PathToGeneratorScript.py"</p>
    </blockquote>
    
    For example, where PATH is the path to the FlyView directory:
    
    <blockquote>
        <p>filename = "PATH/examples/create_arena_in_blender.py"</p>
    </blockquote>

2. Execute the filename:
    <blockquote>
        <p>exec(compile(open(filename).read(), filename, 'exec'))</p>
    </blockquote>

3. Explore the scene in the left window

Creating the stereo camera in blender, and rendering a trajectory
--------------
1. Enter the following at the python command prompt, where PATH is the path to the FlyView directory
    <blockquote>
        <p>filename = "PATH/blender/create_wideangle_camera_stereo.py"</p>
    </blockquote>
2. Execute the file, this creates the object "stereocamera," which is associated with two 180 degree cameras: 
    <blockquote>
        <p>exec(compile(open(filename).read(), filename, 'exec'))</p>
    </blockquote>
3. Run the following code in the command prompt, where "filename" is the filename of your desired trajectory (see below), and "destination" is the path to a (empty) directory where you wish to have the images saved
    <blockquote>
        <p>stereocamera.render_trajectory(filename, destination)</p>
    </blockquote>
    
    For example, to run the example trajectory and save the images run the following, where PATH is the path to the FlyView directory:
    <blockquote>
        <p>filename = "PATH/examples/example_trajectory.csv"
        destination = "PATH/examples/rectilinear"
        stereocamera.render_trajectory(filename, destination)
        </p>
    </blockquote>
    
    
    
    
    
    
    

