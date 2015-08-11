Instructions
==============

For a simple static rendering of a 2D scene with a single image
-------------

Ignore the blender stuff. From the fly_view directory, simple run the script as:

python ./fly_view_stereo.py --image=IMAGE_FILENAME --output=OUTPUT_FILENAME(must be .png)

if you intend to render more than a single image, save the ommap's with command:
 --save-ommap=DIRECTORY_WHERE_TO_SAVE
 
then, next time, add the command:
 --ommap=DIRECTORY_WHERE_OMMAPS_SAVED
 
Introduction
-------------

The scripts in this package use blender, a free open source 3D modelling program, to render fly-view stills and movies. It is a bit roundabout and inelegant, but it works. The code uses blender to render the scene from two cameras for every desired point along a trajectory. These images are then converted to fly-view images. The entire process does take quite some time. 

Much of the code is thanks to Dr. Andrew Straw, and based on the measurements made by Buchner (1971).
 
Download and setup blender
--------------
1. Download blender: http://www.blender.org/
2. Consider learning some of the basics
3. Open blender, and in the upper right corner of the main window, right click and drag left to create a second window
4. In the bottom left corner of that new window, click on the small 3D cube icon, and select "python console" from the pop up list

Creating the scene in blender
--------------
1. Enter the following at the python command prompt, which assigns the correct file to the variable filename:
    ```python
    filename = "~/PathToGeneratorScript.py"
    ```
    
    For example, where PATH is the path to the FlyView directory:
    
    ```python
    filename = "PATH/examples/create_arena_in_blender.py"
    ```

2. Execute the filename:
    ```python
    exec(compile(open(filename).read(), filename, 'exec'))
    ```

3. Explore the scene in the left window

Creating the stereo camera in blender, and rendering a trajectory
--------------
1. Enter the following at the python command prompt, where PATH is the path to the FlyView directory
    ```python
    filename = "PATH/blender/create_wideangle_camera_stereo.py"
    ```
    
2. Execute the file, this creates the object "stereocamera," which is associated with two 180 degree cameras: 
    ```python
    exec(compile(open(filename).read(), filename, 'exec'))
    ```
    
3. Run the following code in the command prompt, where "filename" is the filename of your desired trajectory (see below for format), and "destination" is the path to a (empty) directory where you wish to have the images saved
    ```python
    stereocamera.render_trajectory(filename, destination)
    ```
    
    For example, to run the example trajectory and save the images run the following, where PATH is the path to the FlyView directory:
    ```python
    filename = "PATH/examples/example_trajectory.csv"
    destination = "PATH/examples/rectilinear"
    stereocamera.render_trajectory(filename, destination)
    ```
    
    This will run a script that moves the stereocamera through the scene to each of the specified points, takes two images, and saves them to disk. Additionally, intrinsic camera matrices for the stereocamera will be saved to the same directory.

4. You can now close blender

### Trajectory file format
The file describing the trajectory should be a simple comma seperated value file that follows the following convention:
x_position, y_position, z_position, heading_in_radians

    
Rendering the Fly-View from the rectilinear images
--------------

### Installing the flyview package (optional)
If you have not yet installed the flyview package, you can do so by running the following from within the FlyView directory:
```python
python ./setup.py
```

### Rendering a flyview trajectory for the first time, this takes a while
From inside the FlyView/flyview directory, run:
```python
python ./fly_view_stereo.py --directory=PATH_TO_IMAGES_FROM_BLENDER --destination=PATH_TO_EMPTY_DIRECTORY_TO_SAVE_FLYVIEW_IMAGES --mat=INTRINSIC_CAMERA_MATRIX --save-ommap=PATH_WHERE_TO_SAVE_CALIBRATION_FOR_FUTURE_USE
```

For example, where PATH is the path to the FlyView directory:
```python
python ./fly_view_stereo.py --directory="PATH/examples/rectilinear" --destination="PATH/examples/flyview" --mat="PATH/examples/rectilinear/intrinsic_camera_matrix_left.csv" --save-ommap="PATH/examples/rectilinear"
```

### Rendering a flyview trajectory given a pre-existing calibration, this is much faster, but still takes time
```python
python ./fly_view_stereo.py --directory=PATH_TO_IMAGES_FROM_BLENDER --destination=PATH_TO_EMPTY_DIRECTORY_TO_SAVE_FLYVIEW_IMAGES --ommap=PATH_TO_DIRECTORY_WHERE_OMMAP_CAN_BE_FOUND
```

Note, ommap points to the *directory* where ommap_left.pickle and ommap_right.pickle can be found, not the actual file(s)






    
    
    
    
    

