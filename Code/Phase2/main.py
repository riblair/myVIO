"""Da whole project...
    * Have a trajectory to follow
        * Will want multiple, so we should generalize this process
        * See that one paper for details
        * Be able to graph trajectories (both GT and estimated)
    * Have the ability to calculate IMU and Gyro values
        * Should be handled by ImuUtils so long as we give them a list of positions. 
    * Create a uniform list of positions between start and end of trajectory
        * figure out how to modulate orientation during this in a way that makes sense....
        * for times sake this might just be zero pitch and roll, where yaw is given by projected direction vector?
    * Create blender pipeline for moving camera, capturing image, saving the other data to a txt file
        * Steal some code from Proj 4 Phase 1, Steal some other code from Project 3
    
    * Model Architecture
        * See papers and such...
        * Need 3 seperate ones...
    * Loss Function
        * See papers and such...
        * Need 3 seperate ones...
    * Training Pipeline
        * Steal code from any such deep-learning phase (proj 2?)
        * Need 3 seperate ones...
        * VERY IMPORTANT, record losses
    * Get everything on the Turing Cluster. 
    * Testing file, 
        * Load model, run it through out-of-dataset trajectory, generate txt file
        * Generate graphs
"""