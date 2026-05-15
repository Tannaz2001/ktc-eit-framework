import logging
import numpy as np
import pprint

# Configure standard logging for the test run
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s [%(name)s]: %(message)s')

# Import the new plugins
from src.ktc_framework.methods.level_set_plugin import LevelSetPlugin
from src.ktc_framework.methods.hull_plugin import HullPlugin

def run_tests():
    print("=======================================")
    print("       TESTING LEVEL SET PLUGIN        ")
    print("=======================================")
    level_set = LevelSetPlugin()
    
    # Create a synthetic "reconstruction" image (256x256)
    reconstruction = np.zeros((256, 256))
    Y, X = np.ogrid[:256, :256]
    dist_from_center = np.sqrt((X - 128)**2 + (Y - 128)**2)
    
    # Set the circle area to 1.0
    reconstruction[dist_from_center <= 50] = 1.0  
    
    ls_result = level_set.run(reconstruction)
    print(f"Number of objects found: {ls_result['n_objects']}")


    print("\n=======================================")
    print("          TESTING HULL PLUGIN          ")
    print("=======================================")
    hull = HullPlugin()
    
    # Create a synthetic "segmentation" map (256x256)
    segmentation = np.zeros((256, 256), dtype=int)
    
    # Add one object of type 1
    segmentation[30:70, 30:70] = 1 
    
    # Add two objects of type 2
    segmentation[150:200, 150:200] = 2
    segmentation[50:80, 200:230] = 2
    
    print("--> Extracting features for Target Label 1:")
    features_1 = hull.run(segmentation, target_label=1)
    pprint.pprint(features_1)
    
    print("\n--> Extracting features for Target Label 2:")
    features_2 = hull.run(segmentation, target_label=2)
    pprint.pprint(features_2)

if __name__ == "__main__":
    run_tests()
