import numpy as np
from pathlib import Path

def create_dummy_dataset():
    root = Path("dummy_data")
    print(f"Creating synthetic KTC dataset at {root.absolute()}...")
    
    for level in [1, 7]:
        level_dir = root / f"level_{level}"
        level_dir.mkdir(parents=True, exist_ok=True)
        
        # Create 15 samples per level
        for i in range(15):
            data = np.random.rand(256) # Mock EIT measurement vector
            
            # For Level 7, randomly inject missing values (NaNs) to test our FeatureTransformer GP logic
            if level == 7 and i % 3 == 0:
                data[np.random.choice(256, 40, replace=False)] = np.nan
                
            np.save(level_dir / f"sample_{i}.npy", data)
            
    print("Done! Dummy dataset is ready.")

if __name__ == "__main__":
    create_dummy_dataset()
