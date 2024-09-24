import h5py
import sys
import numpy as np

def analyze_h5_file(file_path):
    try:
        # Open the HDF5 file
        with h5py.File(file_path, 'r') as f:
            print(f"Analyzing file: {file_path}\n")
            print(f"Found {len(f.keys())} dataset(s) in the file:\n")
            """
            # Iterate through all datasets in the file
            events = h5py.File(file_path, 'r')
            x = events['x'][:]
            y = events['y'][:]
            p = events['p'][:]
            t = events['t'][:]
            table = np.array([x,y,p,t])
            table2=table.transpose()
            #np.savetxt('eventler.txt', table2, delimiter=',')
            """
            for key in f.keys():
                dataset = f[key]
                print(f"Dataset Name: {key}")
                print(f" - Shape: {dataset.shape}")
                print(f" - Data Type: {dataset.dtype}")
                
                # Print a few lines of data
                num_rows = dataset.shape[0]
                print(f" - Number of Rows: {num_rows}")

                # Display first few rows (if dataset is not too large)
                num_preview = min(5, num_rows)  # Show up to 5 rows
                print(f" - First {num_preview} rows of data:")
                print(dataset[num_preview:])

                # Print the first element of the dataset
                if num_rows > 0:
                    first_element = dataset[0]
                    print(f" - First element of the dataset: {dataset[0]}")
                    print(f" - Shape of the first element: {first_element.shape if hasattr(first_element, 'shape') else 'N/A'}")
                
                
    except Exception as e:
        print(f"An error occurred while analyzing the file: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python h5_reader.py <path_to_h5_file>")
    else:
        file_path = sys.argv[1]
        analyze_h5_file(file_path)
