import numpy as np
import scipy.io

anomaly_mask_path = "../data/xgbod_out.npy"  
original_mat_path = "/share/landseer/xgbod_v2/datasets/cifar-10-batches-mat/test_batch.mat"  
cleaned_mat_path = "../data/xgbod_cleaned_data.mat"  

anomaly_mask = np.load(anomaly_mask_path)  

mat_data = scipy.io.loadmat(original_mat_path)

key = list(mat_data.keys())[-1]  
data = mat_data[key]  

filtered_data = data[anomaly_mask == 0] 

scipy.io.savemat(cleaned_mat_path, {key: filtered_data})

print(f"Filtered data saved to {cleaned_mat_path}")