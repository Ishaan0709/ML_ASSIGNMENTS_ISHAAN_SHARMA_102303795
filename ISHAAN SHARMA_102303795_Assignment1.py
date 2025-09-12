"ques 1(a)"
import numpy as np

arr=np.array([1,2,3,6,4,5])
print("Original : ",arr)

print("Reversed (slicing):", arr[::-1])

"ques 1(b)"
array1 = np.array([[1, 2, 3], [2, 4, 5], [1, 2, 3]])
print("Original:\n", array1)

print("Flattened (flatten):", array1.flatten())

"ques 1(c)"
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[1, 2], [3, 4]])

print("Equal:", np.array_equal(arr1, arr2))   # True
print("Element-wise comparison:\n", arr1 == arr2)

"ques 1(d)"
# i
x = np.array([1,2,3,4,5,1,2,1,1,1])
val_x = np.bincount(x).argmax()
indices_x = np.where(x == val_x)
print("x -> Most frequent:", val_x, "Indices:", indices_x)

# ii
y = np.array([1, 1, 1, 2, 3, 4, 2, 4, 3, 3])
val_y = np.bincount(y).argmax()
indices_y = np.where(y == val_y)
print("y -> Most frequent:", val_y, "Indices:", indices_y)

"ques 1(e)"
gfg = np.matrix('[4, 1, 9; 12, 3, 1; 4, 5, 6]')
print("Matrix:\n", gfg)

print("Sum of all elements:", gfg.sum())
print("Row-wise sum:", gfg.sum(axis=1))
print("Column-wise sum:", gfg.sum(axis=0))

"ques 1(f)"
n_array = np.array([[55, 25, 15], [30, 44, 2], [11, 45, 77]])

print("Sum of diagonal elements:", np.trace(n_array))

# Eigen values & vectors
eigen_values, eigen_vectors = np.linalg.eig(n_array)
print("Eigenvalues:", eigen_values)
print("Eigenvectors:\n", eigen_vectors)

# Inverse
print("Inverse:\n", np.linalg.inv(n_array))

# Determinant
print("Determinant:", np.linalg.det(n_array))

"ques 1(g)"
# First case
p1 = np.array([[1, 2], [2, 3]])
q1 = np.array([[4, 5], [6, 7]])

print("Multiplication (p1*q1):\n", np.dot(p1, q1))
print("Covariance:\n", np.cov(p1.flatten(), q1.flatten()))

# Second case
p2 = np.array([[1, 2], [2, 3], [4, 5]])
q2 = np.array([[4, 5, 1], [6, 7, 2]])

print("Multiplication (p2*q2):\n", np.dot(p2, q2))
print("Covariance:\n", np.cov(p2.flatten(), q2.flatten()))

"ques 1(h)"
x = np.array([[2, 3, 4], [3, 2, 9]])
y = np.array([[1, 5, 0], [5, 10, 3]])

print("Inner product:\n", np.inner(x, y))
print("Outer product:\n", np.outer(x, y))

# Cartesian product using meshgrid + reshape
cartesian = np.array(np.meshgrid(x.flatten(), y.flatten())).T.reshape(-1, 2)
print("Cartesian product:\n", cartesian)

"ques 2(a)"
import numpy as np
array = np.array([[1, -2, 3], [-4, 5, -6]])
# i
print("Absolute values:\n", np.abs(array))

# ii
# Flattened array
print("Flattened percentiles:")
print("25th:", np.percentile(array, 25))
print("50th:", np.percentile(array, 50))
print("75th:", np.percentile(array, 75))

# Column-wise
print("\nColumn-wise percentiles:")
print("25th:", np.percentile(array, 25, axis=0))
print("50th:", np.percentile(array, 50, axis=0))
print("75th:", np.percentile(array, 75, axis=0))

# Row-wise
print("\nRow-wise percentiles:")
print("25th:", np.percentile(array, 25, axis=1))
print("50th:", np.percentile(array, 50, axis=1))
print("75th:", np.percentile(array, 75, axis=1))

# iii
# Flattened
print("\nFlattened:")
print("Mean:", np.mean(array))
print("Median:", np.median(array))
print("Std Dev:", np.std(array))

# Column-wise
print("\nColumn-wise:")
print("Mean:", np.mean(array, axis=0))
print("Median:", np.median(array, axis=0))
print("Std Dev:", np.std(array, axis=0))

# Row-wise
print("\nRow-wise:")
print("Mean:", np.mean(array, axis=1))
print("Median:", np.median(array, axis=1))
print("Std Dev:", np.std(array, axis=1))


"ques 2(b)"
a = np.array([-1.8, -1.6, -0.5, 0.5, 1.6, 1.8, 3.0])
print("Floor:", np.floor(a))
print("Ceil:", np.ceil(a))
print("Truncated:", np.trunc(a))
print("Rounded:", np.round(a))

"ques 3(a)"
import numpy as np
array = np.array([10, 52, 62, 16, 16, 54, 453])
print("Sorted array:", np.sort(array))
print("Indices of sorted array:", np.argsort(array))
print("4 smallest elements:", np.sort(array)[:4])
print("5 largest elements:", np.sort(array)[-5:])

"ques 3(b)"
array2 = np.array([1.0, 1.2, 2.2, 2.0, 3.0, 2.0])
integers = array2[array2 == array2.astype(int)]
print("Integer elements only:", integers)
floats = array2[array2 != array2.astype(int)]
print("Float elements only:", floats)

"ques 4(a)"
from PIL import Image
import numpy as np
import os

print("Current Working Directory:", os.getcwd())

def img_to_array(path, save_path="image_array.txt"):
   
    img = Image.open(path)
    arr = np.array(img)
    
    if arr.ndim == 2:
        print("Grayscale Image Detected")
        np.savetxt(save_path, arr, fmt='%d')
    elif arr.ndim == 3 and arr.shape[2] == 3:
        print("RGB Image Detected")
        np.savetxt(save_path, arr.reshape(-1, 3), fmt='%d')
    else:
        print("Unsupported image format")
        return None
    
    print(f"Image saved as array in: {save_path}")
    return arr

arr = img_to_array("sample.png", "image_array.txt")

"ques 4(b)"
def load_array(file_path):
    data = np.loadtxt(file_path, dtype=int)
    print("Loaded array shape:", data.shape)
    return data

# Function call
loaded_arr = load_array("image_array.txt")
print(loaded_arr)


