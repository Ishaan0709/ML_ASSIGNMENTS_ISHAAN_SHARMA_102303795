import numpy as np
from PIL import Image
import os

# Q1(a) reverse
arr = np.array([1,2,3,6,4,5])
print("Original : ", arr)
print("Reversed (slicing):", arr[::-1])

# Q1(b) flatten
array1 = np.array([[1,2,3],[2,4,5],[1,2,3]])
print("Original:\n", array1)
print("Flattened (flatten):", array1.flatten())
print("Flattened (ravel):", array1.ravel())

# Q1(c) compare
arr1 = np.array([[1,2],[3,4]])
arr2 = np.array([[1,2],[3,4]])
print("Equal:", np.array_equal(arr1, arr2))
print("Element-wise comparison:\n", arr1 == arr2)

# Q1(d) most frequent value and indices
x = np.array([1,2,3,4,5,1,2,1,1,1])
val_x = np.bincount(x).argmax()
indices_x = np.where(x == val_x)[0]
print("x -> Most frequent:", val_x, "Indices:", indices_x)

y = np.array([1,1,1,2,3,4,2,4,3,3])
val_y = np.bincount(y).argmax()
indices_y = np.where(y == val_y)[0]
print("y -> Most frequent:", val_y, "Indices:", indices_y)

# Q1(e) matrix sums
gfg = np.matrix('[4, 1, 9; 12, 3, 1; 4, 5, 6]')
print("Matrix:\n", gfg)
print("Sum of all elements:", gfg.sum())
print("Row-wise sum:", gfg.sum(axis=1))
print("Column-wise sum:", gfg.sum(axis=0))

# Q1(f) linear algebra
n_array = np.array([[55,25,15],[30,44,2],[11,45,77]])
print("Sum of diagonal elements:", np.trace(n_array))
e_vals, e_vecs = np.linalg.eig(n_array)
print("Eigenvalues:", e_vals)
print("Eigenvectors:\n", e_vecs)
print("Inverse:\n", np.linalg.inv(n_array))
print("Determinant:", np.linalg.det(n_array))

# Q1(g) multiply and covariance
p1 = np.array([[1,2],[2,3]])
q1 = np.array([[4,5],[6,7]])
print("Multiplication (p1*q1):\n", np.dot(p1,q1))
print("Covariance:\n", np.cov(p1.flatten(), q1.flatten()))

p2 = np.array([[1,2],[2,3],[4,5]])
q2 = np.array([[4,5,1],[6,7,2]])
print("Multiplication (p2*q2):\n", np.dot(p2,q2))
print("Covariance:\n", np.cov(p2.flatten(), q2.flatten()))

# Q1(h) inner, outer, cartesian product
x = np.array([[2,3,4],[3,2,9]])
y = np.array([[1,5,0],[5,10,3]])
print("Inner product:\n", np.inner(x,y))
print("Outer product:\n", np.outer(x,y))

# Cartesian product (pairs of flattened elements)
cartesian = np.array(np.meshgrid(x.flatten(), y.flatten())).T.reshape(-1, 2)
print("Cartesian product (pairs):\n", cartesian[:10], "...")  # show only first 10

# Q2(a) math & stats
array = np.array([[1, -2, 3], [-4, 5, -6]])
print("Absolute values:\n", np.abs(array))
print("Flattened percentiles:")
print("25th:", np.percentile(array, 25))
print("50th:", np.percentile(array, 50))
print("75th:", np.percentile(array, 75))
print("Column-wise percentiles 25/50/75:", 
      np.percentile(array, [25,50,75], axis=0))
print("Row-wise percentiles 25/50/75:", 
      np.percentile(array, [25,50,75], axis=1))

print("Flattened mean/median/std:", np.mean(array), np.median(array), np.std(array))
print("Column-wise mean/median/std:", np.mean(array,axis=0), np.median(array,axis=0), np.std(array,axis=0))
print("Row-wise mean/median/std:", np.mean(array,axis=1), np.median(array,axis=1), np.std(array,axis=1))

# Q2(b) floor/ceil/trunc/round
a = np.array([-1.8, -1.6, -0.5, 0.5, 1.6, 1.8, 3.0])
print("Floor:", np.floor(a))
print("Ceil:", np.ceil(a))
print("Truncated:", np.trunc(a))
print("Rounded:", np.round(a))

# Q3(a) sorting
array3 = np.array([10,52,62,16,16,54,453])
print("Sorted array:", np.sort(array3))
print("Indices of sorted array:", np.argsort(array3))
print("4 smallest elements:", np.sort(array3)[:4])
print("5 largest elements:", np.sort(array3)[-5:])

# Q3(b) integer vs float elements
array2 = np.array([1.0, 1.2, 2.2, 2.0, 3.0, 2.0])
# Use isclose for float==int test
ints_mask = np.isclose(array2, np.round(array2))
integers = array2[ints_mask]
floats = array2[~ints_mask]
print("Integer elements only:", integers)
print("Float elements only:", floats)

# ============== Q4: image functions ===========================
print("\nCurrent Working Directory:", os.getcwd())

def img_to_array(path, save_path="image_array.txt"):
    """
    Read image from path (grayscale or RGB), save numeric array to save_path.
    Returns numpy array of image.
    """
    if not os.path.exists(path):
        print(f"Image file not found at: {path}")
        return None

    try:
        img = Image.open(path)
    except Exception as e:
        print("Error opening image:", e)
        return None

    arr = np.array(img)
    # Convert to integers just in case
    arr = arr.astype(np.int32)

    if arr.ndim == 2:
        # grayscale
        print("Grayscale Image Detected. Shape:", arr.shape)
        np.savetxt(save_path, arr, fmt='%d')
    elif arr.ndim == 3 and arr.shape[2] in (3,4):  # RGB or RGBA
        # if RGBA, drop alpha
        if arr.shape[2] == 4:
            arr = arr[:,:,:3]
        print("RGB Image Detected. Shape:", arr.shape)
        # Save as rows of RGB triplets
        h, w, c = arr.shape
        np.savetxt(save_path, arr.reshape(-1, c), fmt='%d')
    else:
        print("Unsupported image format or shape:", arr.shape)
        return None

    print(f"Image saved as array in: {save_path}")
    return arr

def load_array(file_path):
    """
    Load saved array from text file (works for both grayscale and RGB saved as rows).
    """
    if not os.path.exists(file_path):
        print("Array file not found:", file_path)
        return None
    data = np.loadtxt(file_path, dtype=int)
    print("Loaded array shape:", data.shape)
    return data

# If sample.png missing, create a sample image to test
sample_path = "sample.png"
if not os.path.exists(sample_path):
    print("sample.png not found — creating a test image 'sample.png' for you.")
    test_img = Image.fromarray(np.random.randint(0,256,(64,64,3), dtype=np.uint8))
    test_img.save(sample_path)

arr = img_to_array(sample_path, "image_array.txt")
if arr is not None:
    loaded_arr = load_array("image_array.txt")
    # show small preview
    print("Preview of loaded array (first 10 rows):\n", loaded_arr[:10])
