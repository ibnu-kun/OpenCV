import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess

# -- VS CODE COMPATIBLE CODE (BATCH PROCESSING + SAVE) --

# Clone repository if not exists
repo_url = "https://github.com/ibnu-kun/OpenCV.git"
repo_path = "OpenCV_local"
if not os.path.exists(repo_path):
    subprocess.run(["git", "clone", repo_url, repo_path])

# Menjadi seperti ini:
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
converse_dir = os.path.join(repo_path, "converse")
save_dir = os.path.join(BASE_DIR, "hasil_tugas1")
os.makedirs(save_dir, exist_ok=True)

if os.path.exists(converse_dir):
    files = [f for f in os.listdir(converse_dir) if f.lower().endswith('.jpg')]
    
    for filename in files:
        img_path = os.path.join(converse_dir, filename)
        # Read as Color (BGR)
        img_bgr = cv2.imread(img_path)
        
        if img_bgr is None:
            print(f"Could not read {filename}")
            continue
            
        # Convert to Grayscale for Edge Detection filters
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        
        # Processing filters (using grayscale for edge detection)
        gauss_processed = cv2.GaussianBlur(img_gray, (3,3), 0)
        sobel_res = cv2.Sobel(gauss_processed, cv2.CV_64F, 1, 1, ksize=3)
        laplace_res = cv2.Laplacian(gauss_processed, cv2.CV_64F)
        canny_res = cv2.Canny(gauss_processed, 100, 200)
        
        # Noise for testing (Applied to color image)
        test_noise = np.random.normal(0, 25, img_bgr.shape).astype(np.uint8)
        noisy_res_bgr = cv2.add(img_bgr, test_noise)

        # Prepare for Saving (Using BGR for imwrite)
        base_name = os.path.splitext(filename)[0]
        outputs = {
            "original": img_bgr,
            "gaussian": gauss_processed,
            "sobel": cv2.normalize(sobel_res, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            "laplacian": cv2.normalize(laplace_res, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            "canny": canny_res,
            "noise": noisy_res_bgr
        }

        for suffix, image_data in outputs.items():
            save_path = os.path.join(save_dir, f"{base_name}_{suffix}.jpg")
            cv2.imwrite(save_path, image_data)

        # Visualization (Convert BGR to RGB for matplotlib)
        titles = ["Original", "Gaussian (Gray)", "Sobel", "Laplacian", "Canny", "Noisy (Color)"]
        # Map images to RGB or Gray depending on content
        img_list = [
            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), 
            gauss_processed, 
            outputs["sobel"], 
            outputs["laplacian"], 
            canny_res, 
            cv2.cvtColor(noisy_res_bgr, cv2.COLOR_BGR2RGB)
        ]
        
        plt.figure(figsize=(12, 8))
        for i in range(6):
            plt.subplot(2, 3, i+1)
            # Tampilkan gambar RGB atau Grayscale sesuai channelnya
            if len(img_list[i].shape) == 3:
                plt.imshow(img_list[i])
            else:
                plt.imshow(img_list[i], cmap='gray')
            plt.title(titles[i])
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{base_name}_visualisasi.png"))
        plt.close()
    
    print(f"All processed images (Original in Color) saved in: {save_dir}")
else:
    print(f"Directory {converse_dir} not found.")