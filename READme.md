# 🧠 Brain Tumor Detection with Custom CNN (AnyNet)

This project explores **deep learning for brain tumor detection** using MRI scans.  
I built a family of lightweight CNNs (AnyNet1–4) and evaluated them on a Kaggle dataset.  
The final model (`AnyNet256`) achieved **88% accuracy**, **97% recall on tumor cases**, and **PR AUC of 0.92**, with interpretability provided via **Grad-CAM** heatmaps.

---

## 📂 Dataset
- Source: [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)  
- Classes: **yes (tumor)**, **no (healthy)**  
- Balanced across ~3k images.  
- Not included in this repo due to Kaggle license.  

### Download instructions
1. Get your `kaggle.json` from [Kaggle account settings](https://www.kaggle.com/account).
2. Place it in your runtime (`~/.kaggle/`).
3. Run:
   ```bash
   kaggle datasets download -d navoneel/brain-mri-images-for-brain-tumor-detection -p ./data
   unzip ./data/brain-mri-images-for-brain-tumor-detection.zip -d ./data/brain_tumor_dataset
