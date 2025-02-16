# YOLO-V8 GRADCAM on YOLOStreak Dataset

We have adapted the Grad-CAM implementation originally designed for **YOLOv5** to work with **YOLOv8** and our fine-tuned model on the **YOLOStreak dataset**. This allows us to visualize which regions the object detection model focuses on when identifying streaks in astronomical images.

A huge thank you to the original authors of the [yolov5-gradcam](https://github.com/pooya-mohammadi/yolov5-gradcam) repository for their great work! 🚀  

## **Installation**  
```bash
pip install -r requirements.txt
```

## **Inference**  
```bash
python main.py --model-path yolov8_finetuned.pt --img-path images/streak.jpg --output-dir outputs
```

**Note:** If you don't have pretrained weights, YOLOv8 will automatically download them.  

For more input arguments, check out `main.py` or run:  
```bash
python main.py -h
```

### **Custom Model Names**  
If using a custom YOLOv8 model, you may also want to specify class names:  
```bash
python main.py --model-path custom-model.pt --img-path img-path.jpg --output-dir outputs --names streak,cosmic_ray,debris
```

## **Examples**  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-repo/yolov8-gradcam/blob/main.ipynb)

### **Visualization**  
<img src="outputs/streak-res.jpg" alt="streak detection heatmap" height="300" width="1200">  

## **Future Work**  
- Extend support for **Grad-CAM++**  
- Implement **Score-CAM**  
- Integrate into deep_utils  

## **References**  
1. [yolov5-gradcam](https://github.com/pooya-mohammadi/yolov5-gradcam)  
2. [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)  

If this work helps your research, please consider citing the original **yolov5-gradcam** repository! 🚀

