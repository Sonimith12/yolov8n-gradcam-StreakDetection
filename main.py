import os
import time
import argparse
import numpy as np
from models.gradcam import YOLOV5GradCAM
from models.yolo_v5_object_detector import YOLOV8TorchObjectDetector


import cv2
from deep_utils import Box, split_extension

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model-path', type=str, default="yolov5s.pt", help='Path to the model')
parser.add_argument('--img-path', type=str, default='images/', help='input image path')
parser.add_argument('--output-dir', type=str, default='outputs', help='output dir')
parser.add_argument('--img-size', type=int, default=640, help="input image size")
parser.add_argument('--target-layer', type=str, default='model.model.21.m.1.cv2.conv',
                    help='The layer hierarchical address to which gradcam will applied,'
                         ' the names should be separated by underline')
parser.add_argument('--method', type=str, default='gradcam', help='gradcam or gradcampp')
parser.add_argument('--device', type=str, default='cpu', help='cuda or cpu')
parser.add_argument('--names', type=str, default=None,
                    help='The name of the classes. The default is set to None and is set to coco classes. Provide your custom names as follow: object1,object2,object3')

args = parser.parse_args()


def get_res_img(bbox, mask, res_img):
    mask = mask.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(
        np.uint8)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    n_heatmat = (Box.fill_outer_box(heatmap, bbox) / 255).astype(np.float32)
    res_img = res_img / 255
    res_img = cv2.add(res_img, n_heatmat)
    res_img = (res_img / res_img.max())
    return res_img, n_heatmat


def put_text_box(bbox, cls_name, res_img):
    x1, y1, x2, y2 = bbox
    # this is a bug in cv2. It does not put box on a converted image from torch unless it's buffered and read again!
    cv2.imwrite('temp.jpg', (res_img * 255).astype(np.uint8))
    res_img = cv2.imread('temp.jpg')
    res_img = Box.put_box(res_img, bbox)
    res_img = Box.put_text(res_img, cls_name, (x1, y1))
    return res_img


def concat_images(images):
    h, w = images[0].shape[:2]  # Get correct height and width
    print(h, w)
    total_height = h * len(images)  # Compute total height for stacking

    # Ensure correct dimensions (Height, Width, Channels)
    base_img = np.zeros((total_height, w, 3), dtype=np.uint8)  
    

    y_offset = 0  # Keeps track of where to place the next image
    for img in images:
        if img.shape[1] == 3:  # If format is (H, C, W), fix it
            img = np.transpose(img, (0, 2, 1))  # Convert (H, C, W) → (H, W, C)
        
        base_img[y_offset:y_offset + h, :, :] = img
        y_offset += h

    return base_img


def main(img_path):
    device = args.device
    input_size = (args.img_size, args.img_size)
    img = cv2.imread(img_path)
    print('[INFO] Loading the model')
    model = YOLOV8TorchObjectDetector(args.model_path, device, img_size=input_size,
                                      names=None if args.names is None else args.names.strip().split(","))
    
    torch_img = model.preprocessing(img[..., ::-1])
    if args.method == 'gradcam':
        saliency_method = YOLOV5GradCAM(model=model, layer_name=args.target_layer, img_size=input_size)
    tic = time.time()
    masks, logits, [boxes, _, class_names, _] = saliency_method(torch_img)
    print("total time:", round(time.time() - tic, 4))
    result = torch_img.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy()
    result = result[..., ::-1]  # convert to bgr
    images = [result]
    for i, mask in enumerate(masks):
        res_img = result.copy()
        bbox, cls_name = boxes[0][i], class_names[0][i]
        res_img, heat_map = get_res_img(bbox, mask, res_img)
        res_img = put_text_box(bbox, cls_name, res_img)
        images.append(res_img)
    final_image = concat_images(images)
    img_name = split_extension(os.path.split(img_path)[-1], suffix='-res')
    output_path = f'{args.output_dir}/{img_name}'
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'[INFO] Saving the final image at {output_path}')
    cv2.imwrite(output_path, final_image)

def folder_main(folder_path):
    device = args.device
    input_size = (args.img_size, args.img_size)
    print('[INFO] Loading the model')
    model = YOLOV8TorchObjectDetector(args.model_path, args.device, img_size=(args.img_size, args.img_size))
    print("model.name: ", model.model.names)
    os.makedirs(args.output_dir, exist_ok=True)

    for item in os.listdir(folder_path):
        img_path = os.path.join(folder_path, item)
        if not os.path.isfile(img_path):  # Skip directories
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARNING] Unable to read image: {img_path}")
            continue

        torch_img = model.preprocessing(img[..., ::-1])  # Convert BGR to RGB

        if args.method == 'gradcam':
            saliency_method = YOLOV5GradCAM(model=model, layer_name=args.target_layer, img_size=input_size)

        tic = time.time()
        saliency_maps, preds = saliency_method(torch_img)


        boxes = preds[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
        class_ids = preds[0].boxes.cls.int().cpu().numpy()  # Class indices
        class_names = [model.model.names[cls_id] for cls_id in class_ids]  # Class names

        print("Total time:", round(time.time() - tic, 4))

        import torch
        if isinstance(torch_img, np.ndarray):
            torch_img = torch.tensor(torch_img, dtype=torch.float32)  # Convert NumPy array to PyTorch tensor

        result = torch_img.squeeze(0)  # Remove batch dimension
        result = result.mul(255).add_(0.5).clamp_(0, 255)  # Normalize and scale
        result = result.permute(1, 2, 0).detach().cpu().numpy()  # Convert to (H, W, C) format

        result = torch_img.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy()
        result = result[..., ::-1]

        images = [result]
        for i, saliency_map in enumerate(saliency_maps):
            res_img = result.copy()
            bbox = boxes[i]  # Bounding box for the current detection
            cls_name = class_names[i]  # Class name for the current detection

            # Overlay saliency map and bounding box
            res_img, _ = get_res_img(bbox, saliency_map, res_img)
            res_img = put_text_box(bbox, cls_name, res_img)
            images.append(res_img)

        # Concatenate images for final output
        final_image = concat_images(images)

        # Save the final image
        img_name = os.path.splitext(os.path.basename(img_path))[0] + '-res.jpg'
        output_path = os.path.join(args.output_dir, img_name)
        print(f'[INFO] Saving the final image at {output_path}')
        cv2.imwrite(output_path, final_image)


if __name__ == '__main__':
    if os.path.isdir(args.img_path):
        folder_main(args.img_path)
    else:
        main(args.img_path)
