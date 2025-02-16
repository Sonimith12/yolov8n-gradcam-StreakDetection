import time
import torch
import torch.nn.functional as F
import numpy as np


def find_yolo_layer(model, layer_name=None):
    """Find YOLO layer for GradCAM and GradCAM++.

    Args:
        model: YOLOv5 or YOLOv8 model.
        layer_name (str, optional): Name of the target layer with its hierarchical structure.

    Returns:
        target_layer: The found layer.

    Raises:
        ValueError: If the layer cannot be found.
    """
    # print("[DEBUG] Available layers in YOLOv8 model:")
    # for name, module in model.model.named_modules():
    #     print(name, "->", module)

    if layer_name is None:
        print("[WARNING] No layer specified. Attempting to find the last Conv2d layer...")
        # Automatically find the last convolutional layer
        for name, module in reversed(model.model._modules.items()):
            if isinstance(module, torch.nn.Conv2d):
                print(f"[INFO] Automatically selected layer: {name}")
                return module
        raise ValueError("No Conv2d layer found in the model!")

    hierarchy = layer_name.split('.')
    target_layer = model.model._modules.get(hierarchy[0])

    if target_layer is None:
        print(f"[ERROR] Layer '{hierarchy[0]}' not found in model. Available layers:")
        print(list(model.model._modules.keys()))
        raise ValueError(f"Layer '{hierarchy[0]}' does not exist.")

    # Traverse the hierarchy
    for h in hierarchy[1:]:
        if h not in target_layer._modules:
            print(f"[ERROR] Sub-layer '{h}' not found inside '{hierarchy[0]}'. Available sub-layers:")
            print(list(target_layer._modules.keys()))
            raise ValueError(f"Sub-layer '{h}' does not exist inside '{hierarchy[0]}'.")
        target_layer = target_layer._modules[h]

    print(f"[INFO] Successfully found layer: {layer_name}")
    return target_layer
class YOLOV5GradCAM:

    def __init__(self, model, layer_name, img_size=(640, 640)):
        self.model = model
        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            print("[DEBUG] Backward hook triggered")
            self.gradients['value'] = grad_output[0].clone()
            return None

        def forward_hook(module, input, output):
            self.activations['value'] = output.clone()
            return None

        target_layer = find_yolo_layer(self.model, layer_name)
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        device = 'cuda' if next(self.model.model.parameters()).is_cuda else 'cpu'
        self.model.model.model(torch.zeros(1, 3, *img_size, device=device))

        print('[INFO] saliency_map size :', self.activations['value'].shape[2:])

    def forward(self, input_img, class_idx=True):
        """
        Args:
            input_img: input image with shape of (1, 3, H, W)
        Returns:
            saliency_maps: List of saliency maps, one for each detected object.
            preds: The YOLOv8 model detections.
        """
        saliency_maps = []
        
        tic = time.time()

    
        if isinstance(input_img, np.ndarray):  # Ensure input is a tensor
            input_img = torch.tensor(input_img, dtype=torch.float32)

        # ✅ Ensure it’s in (B, C, H, W) format
        if input_img.shape[-1] == 3:  
            input_img = input_img.permute(0, 3, 1, 2).contiguous() 

        
        # ✅ Ensure input is in BCHW format (Batch, Channels, Height, Width)
        if input_img.shape[-1] == 3:  # Check if channels are last
            input_img = input_img.permute(0, 3, 1, 2).contiguous()  # Convert to (B, C, H, W)
        
        # ✅ Resize to be divisible by 32 (YOLO requirement)
        _, _, h, w = input_img.shape
        new_h = (h // 32) * 32
        new_w = (w // 32) * 32
        input_img = F.interpolate(input_img, size=(new_h, new_w), mode="bilinear", align_corners=False)

        # ✅ Enable gradients
        input_img.requires_grad = True

        # ✅ Ensure model is in evaluation mode
        self.model.model.eval()

        preds = self.model.model(input_img)  
        print("[INFO] Model forward took:", round(time.time() - tic, 4), "seconds")
        if len(preds) == 0 or preds[0].boxes is None:
            print("[WARNING] No detections found.")
            return [], preds

        boxes = preds[0].boxes.xyxy  # Bounding boxes (x1, y1, x2, y2)
        confs = preds[0].boxes.conf  # Confidence scores
        class_ids = preds[0].boxes.cls.int()  # Class indices
        
        for box, conf, cls in zip(boxes, confs, class_ids):
            score = conf.clone().detach().requires_grad_(True)

            self.model.model.zero_grad()

            tic = time.time()
            score.backward(retain_graph=True)  # Compute gradients for Grad-CAM
            print(f"[INFO] Class {cls}, model-backward took:", round(time.time() - tic, 4), "seconds")

            if 'value' not in self.gradients:
                print("[ERROR] Gradients not found. Backward hook not triggered.")
                continue

            gradients = self.gradients['value']
            activations = self.activations['value']

            b, k, u, v = gradients.size()
            alpha = gradients.view(b, k, -1).mean(2)  # Global average pooling
            weights = alpha.view(b, k, 1, 1)
            saliency_map = (weights * activations).sum(1, keepdim=True)

            saliency_map = F.relu(saliency_map)

            saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)

            saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
            saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min + 1e-7)

            saliency_maps.append(saliency_map)

        return saliency_maps, preds

    def __call__(self, input_img):
        return self.forward(input_img)
