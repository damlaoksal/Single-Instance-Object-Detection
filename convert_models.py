import os
import sys
import yaml
import torch
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODELS_ONNX_DIR = os.path.join(BASE_DIR, "models_onnx")
OSTRACK_DIR = os.path.join(BASE_DIR, "OSTrack")
sys.path.append(OSTRACK_DIR)

try:
    from lib.models.ostrack import build_ostrack
    from lib.config.ostrack.config import cfg
except ImportError:
    pass

def update_easydict(target_dict, new_dict):
    for k, v in new_dict.items():
        if isinstance(v, dict) and k in target_dict:
            update_easydict(target_dict[k], v)
        else:
            target_dict[k] = v

def find_checkpoint(base_path):
    if os.path.isfile(base_path):
        return base_path
    paths = [os.path.join(r, f) for r, _, files in os.walk(base_path) for f in files if f.endswith(('.pth', '.pt', '.tar', '.pyth'))]
    return max(paths, key=os.path.getsize) if paths else None

def export_yolo():
    print("\n[1/3] Exporting YOLO11s to ONNX...")
    os.makedirs(MODELS_ONNX_DIR, exist_ok=True)
    
    yolo_weight = os.path.join(MODELS_DIR, "yolo11s.pt")
    if not os.path.exists(yolo_weight):
        yolo_weight = "yolo11s.pt"
        
    YOLO(yolo_weight).export(format="onnx", imgsz=640, dynamic=False, simplify=True, opset=17)
    
    generated_onnx = "yolo11s.onnx"
    if not os.path.exists(generated_onnx):
        generated_onnx = os.path.join(MODELS_DIR, "yolo11s.onnx")

    if os.path.exists(generated_onnx):
        target_path = os.path.join(MODELS_ONNX_DIR, "yolo11s.onnx")
        if os.path.exists(target_path):
            os.remove(target_path)
        os.replace(generated_onnx, target_path)
    print("[SUCCESS] YOLO11s exported.")

def export_osnet():
    print("\n[2/3] Exporting OSNet to ONNX...")
    import torchreid
    model = torchreid.models.build_model(name='osnet_ain_x0_5', num_classes=1000, pretrained=False)
    
    weight_path = os.path.join(MODELS_DIR, "osnet_ain_x0_5_imagenet.pyth")
    if not os.path.exists(weight_path):
        weight_path = "osnet_ain_x0_5_imagenet.pyth"
        
    ckpt = torch.load(weight_path, map_location='cpu')
    model.load_state_dict(ckpt.get('state_dict', ckpt))
    model.eval()

    os.makedirs(MODELS_ONNX_DIR, exist_ok=True)
    torch.onnx.export(
        model, torch.randn(1, 3, 256, 128), os.path.join(MODELS_ONNX_DIR, "osnet_ain_x0_5.onnx"),
        export_params=True, opset_version=17, do_constant_folding=True,
        input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("[SUCCESS] OSNet exported.")

def export_ostrack():
    print("\n[3/3] Exporting OSTrack to ONNX...")
    exp_dir = os.path.join(OSTRACK_DIR, "experiments", "ostrack")
    yaml_file = next((os.path.join(exp_dir, f) for f in os.listdir(exp_dir) if f.endswith('.yaml') and '256' in f), None)
    
    if yaml_file:
        with open(yaml_file, 'r', encoding='utf-8') as f:
            update_easydict(cfg, yaml.safe_load(f))
    
    model = build_ostrack(cfg, training=False)
    
    target_path = os.path.join(MODELS_DIR, "OSTrack_ep0300.pth")
    ckpt_file = find_checkpoint(target_path) or find_checkpoint(MODELS_DIR) or find_checkpoint(BASE_DIR)
    
    ckpt = torch.load(ckpt_file, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt.get('net', ckpt), strict=False)
    model.eval()

    os.makedirs(MODELS_ONNX_DIR, exist_ok=True)
    torch.onnx.export(
        model, (torch.randn(1, 3, 128, 128), torch.randn(1, 3, 256, 256)),
        os.path.join(MODELS_ONNX_DIR, "ostrack_256.onnx"),
        export_params=True, opset_version=16, do_constant_folding=True,
        input_names=['template', 'search'], output_names=['output']
    )
    print("[SUCCESS] OSTrack exported.")

if __name__ == "__main__":
    export_yolo()
    export_osnet()
    export_ostrack()
    print("\nAll models exported successfully to 'models_onnx/' directory.")