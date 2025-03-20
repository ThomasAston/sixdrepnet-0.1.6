import time
import os
import argparse
import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from face_detection import RetinaFace
import utils
from model import SixDRepNet

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation on a video using 6DRepNet.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device id to use, set -1 to use CPU')
    parser.add_argument('--video', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--snapshot', type=str, required=True, help='Path to the model snapshot')
    parser.add_argument('--output', type=str, default='', help='Path to save the output video (optional)')
    return parser.parse_args()

transformations = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda:%d' % args.gpu if args.gpu >= 0 else 'cpu')
    
    print('Loading model...')
    model = SixDRepNet(backbone_name='RepVGG-B1g2', backbone_file='', deploy=True, pretrained=False)
    saved_state_dict = torch.load(args.snapshot, map_location='cpu')
    model.load_state_dict(saved_state_dict.get('model_state_dict', saved_state_dict))
    model.to(device)
    model.eval()
    
    detector = RetinaFace(gpu_id=args.gpu)
    cap = cv2.VideoCapture(args.video)
    
    if not cap.isOpened():
        raise IOError("Cannot open video file")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    else:
        out = None
    
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            faces = detector(frame)
            for box, landmarks, score in faces:
                if score < 0.95:
                    continue
                
                x_min, y_min, x_max, y_max = map(int, box)
                bbox_width, bbox_height = x_max - x_min, y_max - y_min
                x_min, y_min = max(0, x_min - int(0.2 * bbox_width)), max(0, y_min - int(0.2 * bbox_height))
                x_max, y_max = x_max + int(0.2 * bbox_width), y_max + int(0.2 * bbox_height)
                
                img = frame[y_min:y_max, x_min:x_max]
                img = Image.fromarray(img).convert('RGB')
                img = transformations(img).unsqueeze(0).to(device)
                
                start = time.time()
                R_pred = model(img)
                end = time.time()
                print(f'Head pose estimation: {(end - start) * 1000:.2f} ms')
                
                euler = utils.compute_euler_angles_from_rotation_matrices(R_pred) * 180 / np.pi
                p_pred_deg, y_pred_deg, r_pred_deg = euler[:, 0].cpu(), euler[:, 1].cpu(), euler[:, 2].cpu()
                
                utils.plot_pose_cube(frame, y_pred_deg, p_pred_deg, r_pred_deg, 
                                     x_min + bbox_width // 2, y_min + bbox_height // 2, size=bbox_width)
                
                with open('predictions.txt', 'a') as f:
                    f.write(f'Roll: {p_pred_deg.item():.2f}, Pitch: {y_pred_deg.item():.2f}, Yaw: {r_pred_deg.item():.2f}\n')

            if out:
                out.write(frame)
            
            cv2.imshow("Demo", frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
