import torch
import cv2
import numpy as np
from PIL import Image
from utils.dist_utils import dist_print
from utils.common import merge_config, get_model
import torchvision.transforms as transforms
import time

def pred2coords(pred, row_anchor, col_anchor, local_width = 1, original_image_width = 1640, original_image_height = 590):
    batch_size, num_grid_row, num_cls_row, num_lane_row = pred['loc_row'].shape
    batch_size, num_grid_col, num_cls_col, num_lane_col = pred['loc_col'].shape

    max_indices_row = pred['loc_row'].argmax(1).cpu()
    # n , num_cls, num_lanes
    valid_row = pred['exist_row'].argmax(1).cpu()
    # n, num_cls, num_lanes

    max_indices_col = pred['loc_col'].argmax(1).cpu()
    # n , num_cls, num_lanes
    valid_col = pred['exist_col'].argmax(1).cpu()
    # n, num_cls, num_lanes

    pred['loc_row'] = pred['loc_row'].cpu()
    pred['loc_col'] = pred['loc_col'].cpu()

    coords = []

    row_lane_idx = [1,2]
    col_lane_idx = [0,3]

    for i in row_lane_idx:
        tmp = []
        if valid_row[0,:,i].sum() > num_cls_row / 2:
            for k in range(valid_row.shape[1]):
                if valid_row[0,k,i]:
                    all_ind = torch.tensor(list(range(max(0,max_indices_row[0,k,i] - local_width), min(num_grid_row-1, max_indices_row[0,k,i] + local_width) + 1)))
                    
                    out_tmp = (pred['loc_row'][0,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5
                    out_tmp = out_tmp / (num_grid_row-1) * original_image_width
                    tmp.append((int(out_tmp), int(row_anchor[k] * original_image_height)))
            coords.append(tmp)

    for i in col_lane_idx:
        tmp = []
        if valid_col[0,:,i].sum() > num_cls_col / 4:
            for k in range(valid_col.shape[1]):
                if valid_col[0,k,i]:
                    all_ind = torch.tensor(list(range(max(0,max_indices_col[0,k,i] - local_width), min(num_grid_col-1, max_indices_col[0,k,i] + local_width) + 1)))
                    
                    out_tmp = (pred['loc_col'][0,all_ind,k,i].softmax(0) * all_ind.float()).sum() + 0.5

                    out_tmp = out_tmp / (num_grid_col-1) * original_image_height
                    tmp.append((int(col_anchor[k] * original_image_width), int(out_tmp)))
            coords.append(tmp)

    return coords

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    # 合併配置，設定模型和資料
    args, cfg = merge_config()
    cfg.batch_size = 1
    print('setting batch_size to 1 for demo generation')
    dist_print('start testing...')

    # 加載車道線檢測模型
    net = get_model(cfg)
    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v
    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()
    net.cuda()

    # 定義圖像預處理
    img_transforms = transforms.Compose([
        transforms.Resize((cfg.train_height, cfg.train_width)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # 打開影片文件
    video_path = "../videos/test0_normal_2.avi"  # 替換為您的影片路徑
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 當沒有更多幀時，結束播放

        # 獲取原始影像尺寸
        img_h, img_w = frame.shape[:2]

        # 預處理圖像
        input_width = cfg.train_width
        input_height = cfg.train_height
        img_resized = cv2.resize(frame, (input_width, input_height))
        img_pil = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
        img_tensor = img_transforms(img_pil)
        img_tensor = img_tensor.unsqueeze(0).cuda()

        # 模型推論，禁用梯度加速推論
        with torch.no_grad():
            pred = net(img_tensor)

        # 獲取車道線座標
        coords = pred2coords(pred, cfg.row_anchor, cfg.col_anchor,
                             original_image_width=img_w, original_image_height=img_h)

        # 在原始圖像上繪製車道線
        for lane in coords:
            for coord in lane:
                cv2.circle(frame, coord, 5, (0, 255, 0), -1)

        # 顯示處理後的圖像
        cv2.imshow('Lane Detection', frame)

        # 控制退出和播放速度
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q'):  # 按 'q' 鍵退出
            break
        elif key == ord('p'):  # 按 'p' 鍵暫停
            while True:
                key2 = cv2.waitKey(0) & 0xFF
                if key2 == ord('p'):  # 再次按 'p' 繼續
                    break
                elif key2 == ord('q'):  # 按 'q' 鍵退出
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

    cap.release()
    cv2.destroyAllWindows()