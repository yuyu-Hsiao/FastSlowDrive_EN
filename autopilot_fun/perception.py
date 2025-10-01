# 影像處理
import cv2
import numpy as np
from PIL import Image

# Torch 與 transform
import torch
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

# UFLD
from UFLD.my_demo_cv import pred2coords
from UFLD.utils.common import get_model
from UFLD.utils.dist_utils import dist_print

# 其他專案內部
from .utils import merge_config 


from loguru import logger


def initialize_model_and_transforms(config_path=None, model_path=None):
    """
    初始化模型與圖片預處理變換。
    Returns:
        net: 已加載權重的神經網絡模型
        img_transforms: 圖像預處理變換
        cfg: 配置文件
        crop_size: 剪裁大小（若適用）
        img_w, img_h: 圖像寬高（取決於數據集）
    """
    # 啟用 cudnn 的最佳化
    torch.backends.cudnn.benchmark = True

    # 使用傳入的配置文件路徑，如果沒有提供則使用默認值
    if config_path is None:
        config_path = 'UFLD/configs/culane_res18.py'
    if model_path is None:
        model_path = 'UFLD/weights/culane_res18.pth'

    # 直接使用配置文件創建 cfg
    from UFLD.utils.config import Config
    

    cfg = Config.fromfile(config_path)

    # 只設置必要的參數，讓其他參數使用配置文件中的值
    cfg.batch_size = 1  # 強制設定 batch_size = 1
    cfg.test_model = model_path

    # 確保這些參數存在，但不強制覆蓋配置文件中的值
    if not hasattr(cfg, 'dataset'):
        cfg.dataset = 'CULane'
    if not hasattr(cfg, 'backbone'):
        cfg.backbone = '18'

    # 設置數據集相關錨點參數
    if cfg.dataset == 'CULane':
        cfg.row_anchor = np.linspace(0.42, 1, cfg.num_row)
        cfg.col_anchor = np.linspace(0, 1, cfg.num_col)
    elif cfg.dataset == 'Tusimple':
        cfg.row_anchor = np.linspace(160, 710, cfg.num_row) / 720
        cfg.col_anchor = np.linspace(0, 1, cfg.num_col)

    dist_print('start testing...')
    '''
    # 合併配置，設定模型和資料
    args, cfg = merge_config()
    cfg.batch_size = 1  # 強制設定 batch_size = 1
    dist_print('start testing...')
    '''


    assert cfg.backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide']

    # 設定每條車道類別數量
    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    # 加載模型
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

    # 定義圖片預處理
    img_transforms = transforms.Compose([
        transforms.Resize((533, 1600), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # 設定數據集相關參數
    if cfg.dataset == 'CULane':
        crop_size = cfg.train_height
        img_w, img_h = 1640, 590
    elif cfg.dataset == 'Tusimple':
        crop_size = None
        img_w, img_h = 1280, 720
    else:
        raise NotImplementedError

    return net, img_transforms, cfg, crop_size, img_w, img_h


def preprocess_frame(frame, img_transforms, crop_size):
    """
    預處理影像數據。
    Args:
        frame: 擷取的螢幕畫面 (NumPy array)
        img_transforms: 預處理變換
        crop_size: 剪裁大小（若適用）

    Returns:
        imgs: 預處理後的影像 (Tensor, GPU)
    """
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # 轉換為 PIL Image
    img = img_transforms(img)  # 應用預處理變換
    if crop_size:
        img = img[:, -crop_size:, :]
    img = img.unsqueeze(0)  # 添加 batch 維度
    imgs = img.cuda()  # 移動到 GPU
    return imgs


def UFLD_process_frame(frame, net, imgs, cfg, monitor):
    """
    對單幀畫面進行車道線偵測，返回車道線座標（不畫圖）。

    Returns:
        coords: 車道線座標列表 [[(x1, y1), (x2, y2), ...], ...]
    """
    with torch.no_grad():
        pred = net(imgs)

    coords = pred2coords(
        pred, 
        cfg.row_anchor, 
        cfg.col_anchor, 
        original_image_width=monitor['width'], 
        original_image_height=monitor['height']
    )

    return coords



def ByteTrack_process_frame(
    frame,
    yolo_model,   # 替換成YOLO模型
    timer,
    tracker=None, # 不再需要單獨的tracker
    trajectory=None,
    args=None,
    exp=None,
    frame_id=None
):
    """
    使用YOLOv8進行檢測和追蹤，返回偵測結果（不畫圖）。

    Returns:
        online_tlwhs: 物件的bounding box列表 [(x, y, w, h), ...]
        online_ids: 物件的ID列表 [id1, id2, ...]
        online_classes: 物件的類別列表 ["Vehicle", "Person", ...]
        have_detection: 是否檢測到車輛
    """
    # 類別映射表（YOLO class_id → 4大類字串）
    yolo_to_simple = {
        # Vehicle
        2: "Vehicle", 5: "Vehicle", 7: "Vehicle", 6: "Vehicle",  # car/bus/truck/train
        # Cyclist
        1: "Vehicle", 3: "Vehicle",                              # bicycle/motorcycle
        # Person
        0: "Person",
        # Traffic sign / light
        9: "Sign", 11: "Sign",                                   # traffic light / stop sign
    }

    timer.tic()  # 開始計時
    
    # 使用YOLO進行追蹤
    '''
    results = yolo_model.track(
        source=frame, 
        persist=True,              # 保持追蹤狀態
        tracker="bytetrack.yaml",  # 使用ByteTrack追蹤器
        verbose=False              # 關閉冗長輸出
    )
    '''
    if isinstance(frame, np.ndarray) and frame.dtype == np.uint8 and frame.shape[2] == 3:
        # 使用YOLOv8進行追蹤，降低閾值以增加檢測機會
        results = yolo_model.track(
            source=frame, 
            persist=True,
            #conf=0.25,  # 降低信心閾值
            #iou=0.45,   # 調整NMS閾值
            tracker="bytetrack.yaml",
            verbose=False
        )
    else:
        logger.error(f"框架格式不正確: shape={frame.shape}, dtype={frame.dtype}")
        return [], [], [], False

    timer.toc()  # 結束計時
    
    online_tlwhs, online_ids, online_classes = [], [], []
    
    if results and len(results) > 0 and results[0].boxes is not None and hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
        # 取得檢測框
        boxes = results[0].boxes.xyxy.cpu().numpy()  # 獲取xyxy格式的框
        # 取得追蹤ID
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        # 取得類別
        cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        
        for box, track_id, cls_id in zip(boxes, ids, cls_ids):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            
            # 轉換成tlwh格式
            tlwh = (x1, y1, w, h)
            
            # 使用類別映射
            simple_class = yolo_to_simple.get(cls_id, "Other")
            
            # 篩選條件（可依實際需求調整）
            #if w * h > args.min_box_area and w / h < args.aspect_ratio_thresh:
            if (w * h > args.min_box_area and w / h < args.aspect_ratio_thresh and simple_class not in ["Other", "Sign"]): 
                online_tlwhs.append(tlwh)
                online_ids.append(track_id)
                online_classes.append(simple_class)
                
                # 更新軌跡
                if trajectory:
                    trajectory.trajectory(tid=track_id, frame_id=frame_id, tlwh=tlwh)
    
    # 判斷是否有車輛檢測
    have_detection = "Vehicle" in online_classes
    
    # 移除非活躍軌跡
    if trajectory:
        trajectory.remove_inactive_trajectories(current_frame_id=frame_id)
    
    return online_tlwhs, online_ids, online_classes, have_detection



