import os
import sys
# 設定當前目錄並加入 UFLD 模組的路徑
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'UFLD'))

from autopilot_fun.utils import (
    make_parser, 
    merge_config, 
    get_window_geometry_by_id, 
    get_window_geometry_by_name,
    danger_zone_from_lane         
)
from autopilot_fun.visualization import draw_trajectory, draw_results, draw_danger_zone
from autopilot_fun.perception import (
    initialize_model_and_transforms,
    preprocess_frame,
    ByteTrack_process_frame,
    UFLD_process_frame
    
    
)

from autopilot_fun.control import (
    initialize_carla,
    control_vehicle,
    PIDController,
    LaneTracker,
    calculate_lane_center,
    calculate_deviation,
    lane_keeping_control,
    obstacle_avoidance_control,
    obstacle_avoidance_control_improved
)

from autopilot_fun.integration import (
    OpenAIIntegration,
    extract_emergency_code,
    generate_dynamic_emergency_prompt
)


# 導入Social-LSTM分類器
from TrajectoryClassification.social_lstm_trainer import SocialLSTMClassifier

import os.path as osp
import time
import cv2
import torch
import mss
import numpy as np
import tqdm
import os, argparse
from PIL import Image
import threading 

import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from loguru import logger



# 引入 ByteTrack 相關模組
from ultralytics import YOLO
from autopilot_fun.utils import SimpleTimer

import pyautogui


class CARLAWorldController:
    """CARLA世界控制器，用於暫停/恢復世界"""
    def __init__(self):
        self.client = None
        self.world = None
        self.original_settings = None
        self.is_paused = False
        
    def initialize(self, client=None, world=None):
        """初始化，可以傳入現有的client和world"""
        try:
            if world is not None:
                # 直接使用傳入的 world
                self.world = world
                # 嘗試獲取 client（如果需要的話）
                logger.info("使用傳入的 world 物件")
            else:
                # 創建新的連接
                import carla  # 確保 carla 已導入
                self.client = carla.Client('127.0.0.1', 2000)
                self.client.set_timeout(10.0)
                self.world = self.client.get_world()
                logger.info("創建新的 CARLA 連接")
            
            # 保存當前設定
            if self.world is not None:
                self.original_settings = self.world.get_settings()
                logger.info(f"已保存原始設定 - 同步模式: {self.original_settings.synchronous_mode}, 時間步: {self.original_settings.fixed_delta_seconds}")
                logger.info("CARLA世界控制器初始化成功")
                return True
            else:
                logger.error("world 物件仍為 None")
                return False
                
        except Exception as e:
            logger.error(f"CARLA世界控制器初始化失敗: {e}")
            import traceback
            logger.error(f"詳細錯誤: {traceback.format_exc()}")
            return False
    
    def pause_world(self):
        """暫停世界"""
        logger.info(f"嘗試暫停世界 - is_paused: {self.is_paused}, world: {self.world is not None}")
        
        if self.is_paused:
            logger.warning("世界已經暫停，跳過")
            return False
        
        if not self.world:
            logger.error("world 物件為 None，無法暫停")
            return False
        
        try:
            current_settings = self.world.get_settings()
            logger.info(f"當前設定 - 同步模式: {current_settings.synchronous_mode}, 時間步: {current_settings.fixed_delta_seconds}")
            
            settings = self.world.get_settings()
            settings.fixed_delta_seconds = 0.0001  # 極小的時間步，幾乎暫停
            self.world.apply_settings(settings)
            self.is_paused = True
            
            # 驗證設定是否生效
            new_settings = self.world.get_settings()
            logger.info(f"暫停後設定 - 同步模式: {new_settings.synchronous_mode}, 時間步: {new_settings.fixed_delta_seconds}")
            logger.warning("CARLA世界已暫停")
            return True
        except Exception as e:
            logger.error(f"暫停世界失敗: {e}")
            return False

    
    def resume_world(self):
        """恢復世界"""
        logger.info(f"嘗試恢復世界 - is_paused: {self.is_paused}, world: {self.world is not None}, original_settings: {self.original_settings is not None}")
        
        if not self.is_paused:
            logger.warning("世界未暫停，跳過恢復")
            return False
        
        if not self.world:
            logger.error("world 物件為 None，無法恢復")
            return False
        
        if not self.original_settings:
            logger.error("original_settings 為 None，無法恢復")
            return False
        
        try:
            logger.info(f"恢復到原始設定 - 同步模式: {self.original_settings.synchronous_mode}, 時間步: {self.original_settings.fixed_delta_seconds}")
            
            # 恢復到原始設定
            self.world.apply_settings(self.original_settings)
            self.is_paused = False
            
            # 驗證設定是否生效
            new_settings = self.world.get_settings()
            logger.info(f"恢復後設定 - 同步模式: {new_settings.synchronous_mode}, 時間步: {new_settings.fixed_delta_seconds}")
            logger.warning("CARLA世界已恢復")
            return True
        except Exception as e:
            logger.error(f"恢復世界失敗: {e}")
            return False



# ===== GUI通訊相關類別 =====
class GUIMessageHandler:
    """處理GUI與MOT之間的訊息通訊"""
    def __init__(self):
        self.pending_gui_messages = []  # 待處理的GUI訊息
        self.gui_response_callback = None  # GUI回應回調
        self.current_frame = None  # 當前幀
        self.frame_save_folder = None  # 幀保存目錄
        
    def set_gui_callback(self, callback):
        """設定GUI回調函數"""
        self.gui_response_callback = callback
        
    def add_gui_message(self, message):
        """添加GUI訊息到處理隊列"""
        self.pending_gui_messages.append({
            'message': message,
            'timestamp': time.time()
        })
        logger.info(f"收到GUI訊息: {message}")
        
    def update_current_frame(self, frame, save_folder):
        """更新當前幀"""
        self.current_frame = frame.copy() if frame is not None else None
        self.frame_save_folder = save_folder
        
    def has_pending_messages(self):
        """檢查是否有待處理的訊息"""
        return len(self.pending_gui_messages) > 0
        
    def get_next_message(self):
        """獲取下一個待處理的訊息"""
        if self.pending_gui_messages:
            return self.pending_gui_messages.pop(0)
        return None
        
    def save_current_frames_for_gui(self):
        """為GUI訊息保存當前幀"""
        if self.current_frame is None or self.frame_save_folder is None:
            return None
            
        try:
            # 創建GUI專用目錄
            gui_frames_folder = osp.join(self.frame_save_folder, 'gui_frames')
            os.makedirs(gui_frames_folder, exist_ok=True)
            
            # 保存當前幀（模擬3張連續幀）
            timestamp = int(time.time())
            image_paths = []
            
            for i in range(3):
                image_path = osp.join(gui_frames_folder, f'gui_frame_{timestamp}_{i}.png')
                cv2.imwrite(image_path, self.current_frame)
                image_paths.append(image_path)
                
            logger.info(f"已為GUI分析保存幀: {image_paths}")
            return image_paths
            
        except Exception as e:
            logger.error(f"保存GUI分析幀時發生錯誤: {e}")
            return None
            
    def send_response_to_gui(self, response):
        """發送回應到GUI"""
        if self.gui_response_callback:
            try:
                self.gui_response_callback(response)
            except Exception as e:
                logger.error(f"發送回應到GUI時發生錯誤: {e}")

    def send_danger_response_to_gui(self, response):
        """發送危險檢測回應到GUI"""
        if self.gui_response_callback:
            try:

                # 第一次通知：系統檢測提示
                system_alert = "系統偵測到危險"
                self.gui_response_callback(system_alert)

                # 格式化危險檢測的回應
                formatted_response = f"Danger Detection Alert \n\n{response}"
                self.gui_response_callback(formatted_response)
                logger.info("危險檢測回應已發送到GUI")
            except Exception as e:
                logger.error(f"發送危險檢測回應到GUI時發生錯誤: {e}")
        else:
            logger.warning("GUI回調函數未設置，無法發送危險檢測回應")


# ===== 主循環 =====
def screenflow_demo(vis_folder, current_time, args, frame_callback=None, stop_event=None, display_options_source=None, gui_callback=None):
    """
    執行屏幕錄製和實時物件檢測的整合流程。
    Args:
        vis_folder:     字符串，保存結果的文件夾路徑。
        current_time:   當前時間的時間戳，用於命名保存文件。
        args:           命令行參數對象，包含用於配置和選項的參數。
        frame_callback: 可選的回調函數，接收處理後的每一幀。
        stop_event:     可選的停止事件，用於從外部停止處理循環。
        display_options_source: 顯示選項來源對象。
        gui_callback:   GUI回調對象，用於設置OpenAI實例和處理GUI訊息。
    Returns:
        None
    """


    os.makedirs("logs", exist_ok=True)

    # 註冊檔案 sink：每天產生新檔，保留 10 天
    logger.add(
        "logs/autopilot_{time:YYYY-MM-DD}.log",
        rotation="00:00",                 # 每天午夜分割
        retention="10 days",              # 保留 10 天
        encoding="utf-8",
        level="INFO"                      # 記錄 INFO 以上
    )

    # ===== 定義獲取即時顯示選項的函數 =====
    def get_current_display_options():
        """獲取當前的顯示選項"""
        if display_options_source and hasattr(display_options_source, 'get_display_options'):
            return display_options_source.get_display_options()
        else:
            # 預設顯示選項
            return {
                'show_trajectory': True,
                'show_bounding_box': True,
                'show_lane_detection': True
            }


    logger.info("開始螢幕流處理...")

    # 添加調試信息
    print(f"DEBUG: vis_folder = {vis_folder}")
    print(f"DEBUG: args.window_name = {getattr(args, 'window_name', 'None')}")
    print(f"DEBUG: args.config = {getattr(args, 'config', 'None')}")
    print(f"DEBUG: args.test_model = {getattr(args, 'test_model', 'None')}")
    print(f"DEBUG: args.path = {getattr(args, 'path', 'None')}")
    
    # 檢查所有可能為 None 的屬性
    for attr_name in dir(args):
        if not attr_name.startswith('_'):
            attr_value = getattr(args, attr_name)
            if attr_value is None:
                print(f"DEBUG: args.{attr_name} = None")


    # 載入YOLO模型
    yolo_model = YOLO('yolov10s.pt')
    #yolo_model = YOLO('yolov8n.pt')  


    # ── 嘗試連線 CARLA，但不讓例外中斷後續流程 ──
    try:
        vehicle, world = initialize_carla()
        vehicle_ok = (vehicle is not None and world is not None)
        #vehicle_ok = True
    except RuntimeError as e:
    #if not vehicle_ok:
        logger.warning(f"Cannot connect to CARLA ({e}), proceed in perception-only mode.")
        vehicle, world = None, None
        vehicle_ok = False


    # 初始化CARLA世界控制器
    world_controller = CARLAWorldController()
    if world is not None:
        world_controller.initialize(world=world)
        logger.info("世界控制器已初始化")
    else:
        logger.warning("無法初始化世界控制器，CARLA未連接")


    # 初始化模型與圖像預處理參數，設置屏幕錄製工具
    net, img_transforms, cfg, crop_size, img_w, img_h = initialize_model_and_transforms(args.config, args.test_model)
    sct = mss.mss()

    '''
    try:
        x, y, width, height = get_window_geometry_by_name(args.window_name)
        if x is None or y is None or width is None or height is None:
            raise ValueError(f"無法找到窗口: {args.window_name}")
    except Exception as e:
        logger.error(f"獲取窗口幾何信息失敗: {e}")
        # 使用默認值
        x, y, width, height = 0, 0, 1920, 1080
        logger.warning("使用默認窗口大小: 1920x1080")
    '''

    # ===== 支援多種視窗捕獲方式 =====
    try:
        # 優先順序：window_id > window_name > 預設值
        if hasattr(args, 'window_id') and args.window_id:
            logger.info(f"使用 Window ID 捕獲: {args.window_id}")
            x, y, width, height = get_window_geometry_by_id(args.window_id)
            if x is None:
                raise ValueError(f"無法找到 Window ID: {args.window_id}")
        elif hasattr(args, 'window_name') and args.window_name:
            logger.info(f"使用 Window Name 捕獲: {args.window_name}")
            x, y, width, height = get_window_geometry_by_name(args.window_name)
            if x is None:
                raise ValueError(f"無法找到窗口: {args.window_name}")
        else:
            raise ValueError("未指定捕獲目標")
            
    except Exception as e:
        logger.error(f"獲取窗口幾何信息失敗: {e}")
        # 使用默認值
        x, y, width, height = 0, 0, 1920, 1080
        logger.warning("使用默認窗口大小: 1920x1080")


    monitor = {"top": y, "left": x, "width": width, "height": height}

    # 配置保存錄影和檢測結果的路徑
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)
    save_path = osp.join(save_folder, "screen_recording.mp4")
    logger.info(f"video save_path is {save_path}")

    if save_folder is None:
        save_folder = os.path.join(os.getcwd(), 'default_output')
        os.makedirs(save_folder, exist_ok=True)

    save_path = osp.join(save_folder, "screen_recording.avi")
    logger.info(f"視頻保存路徑: {save_path}")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    vid_writer = cv2.VideoWriter(
        save_path, fourcc, 30, (monitor["width"], monitor["height"])
    )

    # 創建用於保存 OpenAI API 分析結果的資料夾
    save_openai_img_folder = osp.join(save_folder, 'openai_img')
    os.makedirs(save_openai_img_folder, exist_ok=True)
    
    openai_api_key = "sk-LDFA"   # 需填入自己的 api key
    openai_integration = OpenAIIntegration(api_key=openai_api_key, output_path=osp.join(save_openai_img_folder, "openai_results.txt"))
    
    # ===== 創建GUI訊息處理器 =====
    gui_message_handler = GUIMessageHandler()
    
    # ===== 如果有GUI回調，設置OpenAI實例和訊息處理器 =====
    if gui_callback:
        try:
            # 將OpenAI實例傳遞給GUI
            gui_callback.set_openai_instance(openai_integration)
            # 將訊息處理器傳遞給GUI
            gui_callback.set_message_handler(gui_message_handler)
            logger.info("已將OpenAI實例和訊息處理器傳遞給GUI")
        except Exception as e:
            logger.error(f"設置GUI回調時發生錯誤: {e}")


    
    Draw_trajectory = draw_trajectory()     # 初始化軌跡繪製器

    # ===== 宣告車道 PID 控制器 =====
    pid_controller = PIDController(kp=0.15, ki=0.005, kd=0.08)              # 建立 PIDController 實例
    lane_tracker = LaneTracker(max_lane_shift=300, smoothing_factor=0.2)     # 建立 LaneTracker 實例

    # ===== 宣告速度 PID 控制器 =====
    speed_pid    = PIDController(kp=0.1, ki=0.005, kd=0.05)
    target_speed = 10.0  # km/h，自訂巡航速度


    # 創建追蹤器和計時器
    timer = SimpleTimer()
    frame_id = 0
    trajectories = {}

    social_lstm_weights_path = 'TrajectoryClassification/weights/2025-05-08/social-lstm.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    social_lstm_model = SocialLSTMClassifier(num_layers=2).to(device)
    social_lstm_model.load_state_dict(torch.load(social_lstm_weights_path, map_location=device))
    social_lstm_model.eval()

    # 用於軌跡存儲和風險判斷
    object_trajectories = {}  # {track_id: deque of positions}
    risk_status = {}  # {track_id: label (0-安全, 1-危險)}
    trajectory_length = 20  # 分類模型所需的軌跡長度
    from collections import deque

    # 危險檢測變數
    danger_detected = False
    danger_object_id = None


    pause_flag = False
    pause_pressed = False   # 暫停按鈕狀態

    emergency_mode = False  # 緊急模式狀態
    emergency_start_time = None  # 緊急模式開始時間
    emergency_duration = 0.0  # 緊急控制持續時間（3秒）
    extracted_code = None

    # ===== GUI訊息處理相關變數 =====
    gui_processing = False
    gui_message_start_time = None

    # 危險區域
    last_danger_zone = None

    # 手動標記危險物件
    manual_danger_objects = {}

    last_time = time.time()

    while True:

        now = time.time()
        dt  = now - last_time
        last_time = now


        # 檢查是否應該停止處理
        if stop_event and stop_event.is_set():
            logger.info("收到停止信號，結束處理...")
            break
        
        # ===== 每次循環都獲取最新的顯示選項 =====
        current_display_options = get_current_display_options()
            
        #logger.info(f"Capturing frame {frame_id}...")
        sct_img = sct.grab(monitor)
        frame = np.array(sct_img)

        if frame is None:
            logger.warning(f"Frame {frame_id} could not be captured.")
            break

        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # 保存原始影像的複本，用於 ByteTrack
        original_frame = frame.copy()

        # ===== 更新GUI訊息處理器的當前幀 =====
        gui_message_handler.update_current_frame(frame, save_folder)


        # =========== 1) ByteTrack 處理 (使用原始未處理影像) ============
        online_tlwhs, online_ids, online_classes, have_detection = [], [], [], False
        
        # 只有在需要顯示物件框或軌跡時才執行物件追蹤（因為控制邏輯可能需要）
        if current_display_options.get('show_bounding_box', True) or current_display_options.get('show_trajectory', True):
            online_tlwhs, online_ids, online_classes, have_detection = ByteTrack_process_frame(
                frame=original_frame,
                yolo_model=yolo_model,  # 使用YOLOv8模型
                timer=timer,
                trajectory=Draw_trajectory,
                args=args,
                frame_id=frame_id
            )
            #logger.debug(f"Frame {frame_id}: ByteTrack 處理完成，檢測到 {len(online_tlwhs)} 個物件")
        else:
            logger.debug(f"Frame {frame_id}: 跳過 ByteTrack 處理（物件框和軌跡都已關閉）")



        # =========== 2) UFLD 處理 (對影像進行必要的預處理) ============
        coords = []
        
        # 只有在需要顯示車道線時才執行車道線檢測（因為控制邏輯需要）
        if current_display_options.get('show_lane_detection', True):
            imgs = preprocess_frame(frame, img_transforms, crop_size)
            coords = UFLD_process_frame(frame, net, imgs, cfg, monitor)
            #logger.debug(f"Frame {frame_id}: UFLD 處理完成，檢測到 {len(coords)} 條車道線")
        else:
            logger.debug(f"Frame {frame_id}: 跳過 UFLD 處理（車道線顯示已關閉）")


        # 2) 計算 danger_zone 並可視化
        if len(coords)>=2 and coords[0] and coords[1]:
            danger_zone = danger_zone_from_lane(coords, frame.shape)
            last_danger_zone = danger_zone
        elif last_danger_zone is not None:
            # 暫時沒抓到，使用上一幀的結果
            danger_zone = last_danger_zone

        #draw_danger_zone(frame, danger_zone, alpha=0.3)
        
        timer.toc()  # 計算處理時間
        fps = 1.0 / timer.average_time if timer.average_time > 0 else 0.0

        # =========== 3) 更新軌跡存儲 ============
        for tlwh, track_id, class_name in zip(online_tlwhs, online_ids, online_classes):
            # 只考慮車輛的軌跡
            if class_name != "Vehicle":
                continue
                
            # 計算中心點座標
            x, y, w, h = tlwh
            center_x, center_y = x + w/2, y + h/2

            # 更新軌跡
            if track_id not in object_trajectories:
                object_trajectories[track_id] = deque(maxlen=trajectory_length)
            
            object_trajectories[track_id].append((center_x, center_y))
            
            # 檢查軌跡長度是否足夠進行分類
            if len(object_trajectories[track_id]) == trajectory_length:
                # 準備數據格式 [1, T, N, 2]
                traj_data = np.array(list(object_trajectories[track_id]))
                traj_tensor = torch.tensor(traj_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                traj_tensor = traj_tensor.permute(0, 2, 1, 3).to(device)  # [1, 1, 20, 2] -> [1, 20, 1, 2]
                
                # 推理
                with torch.no_grad():
                    logits = social_lstm_model(traj_tensor)
                    pred = logits.argmax(dim=-1).item()
                
                # 更新風險狀態
                risk_status[track_id] = pred
                
                # 如果檢測到危險狀態，記錄並準備觸發LLM
                if pred == 1 and not danger_detected:
                    danger_detected = True
                    danger_object_id = track_id
                    logger.warning(f"危險檢測! 物體ID: {track_id}")

                    # ===== 準備動態提示詞 =====
                    # 收集簡單的物體資訊
                    object_info = {
                        'class': class_name,
                        'position': (center_x, center_y)
                    }
                    
                    # 生成動態提示詞 (傳入畫面寬度)
                    dynamic_prompt = generate_dynamic_emergency_prompt(
                        danger_object_id, 
                        object_info, 
                        screen_width=monitor["width"]  # 使用實際畫面寬度
                    )
                    
                    # 設定提示詞（如果動態生成失敗，會自動使用預設的）
                    if dynamic_prompt:
                        openai_integration.set_prompt(dynamic_prompt)
                        logger.info(f"使用動態提示詞: {dynamic_prompt[:100]}...")
                    else:
                        logger.warning("動態提示詞生成失敗，使用預設提示詞")

                    # ❶ 這裡把要分析的幀號換成 [現在, +5, +10]
                    start = frame_id
                    openai_integration.frames_to_analyze = [start, start+5, start+10]

                    # ⭐ 立即通知GUI
                    if gui_message_handler and gui_message_handler.gui_response_callback:
                        gui_message_handler.gui_response_callback("系統偵測到危險")
                        logger.info("已立即通知GUI危險檢測")


            # ===== 檢查物件是否進入危險區域 =====
            # 使用 OpenCV 的點在多邊形內檢測
            point_in_danger_zone = cv2.pointPolygonTest(danger_zone, (center_x, center_y), False) >= 0
            
            if point_in_danger_zone:
                risk_status[track_id] = 1

            if point_in_danger_zone and not danger_detected:
                danger_detected = True
                danger_object_id = track_id
                logger.warning(f"危險區域檢測! 物體ID: {track_id} 進入危險區域")

                

                # ===== 準備動態提示詞 =====
                object_info = {
                    'class': class_name,
                    'position': (center_x, center_y)
                }
                
                # 生成動態提示詞 (傳入畫面寬度)
                dynamic_prompt = generate_dynamic_emergency_prompt(
                    danger_object_id, 
                    object_info, 
                    screen_width=monitor["width"]
                )
                
                # 設定提示詞（如果動態生成失敗，會自動使用預設的）
                if dynamic_prompt:
                    openai_integration.set_prompt(dynamic_prompt)
                    logger.info(f"使用動態提示詞: {dynamic_prompt[:100]}...")
                else:
                    logger.warning("動態提示詞生成失敗，使用預設提示詞")

                start = frame_id
                openai_integration.frames_to_analyze = [start, start+5, start+10]

                # 立即通知GUI
                if gui_message_handler and gui_message_handler.gui_response_callback:
                    gui_message_handler.gui_response_callback("系統偵測到危險")
                    logger.info("已立即通知GUI危險檢測")
            ##############################################################################

            if track_id in manual_danger_objects and manual_danger_objects[track_id]:
                risk_status[track_id] = 1
            elif track_id in manual_danger_objects and not manual_danger_objects[track_id]:
                risk_status[track_id] = 0


        # =========== 4) 統一繪製結果 ============
        draw_results(
            frame=frame, 
            coords=coords, 
            online_tlwhs=online_tlwhs, 
            online_ids=online_ids, 
            online_classes=online_classes, 
            trajectory=Draw_trajectory, 
            frame_id=frame_id, 
            fps=fps, 
            control_text=extracted_code, 
            pause_pressed=pause_pressed,
            risk_status=risk_status,
            display_options=current_display_options  # 使用即時顯示選項
        )



        # =========== 5.1) 處理GUI訊息 ============     
        if gui_message_handler.has_pending_messages() and not gui_processing:
            gui_message_data = gui_message_handler.get_next_message()
            if gui_message_data:
                user_message = gui_message_data['message']
                logger.info(f"開始處理GUI訊息: {user_message}")
                
                # 保存當前幀用於GUI分析
                image_paths = gui_message_handler.save_current_frames_for_gui()
                if image_paths:
                    gui_processing = True
                    gui_message_start_time = time.time()
                    
                    # 使用chat模式處理GUI訊息
                    try:
                        analysis_thread = threading.Thread(
                            target=openai_integration.analyze_images,
                            args=(image_paths, "chat", user_message),
                            daemon=True
                        )
                        analysis_thread.start()
                        logger.info("已啟動GUI訊息分析線程")
                    except Exception as e:
                        logger.error(f"啟動GUI分析時發生錯誤: {e}")
                        gui_processing = False
                else:
                    logger.error("無法保存當前幀用於GUI分析")

        # =========== 5.2) 檢查GUI分析完成 ============
        if gui_processing and openai_integration.is_gui_analysis_done():
            gui_response = openai_integration.get_gui_response()
            if gui_response:
                logger.info(f"GUI分析完成: {gui_response}")
                gui_message_handler.send_response_to_gui(gui_response)
                
            # 重置GUI處理狀態
            gui_processing = False
            openai_integration.reset_gui_status()
            logger.info("GUI處理狀態已重置")

        # =========== 5.3) GUI處理超時檢查 ============
        if gui_processing and gui_message_start_time and (time.time() - gui_message_start_time > 30):
            logger.warning("GUI訊息處理超時，重置狀態")
            gui_processing = False
            gui_message_handler.send_response_to_gui("處理超時，請稍後再試。")
            openai_integration.reset_gui_status()

        # =========== 6.1) OpenAI分析 (緊急模式，保持原有邏輯) ============
        if danger_detected:
            save_done = openai_integration.save_and_analyze(frame_id, frame, save_openai_img_folder)
            
            if save_done and not pause_pressed:
                pause_pressed = True
                logger.warning(f"---------------------檢測到危險物體 ID: {danger_object_id}------------------")

                # 使用世界控制器暫停世界，而不是按鍵
                if world_controller.pause_world():
                    logger.warning("已通過世界控制器暫停CARLA世界")
                else:
                    logger.error("世界控制器暫停失敗")

        # ===== 6.2)等待 OpenAI API 分析完成（緊急模式）並通知GUI =====
        if openai_integration.analysis_done and pause_pressed:
            response = openai_integration.get_last_response()
            if response:
                logger.warning(f"OpenAI 生成的駕駛指令：{response}")
                extracted_code = extract_emergency_code(response)   # 提取指令並執行
                logger.warning(f"OpenAI 生成的駕駛指令：{extracted_code}")
                
                # ===== 將危險檢測結果發送到GUI =====
                if gui_message_handler and gui_message_handler.gui_response_callback:
                    gui_message_handler.send_danger_response_to_gui(response)

            pause_pressed = False  # 重置狀態以便下次可以按 pause

            # 使用世界控制器恢復世界，而不是按鍵
            if world_controller.resume_world():
                logger.warning("已通過世界控制器恢復CARLA世界")
            else:
                logger.error("世界控制器恢復失敗")

            logger.warning("-------------------Resuming----------------")
            emergency_mode = True
            emergency_start_time = time.time()

            # ===== 重置提示詞為預設值 =====
            openai_integration.set_prompt(None)  # 重置為預設提示詞
            logger.info("已重置為預設提示詞")

        # =========== 7) 車輛控制  ============
        if vehicle_ok:                  # → 只有真的抓到 vehicle 才呼叫 control_vehicle

            # =========== 7.1) 車道線維持控制 ============
            steer = 0.0
            if len(coords) >= 2:
                lane_center, coeffs_pair = calculate_lane_center(
                    coords, 
                    frame_width=monitor["width"],
                    frame_height=monitor["height"],
                    look_ahead=150,  # 根據實際場景調整
                    tracker=lane_tracker
                )
                #deviation = lane_center - monitor["width"] / 2
                deviation = calculate_deviation(lane_center, monitor["width"])
                a, b, c = coeffs_pair[0] 

                curvature = abs(2 * a) / ( (1 + (2*a*150 + b)**2) ** 1.5 )

                kappa_thresh = 0.0005
                if curvature < kappa_thresh:
                    # 「直線模式」
                    pid_controller.kp = 0.18
                    lane_tracker.smoothing_factor = 0.6
                else:
                    # 「彎道模式」
                    pid_controller.kp = 0.35
                    lane_tracker.smoothing_factor = 0.25


                steer = lane_keeping_control(deviation, monitor["width"], pid_controller, dt)
                #steer = lane_keeping_control(deviation, monitor["width"], pid_controller)

            # =========== 7.2) 安全距離維持  ============
            obstacle_throttle, obstacle_brake = obstacle_avoidance_control_improved(
                online_tlwhs=online_tlwhs,
                online_classes=online_classes,
                coords=coords,
                frame_width=monitor["width"],
                frame_height=monitor["height"]
            )


            vel = vehicle.get_velocity()
            # CARLA 給的是 m/s，要換成 km/h
            speed_mps = (vel.x**2 + vel.y**2 + vel.z**2) ** 0.5
            speed_kmh = speed_mps * 3.6

            #logger.info(f"車速 (km/h)： {speed_kmh:.2f}")


            #  巡航 PID：算出理想油門 (0.0 ~ 1.0)
            speed_error     = target_speed - speed_kmh
            pid_out = speed_pid.compute(speed_error, dt)


            if pid_out >= 0:
                cruise_throttle = np.clip(pid_out, 0.0, 1.0)
                cruise_brake    = 0.0
            else:
                cruise_throttle = 0.0
                cruise_brake    = np.clip(-pid_out, 0.0, 1.0)

            # 最終油門 = 取 巡航油門 和 避障油門 的 min
            brake = max(cruise_brake, obstacle_brake)
            #throttle = min(cruise_throttle, obstacle_throttle)
            if brake > 0:
                throttle = 0.0
            else:
                throttle = min(cruise_throttle, obstacle_throttle)

            if emergency_mode == True:
                if extracted_code:
                    exec(extracted_code)
                if time.time() - emergency_start_time >= emergency_duration:
                    emergency_mode = False
                    extracted_code = None
            else:
                control_vehicle(vehicle, throttle=throttle, steer=steer, brake=brake)

        # 顯示與輸出
        if frame_callback:  # 如果有回調函數，則調用它
            frame_callback(frame)
        else:  # 否則使用 OpenCV 顯示
            cv2.imshow("YOLOX Screen Tracking", frame)
            
        if args.save_result:
            vid_writer.write(frame)
            
        # 檢查鍵盤輸入 (僅當沒有回調函數時)
        if not frame_callback:
            ch = cv2.waitKey(1)
            if ch == 27 or ch in [ord("q"), ord("Q")]:
                logger.info("Exiting...")
                break

        frame_id += 1

     # 保存結果文件
    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.write('\n'.join([f"{tid}" for tid in trajectories.keys()]))
        logger.info(f"save results to {res_file}")
        

    # 僅當使用 OpenCV 顯示時才關閉窗口
    if not frame_callback:
        cv2.destroyAllWindows()
        logger.add("logs/my_log_{time}.log", rotation="10 MB", encoding="utf-8", retention="10 days")
    
    return True  # 返回成功狀態


def main_script():
    # 使用 make_parser 來獲取命令行參數
    args = make_parser().parse_args()

    #initialize_model_and_transforms()

    vis_folder = 'YOLO_track_outputs'
    os.makedirs(vis_folder, exist_ok=True)
    current_time = time.localtime()
    screenflow_demo(vis_folder, current_time, args)



if __name__ == "__main__":
    main_script()