import argparse
from UFLD.utils.common import str2bool
from UFLD.utils.config import Config
import numpy as np
from UFLD.utils.dist_utils import dist_print

import subprocess
import re

import time


def make_parser():
    """
    設定命令行參數解析器，支持 ByteTrack 和 UFLD 所需的多種參數。
    Returns:
        argparse.ArgumentParser: 已配置的參數解析器。
    """    
    parser = argparse.ArgumentParser("ByteTrack Demo!")

    # UFLD 相關參數
    parser.add_argument('--config', help = 'path to config file')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dataset', default = None, type = str)
    parser.add_argument('--data_root', default = None, type = str)
    parser.add_argument('--epoch', default = None, type = int)
    parser.add_argument('--batch_size', default = None, type = int)
    parser.add_argument('--optimizer', default = None, type = str)
    parser.add_argument('--learning_rate', default = None, type = float)
    parser.add_argument('--weight_decay', default = None, type = float)
    parser.add_argument('--momentum', default = None, type = float)
    parser.add_argument('--scheduler', default = None, type = str)
    parser.add_argument('--steps', default = None, type = int, nargs='+')
    parser.add_argument('--gamma', default = None, type = float)
    parser.add_argument('--warmup', default = None, type = str)
    parser.add_argument('--warmup_iters', default = None, type = int)
    parser.add_argument('--backbone', default = None, type = str)
    parser.add_argument('--griding_num', default = None, type = int)
    parser.add_argument('--use_aux', default = None, type = str2bool)
    parser.add_argument('--sim_loss_w', default = None, type = float)
    parser.add_argument('--shp_loss_w', default = None, type = float)
    parser.add_argument('--note', default = None, type = str)
    parser.add_argument('--log_path', default = None, type = str)
    parser.add_argument('--finetune', default = None, type = str)
    parser.add_argument('--resume', default = None, type = str)
    parser.add_argument('--test_model', default = None, type = str)
    parser.add_argument('--test_work_dir', default = None, type = str)
    parser.add_argument('--num_lanes', default = None, type = int)
    parser.add_argument('--auto_backup', action='store_false', help='automatically backup current code in the log path')
    parser.add_argument('--var_loss_power', default = None, type = float)
    parser.add_argument('--num_row', default = None, type = int)
    parser.add_argument('--num_col', default = None, type = int)
    parser.add_argument('--train_width', default = None, type = int)
    parser.add_argument('--train_height', default = None, type = int)
    parser.add_argument('--num_cell_row', default = None, type = int)
    parser.add_argument('--num_cell_col', default = None, type = int)
    parser.add_argument('--mean_loss_w', default = None, type = float)
    parser.add_argument('--fc_norm', default = None, type = str2bool)
    parser.add_argument('--soft_loss', default = None, type = str2bool)
    parser.add_argument('--cls_loss_col_w', default = None, type = float)
    parser.add_argument('--cls_ext_col_w', default = None, type = float)
    parser.add_argument('--mean_loss_col_w', default = None, type = float)
    parser.add_argument('--eval_mode', default = None, type = str)
    parser.add_argument('--eval_during_training', default = None, type = str2bool)
    parser.add_argument('--split_channel', default = None, type = str2bool)
    parser.add_argument('--match_method', default = None, type = str, choices = ['fixed', 'hungarian'])
    parser.add_argument('--selected_lane', default = None, type = int, nargs='+')
    parser.add_argument('--cumsum', default = None, type = str2bool)
    parser.add_argument('--masked', default = None, type = str2bool)

    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # tracking args
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=5.0,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")

    parser.add_argument(
        "--window_id", type=str, default=None,
        help="Specify the title of the window to capture"
    )
    parser.add_argument(
        "--window_name", type=str, default="pygame window",
        help="Specify the title of the window to capture (e.g. 'pygame window')"
    )
    return parser


def merge_config():
    """
    合併命令行參數和配置文件中的參數。
    Returns:
        args: 命令行參數對象
        cfg: 配置文件對象
    """
    args = make_parser().parse_args()
    cfg = Config.fromfile(args.config)

    # 更新配置中的參數
    items = ['dataset','data_root','epoch','batch_size','optimizer','learning_rate',
    'weight_decay','momentum','scheduler','steps','gamma','warmup','warmup_iters',
    'use_aux','griding_num','backbone','sim_loss_w','shp_loss_w','note','log_path',
    'finetune','resume', 'test_model','test_work_dir', 'num_lanes', 'var_loss_power', 'num_row', 'num_col', 'train_width', 'train_height',
    'num_cell_row', 'num_cell_col', 'mean_loss_w','fc_norm','soft_loss','cls_loss_col_w', 'cls_ext_col_w', 'mean_loss_col_w', 'eval_mode', 'eval_during_training', 'split_channel', 'match_method', 'selected_lane', 'cumsum', 'masked']
    for item in items:
        if getattr(args, item) is not None:
            dist_print('merge ', item, ' config')
            setattr(cfg, item, getattr(args, item))

    # 設定數據集相關錨點參數
    if cfg.dataset == 'CULane':
        cfg.row_anchor = np.linspace(0.42,1, cfg.num_row)
        cfg.col_anchor = np.linspace(0,1, cfg.num_col)
    elif cfg.dataset == 'Tusimple':
        cfg.row_anchor = np.linspace(160,710, cfg.num_row)/720
        cfg.col_anchor = np.linspace(0,1, cfg.num_col)
    elif cfg.dataset == 'CurveLanes':
        cfg.row_anchor = np.linspace(0.4, 1, cfg.num_row)
        cfg.col_anchor = np.linspace(0, 1, cfg.num_col)
    
    return args, cfg


def get_window_geometry_by_id(window_id):
    """
    使用 xwininfo 獲取指定 Window ID 的幾何資訊。

    Args:
        window_id (str): 要擷取的視窗 ID（16 進位字串，如 '0x2a00004'）。

    Returns:
        tuple: 包含 (x, y, width, height) 的視窗幾何資訊。
    """
    try:
        # 使用 subprocess 呼叫 xwininfo 並過濾出視窗資訊
        result = subprocess.run(['xwininfo', '-id', window_id], stdout=subprocess.PIPE, text=True)
        if result.returncode != 0 or "xwininfo" not in result.stdout:
            raise Exception(f"No window with ID '{window_id}' found!")

        output = result.stdout

        # 從輸出中提取座標和尺寸資訊
        x = int(next(line.split(":")[1].strip() for line in output.splitlines() if "Absolute upper-left X" in line))
        y = int(next(line.split(":")[1].strip() for line in output.splitlines() if "Absolute upper-left Y" in line))
        width = int(next(line.split(":")[1].strip() for line in output.splitlines() if "Width" in line))
        height = int(next(line.split(":")[1].strip() for line in output.splitlines() if "Height" in line))

        return x, y, width, height

    except Exception as e:
        print(f"Error: {e}")
        return None
    

def get_window_geometry_by_name(window_name):
    """
    使用 xwininfo -name 根據視窗標題抓取視窗幾何資訊。

    Args:
        window_name (str): 要擷取的視窗標題（可部分匹配）。

    Returns:
        tuple: (x, y, width, height)，或在失敗時回傳 None。
    """
    if window_name is None:
        print(f"[get_window_geometry_by_name] 錯誤：window_name 為 None")
        return None

    try:
        # ─── 改動 1 ───
        # 不再用 wmctrl，改成直接用 xwininfo -name
        # xwininfo 會搜尋第一個 title 完全符合的視窗
        result = subprocess.run(
            ['xwininfo', '-name', window_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            # 如果找不到視窗或命令失敗，就拋例外
            raise RuntimeError(f"xwininfo 找不到標題為 '{window_name}' 的視窗")

        out = result.stdout

        # ─── 原本的解析邏輯 ───
        # 用正則從輸出提取數值
        x      = int(next(
            re.search(r"Absolute upper-left X:\s*(\d+)", line).group(1)
            for line in out.splitlines() if "Absolute upper-left X" in line
        ))
        y      = int(next(
            re.search(r"Absolute upper-left Y:\s*(\d+)", line).group(1)
            for line in out.splitlines() if "Absolute upper-left Y" in line
        ))
        width  = int(next(
            re.search(r"Width:\s*(\d+)", line).group(1)
            for line in out.splitlines() if line.strip().startswith("Width")
        ))
        height = int(next(
            re.search(r"Height:\s*(\d+)", line).group(1)
            for line in out.splitlines() if line.strip().startswith("Height")
        ))

        return x, y, width, height

    except Exception as e:
        # ─── 改動 2 ───
        # 把錯誤打印改得更明確，並回傳 None
        print(f"[get_window_geometry_by_name] 錯誤：{e}")
        return None
    

class SimpleTimer:
    """
    簡單的計時器類，用於替代ByteTrack的Timer。
    """
    def __init__(self):
        self.start_time = 0
        self.end_time = 0
        self.times = []
        
    def tic(self):
        """開始計時"""
        self.start_time = time.time()
        
    def toc(self):
        """結束計時，返回經過的時間（秒）"""
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        self.times.append(elapsed)
        if len(self.times) > 30:  # 保留最近30個時間
            self.times.pop(0)
        return elapsed
        
    @property
    def average_time(self):
        """計算平均時間"""
        if not self.times:
            return 0
        return sum(self.times) / len(self.times)    
    


def danger_zone_from_lane(coords, frame_shape):
    """
    根據左右車道線 coords，動態計算梯形危險區域頂點。
    coords: [left_lane_pts, right_lane_pts]
    frame_shape: frame.shape (h, w, _)
    回傳: np.ndarray shape (4,2) order=[left_mid, right_mid, right_bot, left_bot]
    """
    h, w = frame_shape[:2]
    if len(coords) < 2 or not coords[0] or not coords[1]:
        # fallback to fixed trapezoid
        return np.array([
            (int(0.4*w), int(0.55*h)),
            (int(0.6*w), int(0.55*h)),
            (int(0.8*w), int(0.95*h)),
            (int(0.2*w), int(0.95*h)),
        ], dtype=np.int32)
    left_lane, right_lane = coords[0], coords[1]
    left_mid  = left_lane[len(left_lane)//2]
    right_mid = right_lane[len(right_lane)//2]
    #left_mid  = (int(0.46*w), int(0.58*h))
    #right_mid = (int(0.54*w), int(0.58*h))
    left_bot  = left_lane[-1]
    right_bot = right_lane[-1]
    return np.array([left_mid, right_mid, right_bot, left_bot], dtype=np.int32)