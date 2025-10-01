import cv2
import torch
import numpy as np

class draw_trajectory:
    """
    用於繪製物件的軌跡。
        - trajectory: 更新指定 ID 的軌跡。
        - draw: 在圖像上繪製所有軌跡。
        - remove_inactive_trajectories: 移除超過最大未更新幀數的軌跡。
    """
    def __init__(self):
        self.trajectories = {}          # 用於存儲每個ID的歷史軌跡
        self.last_update_frame = {}
        
    def trajectory(self,tid, frame_id, tlwh):
        """
        更新指定 ID 的軌跡。
        
        args：
            tid (int): 物件 ID。
            frame_id (int): 當前幀的 ID。
            tlwh (tuple): 包含物件的左上角座標 (x, y)、寬度 (w)、高度 (h)。
        """
        # 更新軌跡字典
        if tid not in self.trajectories:
            self.trajectories[tid] = []
        self.trajectories[tid].append((int(tlwh[0] + tlwh[2] / 2), int(tlwh[1] + tlwh[3] / 2)))
        self.last_update_frame[tid] = frame_id

        # 保留每個ID的最近15個點
        if len(self.trajectories[tid]) > 15:
            self.trajectories[tid].pop(0)

    # ===== 添加顯示控制參數 =====
    def draw(self, online_im, show_trajectory=True):
        """
        在圖像上繪製所有物件的軌跡。
        
        args：
            online_im (numpy array): 當前幀的圖像。
            show_trajectory (bool): 是否顯示軌跡，新增參數
        """
        # 只有在 show_trajectory 為 True 時才繪製軌跡
        if not show_trajectory:
            return
            
        for tid, traj in self.trajectories.items():
            for i in range(1, len(traj)):
                cv2.line(online_im, traj[i - 1], traj[i], (0, 255, 0), 2)

    def remove_inactive_trajectories(self, current_frame_id, max_inactive_frames=1):
        """
        刪除超過 max_inactive_frames 未更新的軌跡。
        
        args：
            current_frame_id (int): 當前幀的 ID。
            max_inactive_frames (int): 最大允許的未更新幀數。
        """
        inactive_ids = []
        for tid, last_frame in self.last_update_frame.items():
            if current_frame_id - last_frame > max_inactive_frames:
                inactive_ids.append(tid)

        for tid in inactive_ids:
            del self.trajectories[tid]
            del self.last_update_frame[tid]


def draw_results(frame, coords, online_tlwhs, online_ids, online_classes, trajectory, frame_id, fps, control_text=None, pause_pressed=False, risk_status=None, display_options=None):
    """
    根據偵測結果在畫面上繪製車道線和物件的 bounding box。

    Args:
        frame: 當前幀影像 (numpy array)
        coords: 車道線座標列表 [[(x1, y1), (x2, y2), ...], ...]
        online_tlwhs: 物件的 bounding box 列表 [(x, y, w, h), ...]
        online_ids: 物件的 ID 列表 [id1, id2, ...]
        trajectory: draw_trajectory 對象，用於繪製軌跡
        frame_id: 當前幀的ID
        fps: 每秒幀數
        control_text: 控制文本
        pause_pressed: 是否暫停
        risk_status: 物件風險狀態字典 {id: risk_value}
        display_options: 顯示選項字典，新增參數
    """

    if display_options is None:
        display_options = {
            'show_trajectory': True,
            'show_bounding_box': True,
            'show_lane_detection': True
        }
    
    '''
    # ============== 在左上角顯示幀號和 FPS ==============
    info_text = f"Frame: {frame_id}, FPS: {fps:.2f}"
    cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    '''

    '''
    if control_text:
        control_code = f": Advanced Code: {control_text}"
        cv2.putText(frame, control_code, (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        if pause_pressed:
            cv2.putText(frame, "PAUSE", (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)       
        else:
            cv2.putText(frame, "Basic Code", (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    '''
    
    
    # ============== 繪製車道線 ==============
    if display_options.get('show_lane_detection', True):
        for idx, lane in enumerate(coords):
            if idx in [0,1]:  # 只繪製第 0 與第 1 條車道線
                for coord in lane:
                    cv2.circle(frame, coord, 5, (0, 255, 0), -1)


    # ============== 繪製物件 bounding box、ID 和類別標籤 ==============
    if display_options.get('show_bounding_box', True):
        for tlwh, tid, cls in zip(online_tlwhs, online_ids, online_classes):
            x, y, w, h = map(int, tlwh)
            color = (255, 0, 0)  # bounding box 顏色

            # 如果有風險狀態，根據風險狀態設置顏色
            if risk_status and tid in risk_status:
                if risk_status[tid] == 1:  # 危險
                    color = (0, 0, 255)  # 紅色
                else:  # 安全
                    color = (0, 255, 0)  # 綠色

            # 繪製 bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # 準備標籤文字
            label = f"ID: {tid}, Class: {cls}"
            
            # 如果有風險狀態，添加到標籤
            if risk_status and tid in risk_status:
                risk_label = "dang" if risk_status[tid] == 1 else "safe"
                label = f"ID: {tid}, Class: {cls}, risk: {risk_label}"
            
            # 文字背景框
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - text_height - baseline - 5), (x + text_width, y), color, -1)

            # 繪製文字（白色字體）
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


    # ============== 繪製軌跡 ==============
    trajectory.draw(online_im=frame, show_trajectory=display_options.get('show_trajectory', True))




def draw_danger_zone(frame, danger_zone, alpha=0.3, color=(0,0,255)):
    """
    在 frame 上用半透明方式繪製危險區域梯形。
    frame: BGR 影像
    danger_zone: np.ndarray (4,2) 區域頂點
    alpha: 透明度
    color: BGR 顏色
    """
    overlay = frame.copy()
    cv2.fillPoly(overlay, [danger_zone], color=color)
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, dst=frame)
    cv2.polylines(frame, [danger_zone], isClosed=True, color=color, thickness=2)