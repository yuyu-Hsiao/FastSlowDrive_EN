import carla
from loguru import logger
import numpy as np
from sklearn.linear_model import RANSACRegressor


def initialize_carla():
    # 連接到 CARLA 伺服器
    client = carla.Client('127.0.0.1', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # 手动推进仿真以更新状态手動推進仿真以更新
    if world.get_settings().synchronous_mode:
        world.tick()  # 關键：同步模式下必須調用

        
    # 尋找角色名稱為 'hero' 的車輛
    hero_vehicle = None
    for actor in world.get_actors():
        if actor.attributes.get('role_name') == 'hero':
            hero_vehicle = actor
            break

    if hero_vehicle is None:
        print("未找到角色名稱為 'hero' 的車輛")
        return None

    print("找到車輛，開始控制...")
    return hero_vehicle, world


def control_vehicle(vehicle, throttle=0.5, steer=0.0, brake=0.0):
    control = carla.VehicleControl()
    control.throttle = throttle
    control.steer = steer
    control.brake = brake
    vehicle.apply_control(control)



class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.prev_error = 0
        self.integral = 0

        self.kp_min, self.kp_max = 0.05, 0.3
        self.ki_min, self.ki_max = 0.0, 0.1

    def compute(self, error, dt=1):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output
    
    # ─── 根據「誤差大小／變化率」微調 Kp、Ki ───
    def adaptive_tune(self, error, dt):
        """
        若絕對誤差大，就加大 Kp；誤差小則降低 Kp。
        Ki 也可根據誤差累積速度微調。
        """
        # 1) 根據當前誤差 e(t) 的絕對值來調整比例增益
        abs_err = abs(error)
        # 參數：當誤差 > 0.5 時，Kp 逐步往 max 值靠近；誤差 < 0.1 時，往 min 值靠近
        if abs_err > 0.5:
            self.kp = min(self.kp * 1.05, self.kp_max)  # 誤差大，往上調高 Kp（最多不超過 kp_max）
        elif abs_err < 0.1:
            self.kp = max(self.kp * 0.95, self.kp_min)  # 誤差很小，往下調低 Kp（最低不低於 kp_min）

        # 2) 若誤差持續有同一號正負，則加小量 Ki；否則略微遞減 Ki
        if error * self.prev_error > 0:
            # 本次誤差與上次誤差同號 → 持續偏移，需要加大積分補償
            self.ki = min(self.ki + 0.001, self.ki_max)
        else:
            # 變號或誤差變小，讓 Ki 微幅收斂
            self.ki = max(self.ki * 0.98, self.ki_min)

class LaneTracker:
    def __init__(self, max_lane_shift=50, smoothing_factor=0.2):
        self.prev_lane_center = None
        self.max_lane_shift = max_lane_shift
        self.smoothing_factor = smoothing_factor  # 移動平均權重

        self.large_shift_counter = 0
        self.required_consistent_frames = 10
        

    def get_stable_lane_center(self, new_lane_center, frame_width):
        if self.prev_lane_center is None:
            self.prev_lane_center = new_lane_center
            self.large_shift_counter = 0
            return new_lane_center
        # 計算本幀偏移量
        shift = abs(new_lane_center - self.prev_lane_center)

        # 如果偏移量過大，開始累積 large_shift_counter
        if shift > self.max_lane_shift:
            logger.info(f"偏移過大")
            # 這一幀算作「大位移」
            self.large_shift_counter += 1
            

            if self.large_shift_counter >= self.required_consistent_frames:
                # 如果已經連續足夠多幀都偏移過大，則更新 prev_lane_center
                smoothed = (self.smoothing_factor * new_lane_center +
                            (1 - self.smoothing_factor) * self.prev_lane_center)
                self.prev_lane_center = smoothed
                # 重置計數器，下一波再重新累積
                self.large_shift_counter = 0
                return smoothed
            else:
                # 還未累計到 required_consistent_frames，暫不接受 new 值
                return self.prev_lane_center

        else:
            # 如果本幀偏移量「不超過門檻」，代表車道中心變動在可接受範圍
            # 這時把 large_shift_counter 歸零，並做一次「一般平滑」更新
            self.large_shift_counter = 0
            smoothed = (self.smoothing_factor * new_lane_center +
                        (1 - self.smoothing_factor) * self.prev_lane_center)
            self.prev_lane_center = smoothed
            return smoothed

def fit_lane_curve(lane_points, frame_height, look_ahead=100):
    if len(lane_points) < 3:
        return None, None

    y = np.array([point[1] for point in lane_points]).reshape(-1, 1)
    x = np.array([point[0] for point in lane_points])

    try:
        ransac = RANSACRegressor()
        ransac.fit(y, x)
        coeffs = np.polyfit(y.flatten(), ransac.predict(y), 2)
    except Exception as e:
        #logger.warning(f"擬合失敗: {e}")
        return None, None

    look_ahead_y = frame_height - look_ahead
    predicted_x = coeffs[0] * look_ahead_y ** 2 + coeffs[1] * look_ahead_y + coeffs[2]
    return predicted_x, coeffs 

def calculate_lane_center(coords, frame_width, frame_height, look_ahead=150, tracker=None):
    if len(coords) < 2:
        return frame_width / 2, None 

    left_lane = coords[0]
    right_lane = coords[1]

    left_predicted, left_coeffs = fit_lane_curve(left_lane, frame_height, look_ahead)
    right_predicted, right_coeffs = fit_lane_curve(right_lane, frame_height, look_ahead)

    if left_predicted is None or right_predicted is None:
        #logger.warning("無法擬合車道線，回傳預設中心")
        return frame_width / 2, None 

    lane_center = (left_predicted + right_predicted) / 2

    if tracker:
        lane_center = tracker.get_stable_lane_center(lane_center, frame_width)

    return lane_center, (left_coeffs, right_coeffs)

def calculate_deviation(lane_center, frame_width):
    vehicle_center = frame_width / 2
    deviation = lane_center - vehicle_center
    #logger.info(f"偏差值: {deviation}")
    return deviation


def lane_keeping_control(deviation, frame_width, pid, dt=1):

    norm_err = deviation / (frame_width / 2)

    # ─── 新增：adaptive 調參 ───
    pid.adaptive_tune(norm_err, dt)

    steer = pid.compute(norm_err, dt)
    steer = np.clip(steer, -0.95, 0.95)  # 限制轉向角度以避免過度修正

    #logger.info(f"控制指令 - 轉向角度: {steer}")
    return steer

def obstacle_avoidance_control(online_tlwhs, online_classes, coords, frame_width, frame_height):
    """
    根據 ByteTrack 偵測結果與車道線來進行避障控制。

    Args:
        vehicle: CARLA 模擬車輛物件
        online_tlwhs: 追蹤到的物件 bounding box [(x, y, w, h), ...]
        online_classes: 物件類別 (用於篩選是否為車輛)
        coords: 車道線座標 [[(x1, y1), (x2, y2), ...], ...]
        frame_width: 畫面寬度
        frame_height: 畫面高度
    """
    # 油門值
    throttle = 0.5
    brake = 0.0

    # 設定安全距離閾值 (越小則越保守)
    distance_threshold = frame_height * 0.5  # 距離過近的閾值
    brake_threshold = frame_height * 0.25  # 需要剎車的距離

    # **計算車道中心**
    #lane_center = calculate_lane_center(coords, frame_width, frame_height)

    # 設定當前車道的範圍
    if len(coords) < 2:
        return throttle, brake# 沒偵測到車道線，直接返回

    left_lane_x = min([p[0] for p in coords[0]])  # 左邊界
    right_lane_x = max([p[0] for p in coords[1]])  # 右邊界

    # **檢查前方是否有車輛**
    for tlwh, cls in zip(online_tlwhs, online_classes):
        #if cls != 2:  # 只考慮車輛 (class_id == 2)
        if cls != "Vehicle":
            continue

        x, y, w, h = tlwh
        obj_center_x = x + w // 2
        obj_bottom_y = y + h  # 車輛的底部位置

        # **判斷車輛是否在當前車道範圍內**
        if left_lane_x < obj_center_x < right_lane_x:
            # **計算車輛與車道中心的偏移量**
            deviation = calculate_deviation(obj_center_x, frame_width)

            # **根據距離決定車速**
            if obj_bottom_y > frame_height - distance_threshold:
                # **過近，減速**
                throttle = 0.0
                brake = 0.0
                #logger.info("前方有車輛，減速！")

            if obj_bottom_y > frame_height - brake_threshold:
                # **非常近，需要剎車**
                throttle = 0.0
                brake = 1.0
                #logger.info("前方車輛距離過近，緊急剎車！")
                return  throttle, brake # 立即停止後續控制
    return  throttle, brake



def get_lane_width_at_y(coords, target_y):
    """
    根據給定的y座標，計算該位置的車道寬度
    Args:
        coords: 車道線座標 [left_lane, right_lane]
        target_y: 目標y座標（物體的位置）
    Returns:
        left_x, right_x: 該y位置的左右車道線x座標，如果無法計算則返回None
    """
    if len(coords) < 2 or not coords[0] or not coords[1]:
        return None, None
    
    left_lane = coords[0]
    right_lane = coords[1]
    
    # 找到最接近target_y的車道線點
    def find_closest_x(lane_points, target_y):
        closest_point = None
        min_distance = float('inf')
        
        for point in lane_points:
            x, y = point
            distance = abs(y - target_y)
            if distance < min_distance:
                min_distance = distance
                closest_point = point
        
        if closest_point:
            return closest_point[0]
        return None
    
    # 更精確的方法：線性插值
    def interpolate_x_at_y(lane_points, target_y):
        # 找到target_y前後的兩個點進行插值
        below_points = [(x, y) for x, y in lane_points if y <= target_y]
        above_points = [(x, y) for x, y in lane_points if y >= target_y]
        
        if not below_points or not above_points:
            return find_closest_x(lane_points, target_y)
        
        # 取最近的點
        p1 = max(below_points, key=lambda p: p[1])  # y值最大的below點
        p2 = min(above_points, key=lambda p: p[1])  # y值最小的above點
        
        if p1[1] == p2[1]:  # 避免除零
            return p1[0]
        
        # 線性插值
        ratio = (target_y - p1[1]) / (p2[1] - p1[1])
        interpolated_x = p1[0] + ratio * (p2[0] - p1[0])
        return interpolated_x
    
    left_x = interpolate_x_at_y(left_lane, target_y)
    right_x = interpolate_x_at_y(right_lane, target_y)
    
    return left_x, right_x


def is_in_current_lane_improved(obj_center_x, obj_center_y, coords, frame_width, safety_margin=50):
    """
    改進的車道內判斷函數
    Args:
        obj_center_x, obj_center_y: 物體中心座標
        coords: 車道線座標
        frame_width: 畫面寬度
        safety_margin: 安全邊距（像素）
    Returns:
        bool: 是否在當前車道內
    """
    # 獲取物體y位置對應的車道寬度
    left_x, right_x = get_lane_width_at_y(coords, obj_center_y)
    
    if left_x is None or right_x is None:
        # 如果無法判斷車道線，使用保守策略：畫面中央區域
        center_x = frame_width / 2
        lane_width = frame_width * 0.3  # 假設車道寬度為畫面的30%
        left_boundary = center_x - lane_width / 2
        right_boundary = center_x + lane_width / 2
    else:
        # 添加安全邊距
        left_boundary = left_x - safety_margin
        right_boundary = right_x + safety_margin
    
    return left_boundary <= obj_center_x <= right_boundary


def obstacle_avoidance_control_improved(online_tlwhs, online_classes, coords, frame_width, frame_height):
    """
    改進的避障控制函數
    """
    throttle = 0.5
    brake = 0.0
    
    # 設定安全距離閾值（根據畫面高度的比例）
    distance_threshold = frame_height * 0.35   # 減速區域（提早一點開始減速）
    brake_threshold = frame_height * 0.2      # 剎車區域
    warning_threshold = frame_height * 0.6    # 警告區域
    
    # 用於記錄最近的威脅
    closest_threat_distance = float('inf')
    has_threat = False
    
    for tlwh, cls in zip(online_tlwhs, online_classes):
        if cls != "Vehicle":  # 只考慮車輛
            continue
            
        x, y, w, h = tlwh
        obj_center_x = x + w / 2
        obj_center_y = y + h / 2
        obj_bottom_y = y + h  # 車輛底部位置（更接近我們）
        
        # 使用改進的車道判斷
        if is_in_current_lane_improved(obj_center_x, obj_center_y, coords, frame_width):
            # 計算相對距離（以畫面高度為基準）
            relative_distance = frame_height - obj_bottom_y
            
            # 只考慮在我們前方的車輛（y座標小於我們的位置）
            our_position_y = frame_height * 0.9  # 假設我們在畫面底部90%的位置
            if obj_bottom_y < our_position_y:
                has_threat = True
                closest_threat_distance = min(closest_threat_distance, relative_distance)
                
                # 根據距離調整控制策略
                if obj_bottom_y > frame_height - brake_threshold:
                    # 非常接近，緊急剎車
                    throttle = 0.0
                    brake = 1.0
                    logger.info(f"緊急剎車！前方車輛距離: {relative_distance:.1f}px")
                    return throttle, brake
                    
                elif obj_bottom_y > frame_height - distance_threshold:
                    # 中等距離，減速
                    # 根據距離動態調整減速強度
                    distance_ratio = (obj_bottom_y - (frame_height - distance_threshold)) / (distance_threshold - brake_threshold)
                    throttle = max(0.0, 0.3 * distance_ratio)  # 最低降到0.3倍油門
                    brake = 0.0
                    #logger.info(f"減速中，前方車輛距離: {relative_distance:.1f}px, 油門: {throttle:.2f}")
                    
                elif obj_bottom_y > frame_height - warning_threshold:
                    # 遠距離，輕微調整
                    throttle = 0.2  # 略微減速
                    brake = 0.0
                    #logger.debug(f"警告區域，輕微減速")
    
    #if not has_threat:
        #logger.debug("前方車道無威脅，維持正常速度")
    
    return throttle, brake


def obstacle_avoidance_control_with_lateral_check(online_tlwhs, online_classes, coords, frame_width, frame_height):
    """
    帶有橫向位置檢查的避障控制（更嚴格的車道判斷）
    """
    throttle = 0.5
    brake = 0.0
    
    # 車輛位置假設（在畫面中央底部）
    ego_x = frame_width / 2
    ego_y = frame_height * 0.9
    
    for tlwh, cls in zip(online_tlwhs, online_classes):
        if cls != "Vehicle":
            continue
            
        x, y, w, h = tlwh
        obj_center_x = x + w / 2
        obj_center_y = y + h / 2
        obj_bottom_y = y + h
        
        # 1. 縱向檢查：物體必須在我們前方
        if obj_bottom_y >= ego_y:
            continue  # 在我們後方或同一位置，忽略
        
        # 2. 橫向檢查：物體必須在我們的車道內
        if not is_in_current_lane_improved(obj_center_x, obj_center_y, coords, frame_width, safety_margin=30):
            continue  # 不在我們車道內，忽略
            
        # 3. 距離檢查：計算實際威脅程度
        longitudinal_distance = ego_y - obj_bottom_y
        lateral_distance = abs(obj_center_x - ego_x)
        
        # 組合距離評估（考慮縱向和橫向距離）
        threat_score = longitudinal_distance - lateral_distance * 0.5  # 橫向距離權重較小
        
        if threat_score < 50:  # 非常接近
            throttle = 0.0
            brake = 1.0
            logger.warning(f"緊急避障！威脅評分: {threat_score:.1f}")
            return throttle, brake
        elif threat_score < 150:  # 中等距離
            throttle = max(0.2, throttle * 0.6)
            logger.info(f"減速避障，威脅評分: {threat_score:.1f}")
        elif threat_score < 250:  # 遠距離預警
            throttle = max(0.4, throttle * 0.8)
            logger.debug(f"預警減速，威脅評分: {threat_score:.1f}")
    
    return throttle, brake
