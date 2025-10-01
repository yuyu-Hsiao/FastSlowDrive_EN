# 引入 openai 相關模組
import threading
from openai import OpenAI
import base64
from loguru import logger
import os.path as osp
import re
import cv2
import time

class OpenAIIntegration:
    def __init__(self, api_key=None, output_path=None, prompt=None):
        # API金鑰管理（延續之前的改進）
        self.api_key = api_key or self._get_api_key_from_env()
        
        # 只有在提供了有效API金鑰時才初始化客戶端
        self.client = None
        if self.api_key:
            try:
                self.client = OpenAI(api_key=self.api_key)
                logger.info("OpenAI 客戶端初始化成功")
            except Exception as e:
                logger.error(f"OpenAI 客戶端初始化失敗: {e}")
        else:
            logger.warning("未提供 OpenAI API 金鑰，LLM 功能將被禁用")
        
        self.output_path = output_path or "openai_responses.txt"
        self.saved_images = []  # 用於存儲待分析的影像路徑
        self.frames_to_analyze = range(90 - 5, 90 + 6, 5)  # 需要分析的幀範圍
        
        # ===== 多種系統提示詞 =====
        self.emergency_system_prompt = (
            "You are the driver of an autonomous vehicle. "
            "Decide the correct driving command from the given images.\n\n"
            "When you answer, ALWAYS follow *exactly* this format:\n"
            "```python\n"
            "control_vehicle(vehicle, throttle=<float>, steer=<float>, brake=<float>)\n"
            "```\n"
            "ADVICE: <1 short sentence>\n"
            "ANALYSIS: <1 short sentence>\n\n"
            "Do NOT output anything else."
        )
        
        self.chat_system_prompt = (
            "You are an AI assistant for an autonomous driving system. "
            "Analyze the provided driving images and answer the user's question helpfully. "
            "Focus on safety, road conditions, and driving advice. "
            "Provide clear, concise answers based on what you can see in the images."
        )
        
        # ===== 不同場景的預設用戶提示詞 =====
        self.default_emergency_prompt = prompt or (
            "Here are three consecutive driving images. "
            "Use your best judgement to keep the vehicle safe and on course. "
            "Generate the control command, a brief driving suggestion, "
            "and a short scene analysis."
        )
        
        self.default_chat_prompt = (
            "Please analyze the current driving scene and provide your assessment."
        )
        
        # ===== 狀態管理：分離緊急模式和GUI模式 =====
        # 緊急模式狀態
        self.analysis_done = False
        self.last_response = None
        
        # GUI模式狀態
        self.gui_analysis_done = False
        self.last_gui_response = None

        # ===== 新增：線程安全鎖 =====
        self._gui_lock = threading.Lock()
        self._emergency_lock = threading.Lock()

    def _get_api_key_from_env(self):
        """從環境變數獲取API金鑰"""
        import os
        return os.getenv('OPENAI_API_KEY')

    def set_api_key(self, api_key):
        """動態設定API金鑰"""
        self.api_key = api_key
        if api_key:
            try:
                self.client = OpenAI(api_key=api_key)
                logger.info("OpenAI API 金鑰更新成功")
                return True
            except Exception as e:
                logger.error(f"API 金鑰設定失敗: {e}")
                return False
        else:
            self.client = None
            logger.warning("API 金鑰已清除")
            return False

    def is_api_available(self):
        """檢查API是否可用"""
        return self.client is not None

    def set_prompt(self, prompt):
        """外部可動態更改緊急情況的提示詞"""
        self.default_emergency_prompt = prompt

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_images(self, image_paths, mode="emergency", user_prompt=None):
        """
        重構的影像分析方法，支援多種模式
        
        Args:
            image_paths: 影像路徑列表
            mode: 分析模式 ("emergency" 或 "chat")
            user_prompt: 自定義用戶提示詞，如果為None則使用預設值
        """
        if not self.is_api_available():
            logger.error("OpenAI API 不可用，無法進行影像分析")
            if mode == "emergency":
                self.last_response = "API 不可用"
                self.analysis_done = True
            else:  # mode == "chat"
                self.last_gui_response = "抱歉，OpenAI API 目前不可用。"
                self.gui_analysis_done = True
            return

        try:
            # 根據模式選擇對應的提示詞和狀態
            if mode == "emergency":
                logger.info('--------------------- 準備進行緊急影像分析 ------------------')
                system_content = self.emergency_system_prompt
                user_text = user_prompt or self.default_emergency_prompt

                # ===== 記錄發送的緊急提示詞 =====
                prompt_type = "動態提示詞" if "⚠️ DANGER ALERT" in user_text else "預設提示詞"
                logger.info(f"[緊急模式] 使用 {prompt_type}")
                logger.info(f"[緊急模式] System 提示詞: {system_content}")
                logger.info(f"[緊急模式] User 提示詞: {user_text}")


                # 重置緊急模式狀態
                self.analysis_done = False
                self.last_response = None
                
            elif mode == "chat":
                logger.info('--------------------- 準備進行GUI影像分析 ------------------')
                system_content = self.chat_system_prompt
                user_text = user_prompt or self.default_chat_prompt

                # ===== 記錄GUI用戶輸入 =====
                logger.info(f"[緊急模式] System 提示詞: {system_content}")
                logger.info(f"[緊急模式] User 提示詞: {user_text}")
  

                # 重置GUI模式狀態
                self.gui_analysis_done = False
                self.last_gui_response = None
                
            else:
                raise ValueError(f"不支援的模式: {mode}")
            
            # 將所有影像編碼
            encoded_images = [self.encode_image(p) for p in image_paths]
            if not encoded_images:
                logger.error("沒有需要分析的圖片")
                if mode == "emergency":
                    self.last_response = "無分析圖片"
                    self.analysis_done = True
                else:
                    self.last_gui_response = "沒有可分析的圖像"
                    self.gui_analysis_done = True
                return

            # 構建訊息
            messages = [
                {
                    "role": "system",
                    "content": system_content
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        *[
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{encoded_image}"
                                }
                            }
                            for encoded_image in encoded_images
                        ]
                    ]
                }
            ]

            # 調用API
            completion = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=800 if mode == "chat" else 400  # GUI模式允許更長回應
            )
            
            response = completion.choices[0].message.content
            
            # 根據模式記錄日誌和儲存結果
            if mode == "emergency":
                logger.info(f'ChatGPT 緊急分析回應: {response}')
                self.last_response = response
            else:  # mode == "chat"
                logger.info(f'ChatGPT GUI分析回應: {response}')
                self.last_gui_response = response

            # 保存到檔案
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            mode_label = "緊急分析" if mode == "emergency" else "GUI分析"
            with open(self.output_path, "a", encoding='utf-8') as f:
                f.write(f"\n[{timestamp}] {mode_label}回應:\n")
                if user_prompt and mode == "chat":
                    f.write(f"用戶問題: {user_prompt}\n")
                f.write(f"回應: {response}\n")
            
            logger.info(f"{mode_label}結果已保存到 {self.output_path}")

        except Exception as e:
            error_msg = f"影像分析失敗: {str(e)}"
            logger.error(error_msg)
            if mode == "emergency":
                self.last_response = error_msg
            else:
                self.last_gui_response = f"分析失敗: {str(e)}"
        finally:
            # 標記對應模式的分析完成
            if mode == "emergency":
                self.analysis_done = True
            else:
                self.gui_analysis_done = True

    def send_gui_message_with_images(self, user_message, image_paths):
        """
        專門處理GUI發送的文字問題+影像的方法
        
        Args:
            user_message: 用戶在GUI中輸入的問題
            image_paths: 當前影像路徑列表
        """
        logger.info(f"收到GUI訊息: {user_message}")
        
        # 在後台執行分析
        threading.Thread(
            target=self.analyze_images,
            args=(image_paths, "chat", user_message),
            daemon=True
        ).start()

    def get_gui_response(self):
        """獲取GUI分析的回應"""
        with self._gui_lock:
            if self.gui_analysis_done:
                return self.last_gui_response
            return None

    def is_gui_analysis_done(self):
        """檢查GUI分析是否完成"""
        with self._gui_lock:
            return self.gui_analysis_done

    def reset_gui_status(self):
        """重置GUI相關狀態"""
        with self._gui_lock:
            self.gui_analysis_done = False
            self.last_gui_response = None

    def save_and_analyze(self, frame_id, frame, save_folder):
        """
        原有的影像保存和分析功能（緊急模式）
        保持向後相容性
        """
        if frame_id in self.frames_to_analyze:
            logger.info(f"[OpenAI] Capturing frame {frame_id} for analysis")

        if frame_id in self.frames_to_analyze:
            image_path = osp.join(save_folder, f"frame_{frame_id}.png")
            cv2.imwrite(image_path, frame)
            self.saved_images.append(image_path)
            if len(self.saved_images) == 3:
                image_paths = self.saved_images.copy()
                # 使用緊急模式進行分析
                threading.Thread(
                    target=self.analyze_images, 
                    args=(image_paths, "emergency"),
                    daemon=True
                ).start()
                self.saved_images.clear()
                
                return True     # 回傳 true 代表開始分析
        return False

    def get_last_response(self):
        """獲取 OpenAI API 產生的最新緊急分析回應"""
        with self._emergency_lock:
            if self.analysis_done:
                return self.last_response
            return None

    def reset_analysis_status(self):
        """重置所有分析狀態"""
        # 緊急模式狀態
        with self._emergency_lock:
            self.analysis_done = False
            self.last_response = None
        
        # GUI模式狀態
        with self._gui_lock:
            self.gui_analysis_done = False
            self.last_gui_response = None
        
        logger.info("所有分析狀態已重置")

    # ===== 向後相容性方法 =====
    def set_emergency_prompt(self, prompt):
        """設定緊急情況的提示詞"""
        self.default_emergency_prompt = prompt

    def set_chat_prompt(self, prompt):
        """設定GUI對話的提示詞"""
        self.default_chat_prompt = prompt


def extract_emergency_code(api_response):
    """提取緊急控制代碼"""
    # 用正則表達式匹配三個反引號中的程式碼區塊
    pattern = r"```python\s*(.*?)\s*```"
    match = re.search(pattern, api_response, re.DOTALL)
    if match:
        return match.group(1)  # 提取 Python 代碼
    else:
        return None  # 如果沒有匹配到，回傳 None
    


def generate_dynamic_emergency_prompt(danger_object_id, object_info=None, screen_width=None):
    """
    簡單生成動態緊急提示詞
    
    Args:
        danger_object_id: 危險物體ID
        object_info: 物體資訊字典 (可選)
        screen_width: 畫面寬度 (可選)
    
    Returns:
        str: 動態提示詞，失敗則返回 None
    """
    try:
        base_prompt = (
            f"⚠️ DANGER ALERT: Object ID {danger_object_id} has been detected as dangerous! "
            "Here are three consecutive driving images. "
        )
        
        # 如果有物體資訊，加入詳細描述
        if object_info:
            if 'class' in object_info:
                base_prompt += f"The dangerous object is a {object_info['class']}. "
            
            if 'position' in object_info and screen_width:
                x, y = object_info['position']
                center_x = screen_width / 2
                left_threshold = center_x * 0.7   # 左側區域
                right_threshold = center_x * 1.3  # 右側區域
                
                if x < left_threshold:
                    position_desc = "on the left side"
                elif x > right_threshold:
                    position_desc = "on the right side" 
                else:
                    position_desc = "in the center"
                base_prompt += f"It is located {position_desc} of the screen. "
        
        base_prompt += (
            f"Focus on the dangerous object (ID: {danger_object_id}) and "
            "use your best judgement to keep the vehicle safe and on course. "
            "Generate the control command, a brief driving suggestion, "
            "and a short scene analysis."
        )
        
        return base_prompt
    
    except Exception as e:
        logger.error(f"生成動態提示詞失敗: {e}")
        return None  # 返回 None，讓系統使用預設提示詞