import os
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np

# 將當前目錄加入路徑以導入自定義模組
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'UFLD'))

# 從 Main 模組導入必要的函數
from main import screenflow_demo
import argparse
from autopilot_fun.utils import make_parser

class GUICallbackHandler:
    """處理GUI與MOT之間的回調通信"""
    def __init__(self, gui_app):
        self.gui_app = gui_app
        self.openai_instance = None
        self.message_handler = None

    def set_openai_instance(self, openai_instance):
        """接收MOT傳遞的OpenAI實例"""
        self.openai_instance = openai_instance
        self.gui_app.log("OpenAI instance connected")

    def set_message_handler(self, message_handler):
        """接收MOT傳遞的消息處理器"""
        self.message_handler = message_handler
        # 設置回調函數，當有回應時通知GUI
        if hasattr(message_handler, 'set_gui_callback'):
            message_handler.set_gui_callback(self.handle_mot_response)
            self.gui_app.log("Message handler connected")
            # 更新連接狀態
            self.gui_app.update_connection_status(True)
        else:
            self.gui_app.log("Warning: Message handler missing set_gui_callback method")

    def send_message_to_mot(self, message):
        """發送消息到MOT系統"""
        if self.message_handler and hasattr(self.message_handler, 'add_gui_message'):
            try:
                self.message_handler.add_gui_message(message)
                self.gui_app.log(f"Message sent to MOT: {message}")
                return True
            except Exception as e:
                self.gui_app.log(f"Error sending message to MOT: {e}")
                return False
        else:
            self.gui_app.log("Error: Message handler not connected or missing add_gui_message method")
            return False

    def handle_mot_response(self, response):
        """處理來自MOT的回應"""
        try:
            # 使用 after 方法在主線程中更新GUI
            self.gui_app.root.after(0, lambda: self.gui_app.handle_llm_response(response))
        except Exception as e:
            self.gui_app.log(f"Error handling MOT response: {e}")

class AutoDrivingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Autonomous Driving System")
        self.root.geometry("1600x900")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 設置變數
        self.window_name = tk.StringVar(value="pygame window")
        self.video_path = tk.StringVar()
        self.config_path = tk.StringVar(value="UFLD/configs/culane_res18.py")
        self.model_path = tk.StringVar(value="UFLD/weights/culane_res18.pth")
        self.save_result = tk.BooleanVar(value=True)
        self.running = False
        self.stop_event = threading.Event()
        self.process_thread = None
        
        # 顯示控制變數
        self.show_trajectory = tk.BooleanVar(value=True)
        self.show_bounding_box = tk.BooleanVar(value=True)
        self.show_lane_detection = tk.BooleanVar(value=True)
        
        # 共享的顯示選項字典，用於動態更新
        self.shared_display_options = {
            'show_trajectory': True,
            'show_bounding_box': True,
            'show_lane_detection': True
        }
        
        # LLM輸入相關變數
        self.llm_input_text = tk.StringVar()
        self.llm_messages = []
        self.llm_enabled = tk.BooleanVar(value=True)
        
        # ===== 新增：GUI回調處理器 =====
        self.gui_callback_handler = GUICallbackHandler(self)
        
        # ===== 新增：LLM處理狀態 =====
        self.llm_processing = False
        self.processing_start_time = None
        
        # 綁定變數變更事件
        self.show_trajectory.trace('w', self.update_display_options)
        self.show_bounding_box.trace('w', self.update_display_options)
        self.show_lane_detection.trace('w', self.update_display_options)
        
        # 創建UI組件
        self.create_widgets()
    
    def update_display_options(self, *args):
        """當GUI選項改變時，更新共享的顯示選項字典"""
        self.shared_display_options['show_trajectory'] = self.show_trajectory.get()
        self.shared_display_options['show_bounding_box'] = self.show_bounding_box.get()
        self.shared_display_options['show_lane_detection'] = self.show_lane_detection.get()
        
        if hasattr(self, 'log_text'):
            self.log(f"Display options updated: Trajectory={self.shared_display_options['show_trajectory']}, "
                    f"Bounding Box={self.shared_display_options['show_bounding_box']}, "
                    f"Lane Detection={self.shared_display_options['show_lane_detection']}")
        
    def create_widgets(self):
        # 主要框架 - 分為左側和右側
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左側框架 - 包含控制面板和影像顯示
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # 右側框架 - 包含日誌和LLM對話
        right_frame = ttk.Frame(main_frame, width=500)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(5, 0))
        right_frame.pack_propagate(False)
        
        # ===== 左側內容 =====
        # 頂部控制面板
        top_frame = ttk.Frame(left_frame)
        top_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 創建一個筆記本控件，用於切換不同的輸入模式
        notebook = ttk.Notebook(top_frame)
        notebook.pack(fill=tk.X, padx=5, pady=5)
        
        # 窗口捕獲頁面
        window_frame = ttk.Frame(notebook)
        notebook.add(window_frame, text="Window Capture")
        
        # 窗口名稱輸入
        ttk.Label(window_frame, text="Window Name:").pack(side=tk.LEFT, padx=(0, 5))
        window_entry = ttk.Entry(window_frame, textvariable=self.window_name, width=20)
        window_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        # 視頻文件頁面
        video_frame = ttk.Frame(notebook)
        notebook.add(video_frame, text="Video File")
        
        # 視頻文件路徑
        ttk.Label(video_frame, text="Video File:").pack(side=tk.LEFT, padx=(0, 5))
        video_entry = ttk.Entry(video_frame, textvariable=self.video_path, width=30)
        video_entry.pack(side=tk.LEFT, padx=(0, 5))
        
        # 瀏覽按鈕
        browse_button = ttk.Button(video_frame, text="Browse", command=self.browse_video)
        browse_button.pack(side=tk.LEFT)
        
        # 顯示選項控制區塊
        display_frame = ttk.LabelFrame(top_frame, text="Display Options")
        display_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 創建三個選項的框架，水平排列
        options_inner_frame = ttk.Frame(display_frame)
        options_inner_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 軌跡顯示選項
        trajectory_checkbox = ttk.Checkbutton(
            options_inner_frame, 
            text="Show Trajectory", 
            variable=self.show_trajectory
        )
        trajectory_checkbox.pack(side=tk.LEFT, padx=(0, 20))
        
        # 物件框顯示選項
        bbox_checkbox = ttk.Checkbutton(
            options_inner_frame, 
            text="Show Bounding Box", 
            variable=self.show_bounding_box
        )
        bbox_checkbox.pack(side=tk.LEFT, padx=(0, 20))
        
        # 車道線顯示選項
        lane_checkbox = ttk.Checkbutton(
            options_inner_frame, 
            text="Show Lane Detection", 
            variable=self.show_lane_detection
        )
        lane_checkbox.pack(side=tk.LEFT, padx=(0, 20))
        
        # 控制按鈕框架
        control_frame = ttk.Frame(top_frame)
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 開始/停止按鈕
        self.start_button = ttk.Button(control_frame, text="Start", command=self.toggle_process)
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # 退出按鈕
        quit_button = ttk.Button(control_frame, text="Exit", command=self.on_closing)
        quit_button.pack(side=tk.LEFT)
        
        # 保存設置框架
        save_frame = ttk.Frame(top_frame)
        save_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 保存結果選項
        save_checkbox = ttk.Checkbutton(save_frame, text="Save Results", variable=self.save_result)
        save_checkbox.pack(side=tk.LEFT)
        
        # 影像顯示區域
        self.canvas_frame = ttk.LabelFrame(left_frame, text="Video Preview")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.canvas = tk.Label(self.canvas_frame, background="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # ===== 右側內容 =====
        # LLM輸入控制區塊
        llm_frame = ttk.LabelFrame(right_frame, text="AI Driving Assistant")
        llm_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # LLM啟用開關
        llm_enable_frame = ttk.Frame(llm_frame)
        llm_enable_frame.pack(fill=tk.X, padx=5, pady=5)
        
        llm_enable_checkbox = ttk.Checkbutton(
            llm_enable_frame, 
            text="Enable AI Assistant", 
            variable=self.llm_enabled
        )
        llm_enable_checkbox.pack(side=tk.LEFT)
        
        # ===== 新增：連接狀態指示 =====
        self.connection_status = tk.StringVar(value="Disconnected")
        status_label = ttk.Label(llm_enable_frame, textvariable=self.connection_status, foreground="red")
        status_label.pack(side=tk.RIGHT, padx=(10, 0))
        
        # 輸入框和發送按鈕框架
        llm_input_frame = ttk.Frame(llm_frame)
        llm_input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 輸入標籤
        ttk.Label(llm_input_frame, text="Message:").pack(side=tk.LEFT, padx=(0, 5))
        
        # 輸入框
        self.llm_input_entry = ttk.Entry(
            llm_input_frame, 
            textvariable=self.llm_input_text, 
            font=('Arial', 9)
        )
        self.llm_input_entry.pack(side=tk.LEFT, padx=(0, 5), fill=tk.X, expand=True)
        
        # 發送按鈕
        self.send_button = ttk.Button(
            llm_input_frame, 
            text="Send", 
            command=self.send_llm_message,
            width=8
        )
        self.send_button.pack(side=tk.RIGHT)
        
        # 快速命令按鈕框架
        quick_commands_frame = ttk.Frame(llm_frame)
        quick_commands_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(quick_commands_frame, text="Quick Commands:").pack(anchor=tk.W, pady=(0, 5))
        
        # 預設快速命令按鈕
        quick_commands = [
            #("Emergency Stop", "Please execute emergency stop immediately!"),
            #("Slow Down", "Please slow down due to potential hazards."),
            #("Change Lane", "Consider changing lanes if it is safe."),
            #("Resume Normal", "Danger has passed, resume normal driving."),
            #("Status Report", "Please provide current driving status report.")
            ("Safety Status", "Is the current driving situation safe? What potential risks should I be aware of?"),
            ("Forward Risk", "Are the vehicles ahead behaving normally? What should I pay attention to?"),
            ("Lane Change", "Is it safe to change lanes now? Which direction would you recommend?"),
            ("Emergency Check", "Is there any immediate danger? What action should I take?"),
            ("Driving Advice", "Based on the current situation, what driving recommendations do you have?")
        ]
        
        # 創建按鈕網格
        button_frame1 = ttk.Frame(quick_commands_frame)
        button_frame1.pack(fill=tk.X, pady=2)
        
        button_frame2 = ttk.Frame(quick_commands_frame)
        button_frame2.pack(fill=tk.X, pady=2)
        
        # 第一行按鈕
        for i, (button_text, command_text) in enumerate(quick_commands[:3]):
            btn = ttk.Button(
                button_frame1, 
                text=button_text,
                command=lambda cmd=command_text: self.send_quick_command(cmd),
                width=14
            )
            btn.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        # 第二行按鈕
        for i, (button_text, command_text) in enumerate(quick_commands[3:]):
            btn = ttk.Button(
                button_frame2, 
                text=button_text,
                command=lambda cmd=command_text: self.send_quick_command(cmd),
                width=14
            )
            btn.pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        # 綁定Enter鍵到發送功能
        self.llm_input_entry.bind('<Return>', lambda event: self.send_llm_message())
        
        # LLM對話區域
        llm_chat_frame = ttk.LabelFrame(right_frame, text="Conversation Log")
        llm_chat_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 創建文字框架和滾動條
        chat_text_frame = ttk.Frame(llm_chat_frame)
        chat_text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.llm_chat_text = tk.Text(
            chat_text_frame, 
            wrap=tk.WORD,
            font=('Arial', 9),
            height=10
        )
        self.llm_chat_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # LLM對話區域滾動條
        llm_chat_scrollbar = ttk.Scrollbar(chat_text_frame, orient="vertical", command=self.llm_chat_text.yview)
        llm_chat_scrollbar.pack(side="right", fill="y")
        self.llm_chat_text.config(yscrollcommand=llm_chat_scrollbar.set)
        self.llm_chat_text.config(state=tk.DISABLED)
        
        # 清除對話按鈕
        clear_chat_button = ttk.Button(
            llm_chat_frame, 
            text="Clear Conversation", 
            command=self.clear_llm_conversation
        )
        clear_chat_button.pack(pady=5)
        
        # 系統日誌區域
        log_frame = ttk.LabelFrame(right_frame, text="System Log")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 創建日誌文字框架和滾動條
        log_text_frame = ttk.Frame(log_frame)
        log_text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_text = tk.Text(
            log_text_frame, 
            height=8,
            font=('Consolas', 9)
        )
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # 系統日誌滾動條
        log_scrollbar = ttk.Scrollbar(log_text_frame, orient="vertical", command=self.log_text.yview)
        log_scrollbar.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=log_scrollbar.set)
        self.log_text.config(state=tk.DISABLED)
        
        # 清除日誌按鈕
        clear_log_button = ttk.Button(
            log_frame, 
            text="Clear Log", 
            command=self.clear_log
        )
        clear_log_button.pack(pady=5)
        
        # 狀態欄
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 初始化LLM對話區域
        self.add_llm_message("System", "AI Driving Assistant is ready. You can send messages to request driving assistance.", "system")

    # ===== LLM相關方法的增強版本 =====
    def send_llm_message(self):
        """發送用戶輸入的訊息給LLM"""
        if not self.llm_enabled.get():
            messagebox.showwarning("Warning", "AI Assistant is disabled. Please enable it first.")
            return
            
        message = self.llm_input_text.get().strip()
        if not message:
            return

        # 檢查是否正在處理中
        if self.llm_processing:
            messagebox.showinfo("Info", "Processing previous request, please wait...")
            return

        # 檢查是否連接到MOT系統
        if not self.gui_callback_handler.message_handler:
            messagebox.showerror("Error", "Not connected to MOT system. Please start system processing first.")
            return
            
        # 清空輸入框
        self.llm_input_text.set("")
        
        # 添加用戶訊息到對話區域
        self.add_llm_message("User", message, "user")
        
        # 設置處理狀態
        self.llm_processing = True
        self.processing_start_time = time.time()
        self.send_button.config(state=tk.DISABLED, text="Processing...")
        
        # 顯示處理狀態
        self.add_llm_message("System", "Analyzing current driving scene and processing your request, please wait...", "processing")
        
        # 通過回調處理器發送到MOT
        success = self.gui_callback_handler.send_message_to_mot(message)
        if not success:
            self.handle_llm_error("Unable to send message to MOT system")
        
        # 啟動超時檢查
        self.root.after(30000, self.check_processing_timeout)  # 30秒超時
    
    def send_quick_command(self, command):
        """發送快速命令"""
        if not self.llm_enabled.get():
            messagebox.showwarning("Warning", "AI Assistant is disabled. Please enable it first.")
            return
            
        # 將命令設置到輸入框並發送
        self.llm_input_text.set(command)
        self.send_llm_message()

    def handle_llm_response(self, response):
        """處理來自MOT的LLM回應"""
        if response == "系統偵測到危險":
            # 第一條訊息：系統通知，使用特殊樣式
            self.add_llm_message("System", "System detected danger", "danger_system")
            return

        # ===== 處理不同類型的回應 =====
        # 檢查是否為危險檢測回應
        if response.startswith("Danger Detection Alert"):
            # 危險檢測回應，不需要重置處理狀態
            self.add_llm_message("System Alert", response, "danger")
            self.log("Received danger detection alert")
            
            # 播放警報音效（可選）
            try:
                # 可以在這裡添加警報音效
                pass
            except:
                pass
        else:
            # 普通GUI訊息回應
            if self.llm_processing:
                # 重置處理狀態
                self.llm_processing = False
                self.send_button.config(state=tk.NORMAL, text="Send")
                
                # 移除處理中的消息
                self.remove_processing_message()
            
            # 添加助手回應
            self.add_llm_message("Assistant", response, "assistant")
            self.log(f"Received LLM response: {response[:100]}...")

    def handle_llm_error(self, error_message):
        """處理LLM錯誤"""
        if self.llm_processing:
            self.llm_processing = False
            self.send_button.config(state=tk.NORMAL, text="Send")
            self.remove_processing_message()
            
        self.add_llm_message("System", f"Error: {error_message}", "error")
        self.log(f"LLM Error: {error_message}")

    def check_processing_timeout(self):
        """檢查處理超時"""
        if self.llm_processing and self.processing_start_time:
            if time.time() - self.processing_start_time > 30:
                self.handle_llm_error("Request processing timeout, please retry")

    def remove_processing_message(self):
        """移除處理中的消息"""
        # 獲取對話文本內容
        self.llm_chat_text.config(state=tk.NORMAL)
        content = self.llm_chat_text.get("1.0", tk.END)
        lines = content.split('\n')
        
        # 找到並移除最後一條包含"正在分析"的消息
        new_lines = []
        for line in lines:
            if "Analyzing current driving scene and processing your request" not in line:
                new_lines.append(line)
        
        # 重新設置文本內容
        self.llm_chat_text.delete("1.0", tk.END)
        self.llm_chat_text.insert("1.0", '\n'.join(new_lines))
        self.llm_chat_text.config(state=tk.DISABLED)
        
        # 重新應用標籤樣式
        self.apply_message_styles()

    def add_llm_message(self, sender, message, message_type="user"):
        """添加訊息到LLM對話區域"""
        self.llm_chat_text.config(state=tk.NORMAL)
        
        # 添加時間戳
        timestamp = time.strftime("%H:%M:%S")
        
        # 根據訊息類型設置不同的格式和顏色
        color_map = {
            "user": "blue",
            "assistant": "green", 
            "system": "gray",
            "processing": "orange",
            "error": "red",
            "danger": "#FF4500", 
            "danger_system": "red"
        }
        
        sender_map = {
            "user": "You",
            "assistant": "Assistant",
            "system": "System",
            "processing": "System",
            "error": "Error",
            "danger": "Danger Alert",  
            "danger_system": "System"
        }
        
        display_sender = sender_map.get(message_type, sender)
        

        # ===== 為危險警報添加特殊格式 =====
        if message_type == "danger":
            # 為危險警報添加特殊格式
            self.llm_chat_text.insert(tk.END, f"\n{'='*50}\n", "danger_separator")
            self.llm_chat_text.insert(tk.END, f"[{timestamp}] {display_sender}\n{message}\n", message_type)
            self.llm_chat_text.insert(tk.END, f"{'='*50}\n\n", "danger_separator")
        elif message_type == "danger_system":
            # 為危險系統訊息添加空行和紅色格式
            self.llm_chat_text.insert(tk.END, f"\n[{timestamp}] {display_sender}: {message}\n", message_type)
        else:
            self.llm_chat_text.insert(tk.END, f"[{timestamp}] {display_sender}: {message}\n", message_type)
        
        # 配置文字顏色
        for msg_type, color in color_map.items():
            self.llm_chat_text.tag_configure(msg_type, foreground=color)
        
        # ===== 危險警報的特殊樣式 =====
        self.llm_chat_text.tag_configure("danger", foreground="#FF4500", font=('Arial', 10, 'bold'))
        self.llm_chat_text.tag_configure("danger_separator", foreground="#FF4500")
        
        # 自動滾動到底部
        self.llm_chat_text.see(tk.END)
        self.llm_chat_text.config(state=tk.DISABLED)
        
        # 儲存到對話歷史
        self.llm_messages.append({
            'timestamp': timestamp,
            'sender': sender,
            'message': message,
            'type': message_type
        })

    def apply_message_styles(self):
        """重新應用訊息樣式"""
        color_map = {
            "user": "blue",
            "assistant": "green", 
            "system": "gray",
            "processing": "orange",
            "error": "red",
            "danger": "#FF4500",
            "danger_system": "red"  # 新增
        }
        
        # 重新配置所有標籤樣式
        for msg_type, color in color_map.items():
            self.llm_chat_text.tag_configure(msg_type, foreground=color)
        
        self.llm_chat_text.tag_configure("danger", foreground="#FF4500", font=('Arial', 10, 'bold'))
        self.llm_chat_text.tag_configure("danger_separator", foreground="#FF4500")
        self.llm_chat_text.tag_configure("danger_system", foreground="red", font=('Arial', 9, 'bold'))  # 新增

    # ===== 更新狀態顯示方法 =====
    def update_connection_status(self, connected=False):
        """更新連接狀態顯示"""
        if connected:
            self.connection_status.set("Connected")
            # 可以在這裡添加更改標籤顏色的邏輯
            self.log("Successfully connected to MOT system")
        else:
            self.connection_status.set("Disconnected")
            self.log("Connection to MOT system lost")

    def get_llm_conversation_history(self):
        """獲取LLM對話歷史"""
        return self.llm_messages.copy()
    
    def clear_llm_conversation(self):
        """清空LLM對話歷史"""
        self.llm_messages.clear()
        self.llm_chat_text.config(state=tk.NORMAL)
        self.llm_chat_text.delete(1.0, tk.END)
        self.llm_chat_text.config(state=tk.DISABLED)
        self.add_llm_message("System", "Conversation history cleared.", "system")
    
    def clear_log(self):
        """清空系統日誌"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.log("System log cleared.")
    
    def get_display_options(self):
        """獲取當前的顯示選項設定"""
        return self.shared_display_options.copy()
        
    def browse_video(self):
        """打開文件選擇對話框選擇視頻文件"""
        from tkinter import filedialog
        file_path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=(
                ("Video Files", "*.mp4 *.avi *.mov *.mkv"),
                ("All Files", "*.*")
            )
        )
        if file_path:
            self.video_path.set(file_path)
            self.log("Video file selected: " + file_path)
    
    def log(self, message):
        """在日誌區域添加消息"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{time.strftime('%H:%M:%S')} - {message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        
    def update_image(self, frame):
        """更新GUI中的影像"""
        try:
            if frame is not None:
                if not isinstance(frame, np.ndarray):
                    self.status_var.set("Error: Invalid image format received")
                    return
                
                h, w = frame.shape[:2]
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:
                    scale = min(canvas_width/w, canvas_height/h)
                    new_w, new_h = int(w*scale), int(h*scale)
                    frame_resized = cv2.resize(frame, (new_w, new_h))
                    
                    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    imgtk = ImageTk.PhotoImage(image=img)
                    
                    self.canvas.configure(image=imgtk)
                    self.canvas.image = imgtk
            
            self.root.update()
            
        except Exception as e:
            self.status_var.set(f"Image display error: {str(e)}")
    
    def toggle_process(self):
        """切換處理過程（開始/停止）"""
        if not self.running:
            self.start_process()
        else:
            self.stop_process()
    
    def start_process(self):
        """開始處理過程"""
        if self.process_thread and self.process_thread.is_alive():
            messagebox.showwarning("Warning", "Processing is already running!")
            return
        
        try:
            self.stop_event.clear()
            self.running = True
            self.start_button.config(text="Stop")
            self.status_var.set(f"Processing window: {self.window_name.get()}")
            
            self.process_thread = threading.Thread(target=self.run_demo)
            self.process_thread.daemon = True
            self.process_thread.start()
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Error starting processing: {str(e)}")
            self.running = False
            self.start_button.config(text="Start")
    
    def stop_process(self):
        """停止處理過程"""
        if self.running:
            self.stop_event.set()
            self.status_var.set("Stopping processing...")
            self.start_button.config(text="Start")
            self.running = False
            
            if self.process_thread and self.process_thread.is_alive():
                self.process_thread.join(timeout=5.0)
                if self.process_thread.is_alive():
                    self.status_var.set("Warning: Processing thread could not be terminated normally")
                else:
                    self.status_var.set("Ready")
    
    def run_demo(self):
        """運行螢幕流處理Demo"""
        try:
            config_path = self.config_path.get().strip()
            model_path = self.model_path.get().strip()
            
            self.log(f"Config file path: {config_path}")
            self.log(f"Model file path: {model_path}")
            
            if not config_path:
                config_path = "UFLD/configs/culane_res18.py"
                self.config_path.set(config_path)
                self.log(f"Using default config file: {config_path}")
                
            if not model_path:
                model_path = "UFLD/weights/culane_res18.pth"
                self.model_path.set(model_path)
                self.log(f"Using default model file: {model_path}")
            
            args_list = [
                '--config', config_path,
                '--test_model', model_path,
                '--save_result',
                '--dataset', 'CULane',
                '--backbone', '18',
                '--griding_num', '200',
                '--use_aux', 'True',
                '--num_row', '72',
                '--num_col', '81',
                '--train_height', '288',
                '--train_width', '800'
            ]
            
            self.log(f"Command line arguments: {args_list}")
            
            try:
                args = make_parser().parse_args(args_list)
                self.log("Argument parsing successful")
            except Exception as parse_error:
                self.log(f"Argument parsing failed: {parse_error}")
                raise parse_error
            
            self.log(f"args.config = {getattr(args, 'config', 'None')}")
            self.log(f"args.test_model = {getattr(args, 'test_model', 'None')}")
            self.log(f"args.path = {getattr(args, 'path', 'None')}")
            self.log(f"args.window_name = {getattr(args, 'window_name', 'None')}")
            
            use_video = bool(self.video_path.get().strip())
            
            if use_video:
                video_path = self.video_path.get().strip()
                if not os.path.isfile(video_path):
                    raise ValueError(f"Video file not found: {video_path}")
                
                args.path = video_path
                args.demo = "video"
                args.window_name = os.path.basename(video_path)
                self.log(f"Using video file mode: {video_path}")
                self.status_var.set(f"Processing video: {os.path.basename(video_path)}")
            else:
                window_name = self.window_name.get().strip()
                if not window_name:
                    window_name = "pygame window"
                    self.window_name.set(window_name)
                
                args.window_name = window_name
                args.demo = "image"
                args.path = window_name
                self.log(f"Using window capture mode: {window_name}")
                self.status_var.set(f"Processing window: {window_name}")
            
            self.log(f"Final args.config = {args.config}")
            self.log(f"Final args.test_model = {args.test_model}")
            self.log(f"Final args.path = {args.path}")
            self.log(f"Final args.window_name = {args.window_name}")
            
            required_files = {
                'UFLD Config File': args.config,
                'UFLD Model File': args.test_model,
                'YOLOv8 Model': 'yolov8n.pt',
                'Social-LSTM Model': 'TrajectoryClassification/weights/2025-05-08/social-lstm.pth'
            }
            
            missing_files = []
            for name, path in required_files.items():
                if path is None:
                    missing_files.append(f"{name}: Path is None")
                    continue
                    
                full_path = os.path.join(current_dir, path) if not os.path.isabs(path) else path
                if not os.path.exists(full_path):
                    missing_files.append(f"{name}: {full_path}")
                    self.log(f"Warning: {name} not found: {full_path}")
                else:
                    self.log(f"Confirmed existence of {name}: {full_path}")
            
            if missing_files:
                error_msg = "The following required files are missing:\n" + "\n".join(missing_files)
                raise FileNotFoundError(error_msg)
            
            # 設置輸出目錄
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            vis_folder = os.path.join(current_dir, 'YOLO_track_outputs', timestamp)
            os.makedirs(vis_folder, exist_ok=True)
            current_time = time.localtime()
            
            self.log(f"Output directory: {vis_folder}")
            self.log("Preparing to call screenflow_demo...")
            
            # 傳遞共享顯示選項給 screenflow_demo
            initial_display_options = self.get_display_options()
            self.log(f"Initial display options: {initial_display_options}")
            
            # ===== 傳遞GUI回調處理器給MOT =====
            self.log("Setting up GUI callback handler...")
            
            # 執行螢幕流處理，傳入回調函數來更新GUI
            screenflow_demo(
                vis_folder=vis_folder,
                current_time=current_time,
                args=args,
                frame_callback=self.update_image,
                stop_event=self.stop_event,
                display_options_source=self,  # 傳遞整個app對象，以便動態讀取顯示選項
                gui_callback=self.gui_callback_handler  
            )
            
            self.log("Processing completed")
            
            # 更新連接狀態
            self.root.after(0, lambda: self.update_connection_status(True))
            
        except Exception as e:
            error_msg = str(e)
            import traceback
            full_traceback = traceback.format_exc()
            self.log(f"Full error traceback:\n{full_traceback}")
            
            self.root.after(0, lambda error=error_msg: self.status_var.set(f"Error: {error}"))
            self.root.after(0, lambda error=error_msg: messagebox.showerror("Error", f"Error during processing:\n{error}"))
            self.root.after(0, lambda: self.update_connection_status(False))
        finally:
            self.root.after(0, lambda: self.start_button.config(text="Start"))
            self.root.after(0, lambda: self.status_var.set("Ready"))
            self.running = False
    
    def on_closing(self):
        """關閉應用程序時的清理操作"""
        if self.running:
            self.stop_process()
            time.sleep(1)  # 給線程一點時間來停止
        
        self.root.destroy()
        sys.exit(0)

if __name__ == "__main__":
    # 初始化前檢查必要的模組和文件
    try:
        # 檢查必要的目錄結構
        required_dirs = ['UFLD', 'TrajectoryClassification', 'autopilot_fun']
        for dir_name in required_dirs:
            if not os.path.isdir(os.path.join(current_dir, dir_name)):
                print(f"Warning: Required directory '{dir_name}' not found")
        
        # 檢查必要的文件
        required_files = [
            'yolov8n.pt',
            os.path.join('TrajectoryClassification', 'weights', '2025-05-08', 'social-lstm.pth')
        ]
        for file_path in required_files:
            full_path = os.path.join(current_dir, file_path)
            if not os.path.isfile(full_path):
                print(f"Warning: Required file '{full_path}' not found")
        
        # 嘗試預先導入必要的模組
        import torch
        import torchvision
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            
        # 啟動GUI
        root = tk.Tk()
        app = AutoDrivingApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Error during initialization: {str(e)}")
        messagebox.showerror("Initialization Error", f"Error during application initialization:\n{str(e)}")
        sys.exit(1)