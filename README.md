# FastSlowDrive

This project is an integrated intelligent self-driving system with the following capabilities:  
- Vehicle and pedestrian detection & tracking (YOLOv10 + ByteTrack)  
- Lane detection & offset control (UFLD_v2 + PID control)  
- Hazardous behavior recognition (Social-LSTM trajectory analysis + ROI warning zones)  
- Semantic driving suggestion generation (OpenAI GPT-4o)  
- Support for scenario testing on the CARLA simulator  

---

## 📌 System Architecture
<p align="center">
  <img src="pic/流程圖v5.jpg" alt="System Architecture" width="75%">
</p>

The system is divided into two main subsystems: **perception/control** and **semantic reasoning**.

- **System 1: Perception and Basic Control Module**  
  - Receives real-time images from the simulation environment and performs:  
    - `Object Detection` + `Multiple Object Tracking` (YOLOv10 + ByteTrack)  
    - `Lane Detection` (UFLD_v2)  
    - `Behavior Classification` (Social-LSTM trajectory analysis + hazard zone estimation)  
  - If no potential risks are detected, the system outputs `Basic Control Code` to execute routine driving tasks.  

- **System 2: Semantic Reasoning and Advanced Control Module**  
  - Activated when risky behavior is detected or when a user command is received.  
  - The `LLM` (Large Language Model) generates semantic risk assessments and driving suggestions, which are displayed via the `UI` and converted into `Advanced Control Code` to control the vehicle in simulation.  

> This architecture combines rule-based and LLM-based control strategies, ensuring safety while introducing semantic understanding capabilities.  

---

## 📌 Workflow
<p align="center">
  <img src="pic/flow_chart_v2.jpg" alt="Workflow" width="75%">
</p>

1. **Input images** are processed by:  
   - `MOT` (Multi-Object Tracking) and `Behavior Classification`  
   - `Lane Detection`  

2. The system checks for:  
   - Hazardous behaviors (via trajectory classification)  
   - Or user-issued commands  

3. If **Yes**:  
   - `LLM Suggestions` are triggered to provide semantic analysis and control recommendations  
   - Control strategy switches to `Advanced Control Code`  

4. If **No**:  
   - The system executes `Basic Control Code`  

5. All control logic is ultimately sent to the simulator to perform driving operations.  

> This workflow balances **real-time responsiveness** with **semantic reasoning**, showcasing the potential of LLM-assisted decision-making in risk scenarios.  

---

## 🔧 Project Structure

| File / Folder        | Description |
|----------------------|-------------|
| `main.py`            | Main entry: integrates perception, tracking, control, and LLM trigger |
| `gui_app.py`         | Interactive UI (optional) |
| `autopilot_fun/perception.py`     | Perception module: multi-object tracking + lane detection |
| `autopilot_fun/utils.py`          | Utility functions: screen capture, hazard zone estimation, timers, etc. |
| `autopilot_fun/visualization.py`  | Visualization: trajectory rendering, lane lines, hazard zones |
| `autopilot_fun/control.py`        | Control module: PID controller, obstacle avoidance, lane keeping |
| `autopilot_fun/integration.py`    | GPT integration: prompt design, API calls, and response handling |
| `TrajectoryClassification/social_lstm_trainer.py` | Behavior classification: Social-LSTM trajectory analysis |
| `UFLD`               | Lane detection module |  

---

## ⚙️ Requirements

- Python >= 3.7  
- [YOLOv10](https://github.com/ultralytics/ultralytics)  
- [UFLD_v2 Lane Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2)  
- OpenAI API Key  
- [CARLA Simulator](https://carla.org/) version 0.9.13  
- Other dependencies listed in `requirements.txt`  

---

## 🚀 Usage

### 1. Start CARLA Simulator
Refer to the detailed setup guide here: [CARLA README](CARLA/README.md)  

### 2. Run the main program

#### 🖥️ GUI Mode
```bash
python gui_app.py
```

#### ⚡ Command Line Mode
```bash
python main.py --config UFLD/configs/culane_res18.py --test_model UFLD/weights/culane_res18.pth --save_result --window_name "pygame"
```

## 📊 Experimental Results

<table>
  <tr>
    <td align="center">
      <img src="pic/system1.gif" alt="system1" width="300">
      <br>
      <b>system1</b>
    </td>
    <td align="center">
      <img src="pic/system1+2.gif" alt="system1+2" width="300">
      <br>
      <b>system1+2</b>
    </td>
  </tr>
</table>

