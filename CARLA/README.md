# CARLA 模擬器安裝與執行指南

## 安裝方式

可以參考以下網站進行安裝：

[CARLA 官網](https://carla.org/)


[CARLA 環境搭建](https://blog.csdn.net/Fengdf666/article/details/135902650?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522171886504016800226559370%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=171886504016800226559370&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-2-135902650-null-null.142^v100^pc_search_result_base8&utm_term=carla%20ros2)



## 執行步驟

完成安裝後，可以依照以下步驟啟動 CARLA 模擬器：

1. 啟動 CARLA 模擬器（選擇GPU驅動）：

```bash
./CarlaUE4.sh -prefernvidia
```

2. 執行車輛控制程式：

```bash
python CARLA_ego_control.py --sync --rolename hero
```


3. 執行危險車輛控制程式：

```bash
python CARLA_dang_control.py
```