"""
gis_quantifier.py
=================
負責地理影像的讀取、色彩解析與遮罩結果的持久化。

職責範圍：
  - 處理含中文路徑的本地圖片讀取（繞過 cv2.imread() 的路徑限制）
  - 從地理圖片中萃取所有不重複的非白色 BGR 像素值，
    供 gis_db.add_color_mapping_level1() 建立色彩對應表使用
  - 將遮罩後的 PNG 落地至輸出目錄，並避免重複儲存相同內容

設計原則：
  - 所有影像在記憶體中以 numpy.ndarray 傳遞，落地只在 save_mask_image() 發生
  - 白色背景的排除閾值為 BGR 各分量 < 254（而非 != 255），
    容許抗鋸齒或壓縮產生的接近白色像素一併被排除，避免雜訊進入色彩對應表

TODO（Phase 3 — 層級二）：
  - [ ] 實作 quantify_mask(out_img, color_map, polygon, crs_utm)：
        接收 png_geographic_mapping() 的回傳值，依 color_map 統計各類別
        像素數與面積（需先將 polygon 投影至 EPSG:3826 計算實際面積）
  - [ ] save_mask_image() 補完重複檢查邏輯（見函式內 TODO）

相依模組：
  cv2（OpenCV）, numpy, datetime, os
  logs_handle（統一 logger）
"""

import cv2
from datetime import datetime
import os
import numpy as np
from logs_handle import logger

# 白色背景排除閾值：BGR 各分量需同時 >= 254 才視為白色背景被排除。
# 使用 < 254 而非 != 255，目的是將抗鋸齒、JPEG 壓縮產生的
# 接近白色像素（例如 253,255,255）一併排除，避免混入色彩對應表。
WHITE_THRESHOLD = 254


def load_image_with_chinese_path(file_path: str) -> "np.ndarray | None":
    """
    讀取含中文或特殊字元路徑的圖片，繞過 cv2.imread() 的路徑編碼限制。

    cv2.imread() 在 Windows 上對非 ASCII 路徑（中文、日文等）行為不一致，
    此函式改以 Python 內建 open() 讀取原始 bytes，再交由 cv2.imdecode() 解碼，
    完全在記憶體中完成，不受路徑編碼影響。

    Args:
        file_path (str): 圖片檔案路徑，支援含中文的絕對或相對路徑。
            支援格式：cv2 可解碼的所有格式（PNG、JPEG、BMP 等）。

    Returns:
        numpy.ndarray | None:
          - 成功：shape=(H, W, 3)，dtype=uint8，色彩順序為 BGR（cv2 預設）
          - 失敗（路徑不存在、IOError、解碼失敗）：None，並記錄 error log

    Notes:
        - cv2.IMREAD_COLOR 忽略 alpha channel；若來源圖片為 RGBA，
          透明區域會被填為黑色
        - 此函式主要用於讀取本地的 WMS 圖例說明圖（illustrative diagram），
          供 decode_png_color_value() 萃取色彩對應表使用；
          WMS 動態下載的影像已在記憶體中，直接使用 fetch_wms_image() 的回傳值即可，
          無需經過此函式
    """
    if not os.path.exists(file_path):
        logger.error(f"檔案不存在：{file_path}")
        return None

    try:
        with open(file_path, 'rb') as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
    except IOError as e:
        logger.error(f"讀取檔案失敗：{e}")
        return None

    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        logger.error(
            f"cv2.imdecode 無法解碼圖片，請確認格式是否正確或檔案是否完整：{file_path}"
        )

    return image


def decode_png_color_value(png: np.ndarray) -> "list[tuple] | None":
    """
    從地理圖片中萃取所有不重複的非白色 BGR 像素值，排序後回傳。

    此函式為 gis_db.add_color_mapping_level1() 的前置步驟，
    產出的色彩列表作為互動式對應表建立流程的輸入。

    白色排除規則：
        BGR 三個分量皆 >= WHITE_THRESHOLD（254）的像素視為白色背景，予以排除。
        閾值設為 254 而非 255，目的是將抗鋸齒或壓縮產生的接近白色雜訊一併過濾。

    Args:
        png (numpy.ndarray): 以 cv2.IMREAD_COLOR 讀取的 BGR 影像陣列，
            shape=(H, W, 3)（彩色）或 shape=(H, W)（灰階）。

    Returns:
        list[tuple] | None:
          - 成功：非白色 BGR 色彩的排序列表，格式為 [(B, G, R), ...]，
            各分量為 int，依 BGR tuple 值升序排列
          - 失敗（png 為 None 或維度不符）：None

    Notes:
        - 彩色影像：以 reshape(-1, 3) 展平後轉 set，時間複雜度 O(H×W)
        - 灰階影像：展平後轉 set，再以 (v, v, v) 格式統一表示，與彩色路徑一致
        - 回傳 list 而非 set，確保順序固定，方便 gis_db 中依序顯示色塊

    TODO（Phase 3）：
        - [ ] 若圖片像素數極大（例如全台灣尺寸），reshape + set 的記憶體佔用可能偏高，
              可評估改用 np.unique(png.reshape(-1, 3), axis=0) 取代 set 操作
    """
    if png is None:
        logger.error("圖片載入失敗（png 為 None）。")
        return None

    if len(png.shape) == 3:  # 彩色圖片 (H, W, C)
        unique_colors_set = set(map(tuple, png.reshape(-1, png.shape[2])))
        unique_colors = [
            (b, g, r) for (b, g, r) in unique_colors_set
            if b < WHITE_THRESHOLD or g < WHITE_THRESHOLD or r < WHITE_THRESHOLD
        ]
    elif len(png.shape) == 2:  # 灰階圖片 (H, W)
        unique_colors_set = set(png.flatten())
        unique_colors = [
            (int(v), int(v), int(v))
            for v in unique_colors_set
            if int(v) < WHITE_THRESHOLD
        ]
    else:
        logger.warning(f"圖片維度不符合預期：{png.shape}，應為 (H,W,3) 或 (H,W)。")
        return None

    unique_colors.sort()

    if WHITE_THRESHOLD <= 255 and any(
        b >= WHITE_THRESHOLD and g >= WHITE_THRESHOLD and r >= WHITE_THRESHOLD
        for (b, g, r) in set(map(tuple, png.reshape(-1, png.shape[2])))
        if len(png.shape) == 3
    ):
        logger.notice(f"圖片中含白色背景（BGR 各分量 >= {WHITE_THRESHOLD}），已自動排除。")

    logger.notice(f"找到 {len(unique_colors)} 種不重複且非白色的 BGR 色彩值。")
    for color in unique_colors:
        logger.notice(f"  BGR: {color}")

    return unique_colors