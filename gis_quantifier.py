import cv2
from datetime import datetime
import os
import numpy as np
from logs_handle import logger

def load_image_with_chinese_path(file_path):
    """
    處理包含中文路徑的圖片讀取問題。
    先使用 Python 內建 open 讀取檔案內容，再讓 cv2 從記憶體解碼。
    
    Args:
        file_path (str): 包含中文的圖片路徑。
    
    Returns:
        numpy.ndarray or None: 載入的圖片陣列，失敗則為 None。
    """
    
    # 檢查路徑是否存在，以避免更深層的錯誤
    if not os.path.exists(file_path):
        logger.error(f"錯誤: 檔案不存在於路徑: {file_path}")
        return None

    # 1. 使用 Python 內建的 open() 函式以二進制模式 ('rb') 讀取圖片檔案
    try:
        with open(file_path, 'rb') as f:
            # 讀取所有位元組資料
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
    except IOError as e:
        logger.error(f"錯誤: 使用 open() 讀取檔案失敗: {e}")
        return None
        
    # 2. 使用 cv2.imdecode() 從記憶體中的位元組陣列解碼圖片
    # cv2.IMREAD_COLOR (1) 表示讀取彩色圖片 (忽略透明度)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        logger.error("錯誤: cv2.imdecode 無法解碼圖片。請檢查圖片格式是否正確或檔案是否完整。")
    
    return image

# 定義純白色的 BGR 值
WHITE_BGR = (254, 254, 254)
def decode_png_color_value(png):
    """
    載入地理圖片，展示圖片中會用到的顏色與數值，並忽略純白色。
    """
    if png is None:
        logger.error("錯誤: 圖片載入失敗 (png 變數為 None)。")
        return None # 返回 None 表示失敗

    # 1. 遍歷圖片像素，收集所有不重複的 BGR 色彩值 (忽略白色)
    if len(png.shape) == 3: # 彩色圖片 (H, W, C)
        # 轉換為 tuple 並使用 set 找出不重複顏色
        unique_colors_set = set(map(tuple, png.reshape(-1, png.shape[2])))
        
        # 篩選掉白色 (255, 255, 255)
        unique_colors = [color for color in unique_colors_set if color != WHITE_BGR]
        unique_colors = [    (b, g, r) for (b, g, r) in unique_colors_set    if b < 254 or g < 254 or r < 254]
    elif len(png.shape) == 2: # 灰階圖片 (H, W)
        unique_colors_set = set(png.flatten())
        # 將灰階值轉換為 BGR (v, v, v) 並篩選掉白色
        unique_colors = [ 
            (int(v), int(v), int(v)) 
            for v in unique_colors_set 
            if (int(v), int(v), int(v)) != WHITE_BGR 
        ]
    else:
        logger.warning(f"警告: 圖片維度不符合預期 ({png.shape})。")
        return None
    
    unique_colors.sort()
    
    # 檢查白色是否被排除 (可選)
    if WHITE_BGR in unique_colors_set:
         logger.notice(f"注意: 圖片中包含背景色 {WHITE_BGR} (白色)，已自動忽略。")

    print(f"找到 {len(unique_colors)} 種不重複且非白色的色彩值 (BGR):")
    for color in unique_colors:
        print(f"  BGR: {color}")        
    
    # 返回找到的非白色顏色列表，供 add_color_mapping 使用
    return unique_colors

def save_mask_image(png, classification, country_name):
    """
    下載圖片並儲存到指定路徑。
    """
    today = datetime.now().strftime("%Y%m%d")
    path_str = f"{country_name}_{classification}_{today}.png"
    path = os.path.join(classification, path_str)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    """ 這裡需要讀取舊圖片，
    讀取classification\{country_name}_{classification}名稱的檔案
    如果有數個則讀取最後一個，並和現在的png做比對，如果圖形一樣則不存檔
    """

    with open(path, 'wb') as f:
        f.write(png)
    logger.notice(f"圖片已成功下載並儲存至: '{path}'")
    return True
