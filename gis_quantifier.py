"""
gis_quantifier.py
=================
負責地理影像的讀取、色彩解析與遮罩結果的持久化。

職責範圍：
  - 處理含中文路徑的本地圖片讀取（繞過 cv2.imread() 的路徑限制）
  - 從地理圖片中萃取所有不重複的非白色 BGR 像素值，
    供 add_color_mapping_level1() 建立色彩對應表使用

設計原則：
  - 所有影像在記憶體中以 numpy.ndarray 傳遞，落地只在 save_mask_image() 發生
  - 白色背景的排除閾值為 BGR 各分量 < 254（而非 != 255），
    容許抗鋸齒或壓縮產生的接近白色像素一併被排除，避免雜訊進入色彩對應表

TODO（Phase 3 — 層級二）：
  - [ ] 實作 quantify_mask(out_img, color_map, polygon, crs_utm)：
        接收 png_geographic_mapping() 的回傳值，依 color_map 統計各類別
        像素數與面積（需先將 polygon 投影至 EPSG:3826 計算實際面積）

相依模組：
  cv2（OpenCV）, numpy, datetime, os
  logs_handle（統一 logger）
"""

import cv2
from datetime import datetime
import os
import numpy as np
from logs_handle import logger
from typing import Union
from file_utils import load_json_data, save_json_data

# 白色背景排除閾值：BGR 各分量需同時 >= 254 才視為白色背景被排除。
# 使用 < 254 而非 != 255，目的是將抗鋸齒、JPEG 壓縮產生的
# 接近白色像素（例如 253,255,255）一併排除，避免混入色彩對應表。
WHITE_THRESHOLD = 254

def is_nearly_white(image: Union[bytes, np.ndarray], white_threshold: int = WHITE_THRESHOLD) -> bool:
    """
    檢查影像內容是否幾乎全白。
    因為存檔開銷極小，最新版本不使用。
    Args:
        image: 影像來源，接受兩種格式：
            - bytes：WMS 下載的原始 binary 資料（fetch_wms_image 的輸入前）
            - np.ndarray：已解碼的影像矩陣，shape=(H, W, C) channel-last，
              例如 png_geographic_mapping 回傳的 out_img.transpose(1, 2, 0)
        white_threshold: 判定為白色的門檻值 (0-255)，預設 254。
    Returns:
        bool: True 表示幾乎全白，False 表示含有顯著內容。
    """
    if not image:
        return False

    # 依輸入型別取得影像矩陣
    if isinstance(image, bytes):
        if not image:
            return False
        img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            logger.error("is_nearly_white：bytes 影像解碼失敗，無法檢查顏色。")
            return False
    elif isinstance(image, np.ndarray):
        img = image
    else:
        logger.error(f"is_nearly_white：不支援的輸入型別 {type(image)}。")
        return False
    
    # 處理顏色通道
    if img.ndim == 3:
        channels = img.shape[2]
        if channels == 4:
            img_to_check = img[:, :, :3]  # 忽略 Alpha
        else:
            img_to_check = img
    else:
        img_to_check = img  # 灰階直接檢查

    mean_value = np.mean(img_to_check)    
    return mean_value >= white_threshold

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

def add_color_mapping_level1(
    illustrative_diagram_name: str,
    unique_colors: list,
    metadata_path: str
) -> None:
    """
    互動式 CLI 工具：為指定圖層的每種像素色彩輸入對應的業務意義，
    並將結果寫入 geographic_color_metadata.json。此為「層級一標籤」建立流程。

    使用情境：
      首次處理新的 WMS 圖層時，由開發者手動執行一次，建立該圖層的色彩對應表。
      建立完成後，量化流程（gis_quantifier.py）即可依此 JSON 進行像素統計。

    流程：
      1. 依 unique_colors 建立 30×30px 色塊並以 cv2.imshow() 顯示，
         方便對照目視識別每種顏色
      2. 載入 metadata_path 的 JSON；若該圖層名稱已存在則提示並中止，
         防止意外覆蓋已建立的對應表
      3. 互動式輸入資料類型（data_type）與每種色彩的對應值（color_map）
      4. 將新條目寫回 JSON 並關閉 cv2 視窗

    Args:
        illustrative_diagram_name (str): 圖層的唯一識別名稱，作為 JSON 的頂層 key，
            建議格式："{layer_name}[illustrative_diagram]"，
            例如 "soil_ph[illustrative_diagram]"。
            應先以 utils.process_string() 標準化（小寫、空格轉底線）後再傳入。
        unique_colors (list[tuple]): 圖片中所有不重複且非白色的 BGR 色彩列表，
            由 gis_quantifier.decode_png_color_value() 產生，
            格式為 [(B, G, R), ...]，各分量為 int。
        metadata_path (str): geographic_color_metadata.json 的檔案路徑。

    Returns:
        None。結果直接寫入 metadata_path 所指的 JSON 檔案。

    Side Effects:
        - 開啟 cv2 視窗（執行期間）
        - 修改 metadata_path 指向的 JSON 檔案
        - 透過 input() 阻塞等待使用者輸入

    Notes:
        - 此函式為一次性的手動維護工具，不應在自動化流程（排程）中呼叫
        - color_map 的 value 目前存為使用者輸入的原始字串；
          若需要數值型別（int/float/list），需在輸入後另行轉換
        - JSON key 格式為 "B,G,R"（逗號分隔字串），例如 "1,254,3"
        - 更新既有 JSON 時，舊版本會被重命名為
          geographic_color_metadata.{YYYYMMDD}.bak 保留備份（TODO 待實作）

    TODO（Phase 3）：
        - [ ] 考慮支援多值輸入（例如 soil_texture 的 value 為 [sand%, silt%, clay%] 列表），
              目前僅儲存單一字串，複雜結構需手動編輯 JSON
        - [ ] 若 unique_colors 數量超過一定閾值（例如 20 種），
              考慮分頁顯示色塊，避免橫向視窗過寬
    """
    if not unique_colors:
        logger.error("add_color_mapping_level1：unique_colors 為空，無法執行。")
        return

    # 1. 建立色塊橫向拼接影像並顯示
    block_size = 30
    color_blocks = [
        np.full((block_size, block_size, 3), (b, g, r), dtype=np.uint8)
        for b, g, r in unique_colors
    ]
    display_image = np.hstack(color_blocks)

    window_name = f'Mapping: {illustrative_diagram_name} Unique Colors (Non-White BGR order)'
    cv2.imshow(window_name, display_image)

    print("\n----------------------------------------------------")
    print(f"請參照視窗 [{window_name}] 中的顏色塊，進行後續輸入。")
    cv2.waitKey(100)  # 確保視窗彈出後再繼續

    def get_input_while_showing(prompt: str) -> str:
        """在等待 input() 時，保持 cv2 視窗活躍。"""
        cv2.waitKey(1)
        return input(prompt)

    # 2. 載入並檢查既有 Metadata，防止覆蓋
    metadata = load_json_data(metadata_path)
    if not metadata:
        metadata = {}

    existing_entry = metadata.get(illustrative_diagram_name)
    if existing_entry is not None:
        logger.warning(f"{illustrative_diagram_name} 的色彩對應資料已存在，請手動修改 JSON 或重新執行。")
        print("已存在的資料如下：")
        print(f"  data_type: {existing_entry.get('data_type')}")
        print(f"  color_to_value: {existing_entry.get('color_to_value')}")
        input("按 Enter 關閉視窗...")
        cv2.destroyAllWindows()
        return

    # 3. 互動式輸入 — 資料類型
    print(f"\n--- 開始為圖層 [{illustrative_diagram_name}] 輸入顏色對應資訊 ---")
    from utils import process_string
    data_type = process_string(
        get_input_while_showing("請輸入此地圖資料的類型 (例如: '分級', 'pH連續分段', '土質'): ")
    )
    print(f"您輸入的資料類型為: {data_type}")

    # 4. 互動式輸入 — 逐色對應
    color_map = {}
    for b, g, r in unique_colors:
        color_key = f"{b},{g},{r}"
        value = process_string(
            get_input_while_showing(f"請輸入 BGR {color_key} 代表的意義或等級: ")
        )
        color_map[color_key] = value
        print(f"  色彩 {color_key} → {value}")

    # 5. 寫回 JSON
    metadata[illustrative_diagram_name] = {
        "data_type": data_type,
        "color_to_value": color_map
    }

    saved_path = save_json_data(metadata, metadata_path)
    if saved_path:
        logger.info(f"色彩對應表已儲存至 {saved_path}")
    else:
        logger.error(f"儲存色彩對應表至 {metadata_path} 時發生錯誤。")

    # 6. 關閉視窗
    print("--- 顏色映射輸入完成，正在關閉視窗 ---")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # 手動維護工具的使用範例：為新圖層建立色彩對應表
    # metadata_path 改為新專案根目錄下的 geographic_color_metadata.json
    metadata_path = 'geographic_color_metadata.json'

    png_path = r"C:\Python\work\farmland_spatial_map\soil_survey\母岩性質.png"
    illustrative_diagram_name = "Parent Material Property(illustrative_diagram)"

    from utils import process_string
    illustrative_diagram_name = process_string(illustrative_diagram_name)
    png_path = os.path.normpath(png_path)

    try:
        if os.path.exists(png_path):
            png = load_image_with_chinese_path(png_path)
            logger.info(f"圖片形狀: {png.shape}")
            unique_colors = decode_png_color_value(png)
            if unique_colors is not None:
                add_color_mapping_level1(illustrative_diagram_name, unique_colors, metadata_path)
        else:
            logger.error(f"找不到檔案: {png_path}")
    except Exception as e:
        logger.error(e)
        