import json
import os
from logs_handle import logger
from utils import process_string
import re

# main_gis.py 或獨立建立 map_name_mapping.py
MAP_NAME_TO_EN_STATIC = {
    # === 農地類 ===
    "農地重要性等級": "farmland_importance_level",
    "農地生產力等級": "farmland_productivity_levels",

    # === 作物適栽性（統一後綴）===
    "水稻適栽性等級分布圖": "crop_suitability_rating_map_rice",
    "甘薯適栽性等級分布圖": "crop_suitability_rating_map_sweet_potato",
    "花生適栽性等級分布圖": "crop_suitability_rating_map_peanut",
    "大豆適栽性等級分布圖": "crop_suitability_rating_map_soybean",
    "玉米適栽性等級分布圖": "crop_suitability_rating_map_corn",
    "小麥適栽性等級分布圖": "crop_suitability_rating_map_wheat",
    "大麥適栽性等級分布圖": "crop_suitability_rating_map_barley",
    "高粱適栽性等級分布圖": "crop_suitability_rating_map_sorghum",
    "粟適栽性等級分布圖": "crop_suitability_rating_map_millet",
    "紅豆適栽性等級分布圖": "crop_suitability_rating_map_azuki_bean",
    "綠豆適栽性等級分布圖": "crop_suitability_rating_map_mung_bean",
    "胡麻適栽性等級分布圖": "crop_suitability_rating_map_sesame",
    "向日葵適栽性等級分布圖": "crop_suitability_rating_map_sunflower",
    "油菜適栽性等級分布圖": "crop_suitability_rating_map_rapeseed",
    "木薯適栽性等級分布圖": "crop_suitability_rating_map_cassava",          # 原「木薯栽性」已修正
    "甘蔗適栽性等級分布圖": "crop_suitability_rating_map_sugarcane",
    "茶葉適栽性等級分布圖": "crop_suitability_rating_map_tea",
    "菸草適栽性等級分布圖": "crop_suitability_rating_map_tobacco",
    "蠶桑適栽性等級分布圖": "crop_suitability_rating_map_mulberry",
    "亞麻(甘蔗)適栽性等級分布圖": "crop_suitability_rating_map_flax",      # 括號移除
    "苧麻適栽性等級分布圖": "crop_suitability_rating_map_ramie",
    "黃麻適栽性等級分布圖": "crop_suitability_rating_map_jute",
    "鐘麻適栽性等級分布圖": "crop_suitability_rating_map_kenaf",
    "瓊麻適栽性等級分布圖": "crop_suitability_rating_map_sisal",
    "藺草適栽性等級分布圖": "crop_suitability_rating_map_rush",
    "香蕉適栽性等級分布圖": "crop_suitability_rating_map_banana",
    "柑桔適栽性等級分布圖": "crop_suitability_rating_map_citrus",
    "鳳梨適栽性等級分布圖": "crop_suitability_rating_map_pineapple",
    "荔枝適栽性等級分布圖": "crop_suitability_rating_map_lychee",
    "龍眼適栽性等級分布圖": "crop_suitability_rating_map_longan",
    "番石榴適栽性等級分布圖": "crop_suitability_rating_map_guava",
    "枇杷適栽性等級分布圖": "crop_suitability_rating_map_loquat",
    "橄欖適栽性等級分布圖": "crop_suitability_rating_map_olive",
    "番荔枝適栽性等級分布圖": "crop_suitability_rating_map_sugar_apple",
    "咖啡適栽性等級分布圖": "crop_suitability_rating_map_coffee",
    "人心果適栽性等級分布圖": "crop_suitability_rating_map_sapodilla",
    "澳洲胡桃適栽性等級分布圖": "crop_suitability_rating_map_macadamia",
    "馬拉巴栗適栽性等級分布圖": "crop_suitability_rating_map_malabar_chestnut",
    "百香果適栽性等級分布圖": "crop_suitability_rating_map_passion_fruit",
    "蓮霧適栽性等級分布圖": "crop_suitability_rating_map_wax_apple",
    "芒果適栽性等級分布圖": "crop_suitability_rating_map_mango",
    "木瓜適栽性等級分布圖": "crop_suitability_rating_map_papaya",
    "可可椰子適栽性等級分布圖": "crop_suitability_rating_map_coconut",
    "檳榔適栽性等級分布圖": "crop_suitability_rating_map_betel_nut",
    "楊桃適栽性等級分布圖": "crop_suitability_rating_map_star_fruit",
    "可可適栽性等級分布圖": "crop_suitability_rating_map_cocoa",
    "酪梨適栽性等級分布圖": "crop_suitability_rating_map_avocado",
    "介壽果(腰果)適栽性等級分布圖": "crop_suitability_rating_map_cashew",
    "印度棗適栽性等級分布圖": "crop_suitability_rating_map_indian_jujube",
    "西印度櫻桃適栽性等級分布圖": "crop_suitability_rating_map_west_indian_cherry",
    "楊梅適栽性等級分布圖": "crop_suitability_rating_map_chinese_bayberry",
    "李適栽性等級分布圖": "crop_suitability_rating_map_plum",
    "梅適栽性等級分布圖": "crop_suitability_rating_map_apricot",
    "桃適栽性等級分布圖": "crop_suitability_rating_map_peach",
    "梨適栽性等級分布圖": "crop_suitability_rating_map_pear",
    "蘋果適栽性等級分布圖": "crop_suitability_rating_map_apple",
    "葡萄適栽性等級分布圖": "crop_suitability_rating_map_grape",
    "柿適栽性等級分布圖": "crop_suitability_rating_map_persimmon",
    "栗適栽性等級分布圖": "crop_suitability_rating_map_chestnut",
    "草莓適栽性等級分布圖": "crop_suitability_rating_map_strawberry",
    # 以下根莖、葉菜、瓜果等可依需求繼續補充（已涵蓋多數常見者）
    "蘿蔔適栽性等級分布圖": "crop_suitability_rating_map_radish",
    # ...（其餘可依相同規則自動生成）

    # === 土壤調查類 ===
    "母岩性質": "soil_survey_parent_material",
    "土壤石灰性及鹽鹼性": "soil_survey_lime_and_salinity",
    "土壤特性": "soil_survey_soil_properties",
    "土壤排水": "soil_survey_drainage",
    "土壤型態及形成方式": "soil_survey_soil_type_and_formation",
    "土壤酸鹼值": "soil_survey_ph",
    "土壤質地(0-30公分)": "soil_survey_texture_0_30cm",
    "土壤質地(30-60公分)": "soil_survey_texture_30_60cm",
    "土壤質地(60-90公分)": "soil_survey_texture_60_90cm",
    "土壤質地(90-150公分)": "soil_survey_texture_90_150cm",
    "土壤坡度": "soil_survey_slope",
}

OVERRIDE_FILE = "geographic_map_name_overrides.json"

def load_map_name_mapping() -> dict:
    """載入靜態 + 使用者動態補充的對照表"""
    mapping = MAP_NAME_TO_EN_STATIC.copy()
    
    if os.path.exists(OVERRIDE_FILE):
        try:
            with open(OVERRIDE_FILE, "r", encoding="utf-8") as f:
                overrides = json.load(f)
                mapping.update(overrides)
            logger.notice(f"已載入 {len(overrides)} 筆動態翻譯覆蓋")
        except Exception as e:
            logger.error(f"載入 overrides.json 失敗: {e}")
    
    return mapping


def save_map_name_override(chinese_name: str, english_name: str) -> bool:
    """將新翻譯存入 overrides.json"""
    overrides = {}
    if os.path.exists(OVERRIDE_FILE):
        try:
            with open(OVERRIDE_FILE, "r", encoding="utf-8") as f:
                overrides = json.load(f)
        except:
            pass
    
    overrides[chinese_name] = english_name
    
    try:
        with open(OVERRIDE_FILE, "w", encoding="utf-8") as f:
            json.dump(overrides, f, ensure_ascii=False, indent=2)
        logger.logs(f"已新增翻譯並存檔：{chinese_name} → {english_name}")
        return True
    except Exception as e:
        logger.error(f"存檔失敗: {e}")
        return False

def get_or_create_map_name_en(chinese_name: str) -> str:
    """
    取得或互動式建立中文圖層名稱的英文對應。
    這是 Phase 3 非常重要的輔助函式。
    
    處理流程：
    1. 先查靜態 + overrides
    2. 若找不到 → 判斷類型並引導使用者輸入
    3. 存入 overrides.json
    """

    mapping = load_map_name_mapping()
    
    if chinese_name in mapping:
        return mapping[chinese_name]
    
    logger.warning(f"未找到翻譯：'{chinese_name}'")
    print(f"\n=== 發現新圖層名稱：{chinese_name} ===")
    
    # 自動判斷類型
    is_crop = "適栽性等級分布圖" in chinese_name
    suggested_type = "crop" if is_crop else None
    
    if not is_crop:
        print("這看起來不是作物適栽性圖層。")
        choice = input("是否為土壤調查類別？(y/n): ").strip().lower()
        if choice == 'y':
            suggested_type = "soil_survey"
        else:
            suggested_type = input("請輸入類型 (crop / soil / other): ").strip().lower()
    
    # 讓使用者輸入核心英文名稱
    if suggested_type == "crop":
        default = chinese_name.replace("適栽性等級分布圖", "").strip()
        print(f"建議作物名稱：{default}")
        
        while True:
            crop_en = input(f"請輸入作物英文名稱 (僅允許小寫英文、數字、底線): ").strip()
            crop_en = process_string(crop_en)   # 先做標準化（去除空格、轉小寫等）
            
            # 正則檢查：只能包含小寫英文字母、數字、底線
            if not re.match(r'^[a-z0-9_]+$', crop_en):
                print("❌ 格式錯誤！只能使用小寫英文、數字和底線 (例如: rice、sweet_potato、corn_2025)")
                continue
            if not crop_en:
                print("❌ 請輸入英文名稱")
                continue
                
            print(f"✅ 接受：{crop_en}")
            break
            
        english_full = f"crop_suitability_rating_map_{crop_en}"

    elif suggested_type == "soil_survey":
        default = chinese_name.replace("土壤", "").strip()
        print(f"建議土壤類型：{default}")
        
        while True:
            soil_en = input(f"請輸入土壤類型英文 (僅允許小寫英文、數字、底線): ").strip()
            soil_en = process_string(soil_en)
            
            # 正則檢查
            if not re.match(r'^[a-z0-9_]+$', soil_en):
                print("❌ 格式錯誤！只能使用小寫英文、數字和底線 (例如: ph、parent_material、texture_0_30cm)")
                continue
            if not soil_en:
                print("❌ 請輸入英文名稱")
                continue
                
            print(f"✅ 接受：{soil_en}")
            break
            
        english_full = f"soil_survey_{soil_en}"
        
    else:
        english_full = input("請手動輸入完整英文分類名稱: ").strip()
    
    # 確認
    print(f"\n將建立：{chinese_name} → {english_full}")
    if input("確認正確？(y/n): ").strip().lower() != 'y':
        logger.logs(f"使用者取消建立圖層英文名稱對應： <{chinese_name}  →  {english_full}> 。")
        return None
    
    # 存檔
    save_map_name_override(chinese_name, english_full)
    
    return english_full