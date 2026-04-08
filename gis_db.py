import geopandas as gpd



def infer_schema_from_geodataframe(gdf: gpd.GeoDataFrame) -> dict:
    """
    從 GeoDataFrame 推斷 metadata_schema。
    """
    PANDAS_TO_SCHEMA = {
        "int64":         "integer",
        "float64":       "float",
        "bool":          "boolean",
        "datetime64[ns]":"datetime",
        "object":        "text",
        "geometry":      "geometry",
    }
    schema = {}
    for col, dtype in gdf.dtypes.items():
        if col == "foreign_key":
            continue
        dtype_str = str(dtype)
        schema[col] = PANDAS_TO_SCHEMA.get(dtype_str, "text")
    return schema