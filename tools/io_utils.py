import os
import pandas as pd

def append_to_csv(csv_path: str, image_name: str, values: dict) -> str:
    """
    将结果附加写入 CSV 文件。
    自动创建目录；重复图像名时覆盖旧值。
    """
    row = {"Image": image_name}
    row.update(values)

    df_new = pd.DataFrame([row])

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_existing = df_existing[df_existing["Image"] != image_name]  # 去重
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_csv(csv_path, index=False)
    return csv_path
