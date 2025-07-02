import os
import pandas as pd

def append_to_csv(csv_path: str, image_name: str, values: dict) -> str:
    """
    将结果附加写入 CSV 文件。
    自动创建目录；重复图像名时覆盖旧值；字段不一致时自动补全。
    """
    row = {"Image": image_name}
    row.update(values)

    df_new = pd.DataFrame([row])
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)

        # ✅ 若新字段不存在于旧数据中，添加空列
        for col in df_new.columns:
            if col not in df_existing.columns:
                df_existing[col] = None

        # ✅ 若旧字段不存在于新数据中，添加空列
        for col in df_existing.columns:
            if col not in df_new.columns:
                df_new[col] = None

        # ✅ 去重（覆盖已有图像）
        df_existing = df_existing[df_existing["Image"] != image_name]
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)

        # ✅ 保证 Image 列在第一列
        columns = ["Image"] + [col for col in df_combined.columns if col != "Image"]
        df_combined = df_combined[columns]
    else:
        df_combined = df_new

    df_combined.to_csv(csv_path, index=False)
    return csv_path
