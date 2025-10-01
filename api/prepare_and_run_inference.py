#!/usr/bin/env python3
"""
Скрипт для подготовки DICOM файлов и запуска inference через API.

Берёт распакованные DICOM файлы, группирует их по StudyInstanceUID,
запаковывает в ZIP архивы и прогоняет через API.
"""

import argparse
import os
import sys
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import pydicom
from tqdm import tqdm


def group_dicom_files_by_study(input_dir: Path) -> dict:
    """Группирует DICOM файлы по StudyInstanceUID."""
    print(f"\n📂 Сканируем директорию: {input_dir}")

    studies = {}
    skipped = 0

    all_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.startswith(".") or file.endswith((".txt", ".pdf", ".jpg", ".png")):
                continue
            all_files.append(Path(root) / file)

    print(f"Найдено {len(all_files)} файлов")

    for file_path in tqdm(all_files, desc="Чтение DICOM файлов"):
        try:
            dcm = pydicom.dcmread(file_path, stop_before_pixels=True)

            study_uid = str(dcm.StudyInstanceUID)

            path_str = str(file_path)
            if "norma" in path_str.lower():
                category = "norma"
            elif "pneumonia" in path_str.lower():
                category = "pneumonia"
            elif "pneumotorax" in path_str.lower():
                category = "pneumotorax"
            else:
                category = "unknown"

            if study_uid not in studies:
                studies[study_uid] = {"files": [], "category": category}

            studies[study_uid]["files"].append(file_path)

        except Exception as e:
            skipped += 1
            continue

    print(f"\n✅ Сгруппировано {len(studies)} исследований")
    print(f"⚠️  Пропущено {skipped} файлов (не DICOM)")

    categories = {}
    for study_info in studies.values():
        cat = study_info["category"]
        categories[cat] = categories.get(cat, 0) + 1

    print("\nРаспределение по категориям:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    return studies


def create_zip_archives(studies: dict, output_dir: Path) -> list:
    """Создаёт ZIP архивы для каждого исследования."""
    output_dir.mkdir(parents=True, exist_ok=True)
    archives = []

    print(f"\n📦 Создаём ZIP архивы в: {output_dir}")

    for study_uid, study_info in tqdm(studies.items(), desc="Создание архивов"):
        category = study_info["category"]
        files = study_info["files"]

        safe_uid = study_uid.replace(".", "_")
        zip_name = f"{category}_{safe_uid}.zip"
        zip_path = output_dir / zip_name

        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for file_path in files:
                    zf.write(file_path, file_path.name)

            archives.append(
                {"path": zip_path, "study_uid": study_uid, "category": category, "num_files": len(files)}
            )

        except Exception as e:
            print(f"❌ Ошибка создания архива {zip_name}: {e}")
            continue

    print(f"✅ Создано {len(archives)} архивов")
    return archives


def run_batch_inference(archives: list, api_url: str, output_excel: Path):
    """Запускает batch inference через API."""
    import requests

    print(f"\n🚀 Запускаем batch inference...")
    print(f"API URL: {api_url}")

    try:
        response = requests.get(f"{api_url}/health", timeout=10)
        if response.status_code == 200:
            print("✅ API доступен")
        else:
            print(f"❌ API недоступен (HTTP {response.status_code})")
            return
    except Exception as e:
        print(f"❌ Ошибка подключения к API: {e}")
        return

    results = []

    for archive_info in tqdm(archives, desc="Обработка архивов"):
        archive_path = archive_info["path"]

        try:
            with open(archive_path, "rb") as f:
                files = {"file": (archive_path.name, f, "application/zip")}
                response = requests.post(f"{api_url}/predict", files=files, timeout=600)

            if response.status_code == 200:
                result = response.json()
                result["path_to_study"] = str(archive_path)
                result["ground_truth_category"] = archive_info["category"]
                result["num_files"] = archive_info["num_files"]
                results.append(result)

                status = "✅" if result["pathology"] == "Патология" else "⚪"
                tqdm.write(
                    f"{status} {archive_path.name}: {result['pathology']} "
                    f"(P={result['probability_of_pathology']:.3f}, GT={archive_info['category']})"
                )

            else:
                error_data = response.json() if response.headers.get("content-type") == "application/json" else {}
                results.append(
                    {
                        "path_to_study": str(archive_path),
                        "study_uid": archive_info["study_uid"],
                        "series_uid": "",
                        "probability_of_pathology": 0.0,
                        "pathology": "Unknown",
                        "most_dangerous_pathology_type": "Unknown",
                        "processing_status": "Failure",
                        "time_of_processing": 0.0,
                        "ground_truth_category": archive_info["category"],
                        "num_files": archive_info["num_files"],
                        "error": error_data.get("error", f"HTTP {response.status_code}"),
                    }
                )

        except Exception as e:
            results.append(
                {
                    "path_to_study": str(archive_path),
                    "study_uid": archive_info["study_uid"],
                    "series_uid": "",
                    "probability_of_pathology": 0.0,
                    "pathology": "Unknown",
                    "most_dangerous_pathology_type": "Unknown",
                    "processing_status": "Failure",
                    "time_of_processing": 0.0,
                    "ground_truth_category": archive_info["category"],
                    "num_files": archive_info["num_files"],
                    "error": str(e),
                }
            )

    df = pd.DataFrame(results)

    columns_order = [
        "path_to_study",
        "ground_truth_category",
        "study_uid",
        "series_uid",
        "probability_of_pathology",
        "pathology",
        "most_dangerous_pathology_type",
        "processing_status",
        "time_of_processing",
        "num_files",
    ]
    columns_order = [col for col in columns_order if col in df.columns]
    other_cols = [col for col in df.columns if col not in columns_order]
    df = df[columns_order + other_cols]

    df.to_excel(output_excel, index=False, engine="openpyxl")
    print(f"\n💾 Результаты сохранены: {output_excel}")

    print("\n" + "=" * 80)
    print("СТАТИСТИКА")
    print("=" * 80)

    total = len(results)
    success = sum(1 for r in results if r["processing_status"] == "Success")
    print(f"Всего: {total}")
    print(f"✅ Успешно: {success} ({success/total*100:.1f}%)")
    print(f"❌ Ошибки: {total - success} ({(total-success)/total*100:.1f}%)")

    if success > 0:
        success_df = df[df["processing_status"] == "Success"]

        print("\nПо ground truth категориям:")
        for cat in ["norma", "pneumonia", "pneumotorax"]:
            cat_df = success_df[success_df["ground_truth_category"] == cat]
            if len(cat_df) > 0:
                pathology_count = (cat_df["pathology"] == "Патология").sum()
                print(f"  {cat}: {len(cat_df)} исследований, {pathology_count} помечено как патология")

        if "most_dangerous_pathology_type" in success_df.columns:
            top_pathologies = success_df["most_dangerous_pathology_type"].value_counts().head(5)
            print("\nТоп-5 наиболее частых опасных патологий:")
            for i, (path, count) in enumerate(top_pathologies.items(), 1):
                print(f"  {i}. {path}: {count}")


def main():
    parser = argparse.ArgumentParser(description="Подготовка данных и запуск inference")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Путь к директории с распакованными DICOM файлами",
    )
    parser.add_argument(
        "--temp-dir", type=str, default=None, help="Директория для временных ZIP архивов (по умолчанию: /tmp/...)"
    )
    parser.add_argument("--api-url", type=str, default="http://localhost:8000", help="URL API сервиса")
    parser.add_argument("--output", type=str, required=True, help="Путь для сохранения Excel файла с результатами")
    parser.add_argument(
        "--skip-zip-creation",
        action="store_true",
        help="Пропустить создание ZIP архивов (использовать существующие)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"❌ Директория не найдена: {input_dir}")
        return 1

    if args.temp_dir:
        temp_dir = Path(args.temp_dir)
    else:
        temp_dir = Path(tempfile.gettempdir()) / "ct_clip_inference_archives"

    output_excel = Path(args.output)

    print("=" * 80)
    print("CT-CLIP + LightGBM Inference Pipeline")
    print("=" * 80)

    if not args.skip_zip_creation:
        studies = group_dicom_files_by_study(input_dir)
        archives = create_zip_archives(studies, temp_dir)
    else:
        print(f"\n📦 Используем существующие архивы из: {temp_dir}")
        archives = []
        for zip_file in temp_dir.glob("*.zip"):
            parts = zip_file.stem.split("_", 1)
            category = parts[0] if len(parts) > 1 else "unknown"
            study_uid = parts[1].replace("_", ".") if len(parts) > 1 else zip_file.stem

            archives.append(
                {"path": zip_file, "study_uid": study_uid, "category": category, "num_files": 0}
            )
        print(f"Найдено {len(archives)} архивов")

    if not archives:
        print("❌ Нет архивов для обработки!")
        return 1

    run_batch_inference(archives, args.api_url, output_excel)

    print("\n✅ Готово!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
