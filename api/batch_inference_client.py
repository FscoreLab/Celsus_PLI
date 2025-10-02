#!/usr/bin/env python3
"""
Batch Inference Client для CT-CLIP + LightGBM API

Обрабатывает папку с ZIP архивами DICOM файлов и создает Excel отчет.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd
import requests
from tqdm import tqdm


def find_zip_archives(input_dir: Path) -> List[Path]:
    """Находит все ZIP архивы в указанной директории."""
    zip_files = list(input_dir.glob("*.zip"))
    return sorted(zip_files)


def process_single_archive(api_url: str, archive_path: Path) -> dict:
    """Отправляет один архив в API и возвращает результат."""
    try:
        start_time = time.time()

        with open(archive_path, "rb") as f:
            files = {"file": (archive_path.name, f, "application/zip")}
            response = requests.post(f"{api_url}/predict", files=files, timeout=600)  # 10 минут

        processing_time = time.time() - start_time

        if response.status_code == 200:
            result = response.json()
            # Преобразуем top_pathologies в строку для most_dangerous_pathology_type
            if "top_pathologies" in result:
                result["most_dangerous_pathology_type"] = (
                    ", ".join(result["top_pathologies"]) if result["top_pathologies"] else ""
                )
            else:
                result["most_dangerous_pathology_type"] = ""
            result["path_to_study"] = str(archive_path.absolute())
            result["processing_time_total"] = processing_time
            return result
        else:
            error_data = response.json() if response.headers.get("content-type") == "application/json" else {}
            return {
                "path_to_study": str(archive_path.absolute()),
                "study_uid": "",
                "series_uid": "",
                "probability_of_pathology": 0.0,
                "pathology": "Unknown",
                "most_dangerous_pathology_type": "Unknown",
                "processing_status": "Failure",
                "time_of_processing": processing_time,
                "error": error_data.get("error", f"HTTP {response.status_code}"),
            }

    except requests.exceptions.Timeout:
        return {
            "path_to_study": str(archive_path.absolute()),
            "study_uid": "",
            "series_uid": "",
            "probability_of_pathology": 0.0,
            "pathology": "Unknown",
            "most_dangerous_pathology_type": "Unknown",
            "processing_status": "Failure",
            "time_of_processing": 0.0,
            "error": "Request timeout (> 10 minutes)",
        }
    except Exception as e:
        return {
            "path_to_study": str(archive_path.absolute()),
            "study_uid": "",
            "series_uid": "",
            "probability_of_pathology": 0.0,
            "pathology": "Unknown",
            "most_dangerous_pathology_type": "Unknown",
            "processing_status": "Failure",
            "time_of_processing": 0.0,
            "error": str(e),
        }


def batch_process(api_url: str, input_dir: Path, output_excel: Path):
    """Обрабатывает все архивы в папке и создает Excel отчет."""
    print("=" * 80)
    print("CT-CLIP + LightGBM Batch Inference Client")
    print("=" * 80)

    print(f"\n🔍 Проверяем доступность API: {api_url}")
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

    print(f"\n📂 Поиск ZIP архивов в: {input_dir}")
    archives = find_zip_archives(input_dir)
    print(f"✅ Найдено {len(archives)} архивов")

    if not archives:
        print("❌ Архивы не найдены!")
        return

    results = []
    print("\n🚀 Начинаем обработку...")

    for archive_path in tqdm(archives, desc="Обработка архивов"):
        result = process_single_archive(api_url, archive_path)
        results.append(result)

        status_emoji = "✅" if result["processing_status"] == "Success" else "❌"
        tqdm.write(
            f"{status_emoji} {archive_path.name}: {result['pathology']} "
            f"(P={result['probability_of_pathology']:.3f}, "
            f"{result['time_of_processing']:.1f}s)"
        )

    df_results = pd.DataFrame(results)

    # Базовые колонки, которые должны быть всегда
    base_columns = [
        "path_to_study",
        "study_uid",
        "series_uid",
        "probability_of_pathology",
        "pathology",
        "processing_status",
        "time_of_processing",
    ]

    # Опциональные колонки
    optional_columns = [
        "most_dangerous_pathology_type",
        "error",
    ]

    # Собираем только те колонки, которые есть в данных
    final_columns = [col for col in base_columns if col in df_results.columns]
    for col in optional_columns:
        if col in df_results.columns:
            final_columns.append(col)

    df_final = df_results[final_columns].copy()

    print(f"\n💾 Сохраняем результаты в: {output_excel}")
    df_final.to_excel(output_excel, index=False, engine="openpyxl")

    print("\n" + "=" * 80)
    print("СТАТИСТИКА ОБРАБОТКИ")
    print("=" * 80)

    total = len(results)
    success = sum(1 for r in results if r["processing_status"] == "Success")
    failure = total - success

    print(f"Всего архивов: {total}")
    print(f"✅ Успешно: {success} ({success/total*100:.1f}%)")
    print(f"❌ Ошибки: {failure} ({failure/total*100:.1f}%)")

    if success > 0:
        success_results = [r for r in results if r["processing_status"] == "Success"]
        pathology_counts = pd.Series([r["pathology"] for r in success_results]).value_counts()

        print("\nРаспределение по диагнозам (успешные):")
        for pathology, count in pathology_counts.items():
            print(f"  {pathology}: {count} ({count/success*100:.1f}%)")

        times = [r["time_of_processing"] for r in success_results]
        print("\nВремя обработки (успешные):")
        print(f"  Среднее: {sum(times)/len(times):.1f} сек")
        print(f"  Мин: {min(times):.1f} сек")
        print(f"  Макс: {max(times):.1f} сек")

        # Статистика по патологиям (если есть в результатах)
        if success_results and "most_dangerous_pathology_type" in success_results[0]:
            dangerous_pathologies = pd.Series(
                [r["most_dangerous_pathology_type"] for r in success_results]
            ).value_counts()
            print("\nТоп-5 наиболее частых опасных патологий:")
            for i, (pathology, count) in enumerate(dangerous_pathologies.head(5).items(), 1):
                print(f"  {i}. {pathology}: {count}")

    print(f"\n✅ Готово! Результаты сохранены в: {output_excel}")


def main():
    parser = argparse.ArgumentParser(description="Batch Inference Client для CT-CLIP + LightGBM API")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Путь к директории с ZIP архивами DICOM файлов",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Путь для сохранения Excel файла (по умолчанию: results_YYYYMMDD_HHMMSS.xlsx)",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default="http://localhost:8000",
        help="URL API сервиса (по умолчанию: http://localhost:8000)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"❌ Директория не найдена: {input_dir}")
        return 1

    if args.output:
        output_excel = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_excel = Path(f"results_{timestamp}.xlsx")

    batch_process(args.api_url, input_dir, output_excel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
