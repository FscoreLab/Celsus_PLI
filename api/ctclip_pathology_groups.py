"""
Группировка CT-CLIP патологий по анатомическим и клиническим признакам.

Используется для создания агрегированных фичей, которые усиливают сигнал
для редких патологий и улучшают интерпретируемость модели.
"""

CTCLIP_PATHOLOGY_GROUPS = {
    # A. Лёгочная паренхима (альвеолярно-интерстициальные и очаговые признаки)
    "lung_parenchyma": [
        "Pulmonary nodule",
        "Lymphangitic carcinomatosis",
        "Pneumonia",
        "Tuberculosis",
        "Pulmonary emphysema",
        "Pulmonary fibrosis",
        "Pneumoconiosis",
        "Pulmonary edema",
        "Atelectasis",
        "Consolidation",
        "Ground-glass opacity",
        "Crazy-paving pattern",
        "Tree-in-bud pattern",
        "Mosaic attenuation",
        "Cavitation",
        "Radiation pneumonitis",
    ],
    
    # B. Дыхательные пути (трахея, бронхи, мелкие дыхательные пути)
    "airways": [
        "Bronchiectasis",
        "Chronic bronchitis",
        "Bronchial obstruction",
    ],
    
    # C. Лёгочные сосуды и крупные сосуды грудной клетки
    "vascular": [
        "Pulmonary embolism",
        "Chronic thromboembolic disease",
        "Main pulmonary artery enlargement",
        "Pulmonary arteriovenous malformation",
        "Aortic dissection",
        "Coronary artery calcification",
    ],
    
    # D. Плевра
    "pleural": [
        "Pleural effusion",
        "Pleural thickening",
        "Pleural plaques",
        "Pneumothorax",
        "Hemothorax",
        "Hydropneumothorax",
    ],
    
    # E. Средостение и лимфоузлы
    "mediastinum": [
        "Mediastinal or hilar lymphadenopathy",
        "Sarcoidosis",
        "Thymoma",
        "Lymphoma",
        "Mediastinal cyst",
        "Retrosternal goiter",
        "Pneumomediastinum",
    ],
    
    # F. Сердце и перикард
    "cardiac": [
        "Pericardial effusion",
        "Cardiac chamber enlargement",
    ],
    
    # G. Грудная стенка, диафрагма и прилежащие структуры
    "chest_wall": [
        "Rib fracture",
        "Chest wall mass or invasion",
        "Subcutaneous emphysema",
        "Diaphragmatic hernia",
        "Hiatal hernia",
        "Pectus excavatum or carinatum",
    ],
    
    # H. Костно-позвоночные структуры
    "bone_spine": [
        "Vertebral compression fracture",
        "Degenerative spine changes",
        "Scoliosis",
        "Bone metastases",
    ],
    
    # I. Послеоперационные изменения
    "postop": [
        "Postoperative changes (lobectomy or pneumonectomy)",
    ],
    
    # J. Экстрапульмональные инциденталомы в зоне сканирования
    "incidentalomas": [
        "Thyroid nodule",
        "Breast mass",
    ],
}


def get_all_grouped_pathologies():
    """Возвращает плоский список всех патологий, включённых в группы."""
    all_pathologies = []
    for group_pathologies in CTCLIP_PATHOLOGY_GROUPS.values():
        all_pathologies.extend(group_pathologies)
    return all_pathologies


def get_group_for_pathology(pathology: str) -> str:
    """Возвращает название группы для данной патологии."""
    for group_name, pathologies in CTCLIP_PATHOLOGY_GROUPS.items():
        if pathology in pathologies:
            return group_name
    return None

