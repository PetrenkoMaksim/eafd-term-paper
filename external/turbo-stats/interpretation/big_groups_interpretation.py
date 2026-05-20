import pandas as pd
import os


def group_features_by_type(csv_path, output_path=None):
    """
    Группирует фичи из CSV файла по категориям:
    - time: фичи со временем
    - sum: фичи с sum
    - mean: фичи с mean
    - cnt: фичи с cnt
    - tfidf: фичи с tf-idf
    - other: остальные фичи
    """
    # Читаем исходный файл
    df = pd.read_csv(csv_path)
    
    # Определяем категории для каждой фичи
    categories = []
    
    for target in df['target']:
        if any(keyword in target for keyword in ['_day', 'period', 'gap_', 'days_ratio', 'tx_per_day', 'active_days']):
            categories.append('time')
        elif '_sum' in target or target.startswith('sum_'):
            categories.append('sum')
        elif '_mean' in target:
            categories.append('mean')
        elif '_cnt' in target:
            categories.append('cnt')
        elif 'tfidf' in target:
            categories.append('tfidf')
        else:
            categories.append('other')
    
    # Добавляем колонку с категорией
    df['category'] = categories
    
    # Вычисляем средние R² по категориям
    category_order = ['time', 'sum', 'mean', 'cnt', 'tfidf', 'other']
    df['category'] = pd.Categorical(df['category'], categories=category_order)
    
    # Группируем по категориям и вычисляем среднее
    df_means = df.groupby('category', observed=True)['r2'].agg(['mean', 'count']).reset_index()
    df_means.columns = ['category', 'r2_mean', 'count']
    df_means = df_means.sort_values('category')
    
    # Сохраняем результат
    if output_path is None:
        base_path = os.path.splitext(csv_path)[0]
        output_path = f"{base_path}_grouped.csv"
    
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df_means.to_csv(output_path, index=False)
    
    # Формируем отчет для записи в файл
    report_lines = []
    report_lines.append(f"Средние R² по категориям. Результат сохранен в: {output_path}\n")
    report_lines.append("=" * 80)
    report_lines.append(f"{'Категория':<15s} | {'Среднее R²':<15s} | {'Количество фичей':<20s}")
    report_lines.append("=" * 80)
    
    for _, row in df_means.iterrows():
        report_lines.append(f"{row['category']:<15s} | {row['r2_mean']:>15.6f} | {int(row['count']):>20d}")
    
    # Сохраняем отчет в текстовый файл
    report_path = output_path.replace('.csv', '_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    # Выводим статистику в консоль
    print('\n'.join(report_lines))
    print(f"\nОтчет сохранен в: {report_path}")
    
    return df_means


if __name__ == "__main__":
    # Путь к исходному файлу
    csv_path = "interpretation/rosbank/LLM4ES.csv"
    
    # Группируем фичи
    grouped_df = group_features_by_type(csv_path)

