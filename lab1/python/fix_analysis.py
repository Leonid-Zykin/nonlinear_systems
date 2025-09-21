#!/usr/bin/env python3
# Скрипт для исправления ошибок в анализе систем

import re

def fix_analysis_file():
    """Исправляет ошибки в файле 3_analysis.tex"""
    
    with open('/home/leonidas/projects/itmo/nonlinear_systems/lab1/3_analysis.tex', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Исправление 1: Система 2 - матрица Якоби для точки (0,1)
    old_matrix = r'\item \(0, 1\): \$J = \\begin\{pmatrix\} 2 & 0 \\\\ 0 & 1 \\end\{pmatrix\}\$, \$\\lambda_1 = 2\$, \$\\lambda_2 = 1\$ --- неустойчивый узел'
    new_matrix = r'\item $(0, 1)$: $J = \\begin{pmatrix} 2 & 0 \\\\ 1 & 1 \\end{pmatrix}$, $\\lambda_1 = 2$, $\\lambda_2 = 1$ --- неустойчивый узел'
    content = content.replace(old_matrix, new_matrix)
    
    # Исправление 2: Система 2 - добавить матрицу Якоби для третьей точки
    third_point_old = r'\item \(1\.26, -1\): Численный анализ показывает седло'
    third_point_new = r'\item $(1.26, -1)$: $J = \\begin{pmatrix} 0 & 1.26 \\\\ -4.76 & 0.26 \\end{pmatrix}$, седло'
    content = content.replace(third_point_old, third_point_new)
    
    # Исправление 3: Система 4 - добавить анализ точки (0,0)
    system4_analysis = content.find('\\textbf{Точки равновесия:}\\n\\begin{itemize}\\n\\item $(0, 0)$ --- изолированная точка')
    if system4_analysis != -1:
        old_text = '\\item $(0, 0)$ --- изолированная точка'
        new_text = '\\item $(0, 0)$ --- неустойчивый фокус ($J = \\begin{pmatrix} 0 & -1 \\\\ 1 & 0 \\end{pmatrix}$, $\\lambda = \\pm i$)'
        content = content.replace(old_text, new_text)
    
    print("Исправления применены!")
    
    with open('/home/leonidas/projects/itmo/nonlinear_systems/lab1/3_analysis_fixed.tex', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("Исправленный файл сохранен как 3_analysis_fixed.tex")

if __name__ == "__main__":
    fix_analysis_file()
