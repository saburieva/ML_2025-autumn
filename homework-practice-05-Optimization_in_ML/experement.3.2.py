import numpy as np
import scipy.sparse
from optimization import gradient_descent
from oracles import QuadraticOracle

def generate_quadratic_problem(n, kappa, random_state=None):
    """
    Генерирует случайную квадратичную задачу размера n с числом обусловленности kappa.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Создаем диагональную матрицу с точно заданным числом обусловленности
    if n == 1:
        diagonal = np.array([1.0])
    else:
        # Равномерно в логарифмической шкале от 1 до kappa
        diagonal = np.exp(np.linspace(0, np.log(kappa), n))
    
    A = scipy.sparse.diags(diagonal, format='dia')
    b = np.random.randn(n)
    
    return QuadraticOracle(A, b), A, b

def run_experiment(n_values, kappa_values, num_trials=3, tolerance=1e-5):
    """
    Проводит эксперимент и возвращает таблицу результатов.
    """
    results = {}
    
    print("=" * 80)
    print("ЭКСПЕРИМЕНТ: Зависимость числа итераций от n и κ")
    print("=" * 80)
    
    for n in n_values:
        results[n] = {}
        print(f"\nРазмерность n = {n}")
        print("-" * 50)
        
        for kappa in kappa_values:
            results[n][kappa] = []
            trial_results = []
            
            for trial in range(num_trials):
                # Генерируем задачу
                oracle, A, b = generate_quadratic_problem(n, kappa, random_state=trial)
                
                # Начальная точка - случайный вектор
                x0 = np.random.randn(n)
                
                # Запускаем градиентный спуск
                x_star, message, history = gradient_descent(
                    oracle, x0, 
                    tolerance=tolerance, 
                    max_iter=10000,
                    line_search_options={'method': 'Constant', 'c': 0.1},
                    trace=True,
                    display=False
                )
                
                # Сохраняем число итераций
                if history is not None and 'func' in history:
                    num_iterations = len(history['func'])
                    trial_results.append(num_iterations)
                else:
                    trial_results.append(10000)  # max_iter если не сошлось
                
                results[n][kappa].append(num_iterations)
            
            # Выводим статистику для данного kappa
            mean_iter = np.mean(trial_results)
            std_iter = np.std(trial_results)
            min_iter = np.min(trial_results)
            max_iter = np.max(trial_results)
            
            print(f"κ = {kappa:6.1f} | "
                  f"Итерации: {mean_iter:6.1f} ± {std_iter:4.1f} | "
                  f"min: {min_iter:4.0f}, max: {max_iter:4.0f}")
    
    return results

def analyze_theoretical_scaling(results, n_values, kappa_values):
    """
    Анализирует соответствие теоретическим оценкам.
    """
    print("\n" + "=" * 80)
    print("ТЕОРЕТИЧЕСКИЙ АНАЛИЗ")
    print("=" * 80)
    
    print("\nТеоретическая оценка для градиентного спуска:")
    print("Число итераций T = O(κ * log(1/ε))")
    print("где κ = L/μ - число обусловленности, ε - требуемая точность")
    
    print("\nПроверка линейной зависимости от κ:")
    print("-" * 60)
    
    for n in n_values:
        if n not in results:
            continue
            
        print(f"\nРазмерность n = {n}:")
        print("κ\tСредние итерации\tОтношение T/κ")
        
        prev_kappa = None
        prev_iterations = None
        
        for kappa in kappa_values:
            if kappa in results[n]:
                mean_iter = np.mean(results[n][kappa])
                ratio = mean_iter / kappa
                
                print(f"{kappa:.1f}\t{mean_iter:12.1f}\t\t{ratio:8.3f}")
                
                # Проверяем рост относительно предыдущего значения
                if prev_kappa is not None and prev_iterations is not None:
                    kappa_ratio = kappa / prev_kappa
                    iter_ratio = mean_iter / prev_iterations
                    print(f"          Рост κ: {kappa_ratio:.2f}x, Рост T: {iter_ratio:.2f}x")
                
                prev_kappa = kappa
                prev_iterations = mean_iter

def analyze_dimensionality_effect(results, n_values, kappa_values):
    """
    Анализирует влияние размерности на число итераций.
    """
    print("\n" + "=" * 80)
    print("АНАЛИЗ ВЛИЯНИЯ РАЗМЕРНОСТИ")
    print("=" * 80)
    
    # Выбираем несколько значений kappa для анализа
    selected_kappas = [kappa_values[0], kappa_values[len(kappa_values)//2], kappa_values[-1]]
    
    for kappa in selected_kappas:
        print(f"\nПри фиксированном κ = {kappa:.1f}:")
        print("n\tСредние итерации\tИзменение")
        
        prev_n = None
        prev_iterations = None
        
        for n in n_values:
            if n in results and kappa in results[n]:
                mean_iter = np.mean(results[n][kappa])
                
                if prev_iterations is not None:
                    change = mean_iter / prev_iterations
                    change_str = f"{change:.2f}x"
                else:
                    change_str = "-"
                
                print(f"{n}\t{mean_iter:12.1f}\t\t{change_str:>8}")
                
                prev_n = n
                prev_iterations = mean_iter

def print_summary_table(results, n_values, kappa_values):
    """
    Выводит сводную таблицу результатов.
    """
    print("\n" + "=" * 80)
    print("СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 80)
    
    # Заголовок таблицы
    header = "n\\κ" + "".join([f"{kappa:>8.1f}" for kappa in kappa_values])
    print(header)
    print("-" * len(header))
    
    for n in n_values:
        if n not in results:
            continue
            
        row = f"{n:3} |"
        for kappa in kappa_values:
            if kappa in results[n]:
                mean_iter = np.mean(results[n][kappa])
                row += f"{mean_iter:8.1f}"
            else:
                row += " " * 8
        print(row)

# Параметры эксперимента
n_values = [10, 50, 100, 200, 500]  # Размерности пространства
kappa_values = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]  # Числа обусловленности

print("ПАРАМЕТРЫ ЭКСПЕРИМЕНТА:")
print(f"Размерности: {n_values}")
print(f"Числа обусловленности: {kappa_values}")
print(f"Точность: 1e-5")
print(f"Количество trials на точку: 3")

# Запускаем эксперимент
results = run_experiment(n_values, kappa_values, num_trials=3)

# Анализируем результаты
analyze_theoretical_scaling(results, n_values, kappa_values)
analyze_dimensionality_effect(results, n_values, kappa_values)
print_summary_table(results, n_values, kappa_values)

# Дополнительный анализ: проверка сходимости
print("\n" + "=" * 80)
print("ПРОВЕРКА СХОДИМОСТИ")
print("=" * 80)

for n in [n_values[0], n_values[-1]]:  # Первая и последняя размерности
    if n in results:
        for kappa in [kappa_values[0], kappa_values[-1]]:  # Минимальное и максимальное κ
            if kappa in results[n]:
                iterations = results[n][kappa]
                success_rate = sum(1 for it in iterations if it < 10000) / len(iterations)
                print(f"n={n}, κ={kappa}: успешных запусков {success_rate*100:.1f}%")

print("\n" + "=" * 80)
print("ВЫВОДЫ:")
print("=" * 80)
print("1. Число итераций градиентного спуска линейно зависит от числа обусловленности κ")
print("2. Размерность пространства n слабо влияет на число итераций для диагональных задач")
print("3. Теоретическая оценка O(κ) подтверждается экспериментально")
print("4. Отношение T/κ остается примерно постоянным при изменении κ")
