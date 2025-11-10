import numpy as np
import matplotlib.pyplot as plt
from oracles import QuadraticOracle
from optimization import gradient_descent
from plot_trajectory_2d import plot_levels, plot_trajectory

def analyze_gradient_descent_cases():
    """
    Анализ траекторий градиентного спуска для различных квадратичных функций
    """
    
    # Случай 1: Хорошо обусловленная функция (число обусловленности ~1)
    print("Случай 1: Хорошо обусловленная функция")
    A1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    b1 = np.array([0.0, 0.0])
    oracle1 = QuadraticOracle(A1, b1)
    
    # Случай 2: Плохо обусловленная функция (большое число обусловленности)
    print("Случай 2: Плохо обусловленная функция")
    A2 = np.array([[1.0, 0.0], [0.0, 100.0]])
    b2 = np.array([0.0, 0.0])
    oracle2 = QuadraticOracle(A2, b2)
    
    # Случай 3: Функция с корреляцией между переменными
    print("Случай 3: Функция с корреляцией")
    A3 = np.array([[2.0, 1.5], [1.5, 2.0]])
    b3 = np.array([1.0, -1.0])
    oracle3 = QuadraticOracle(A3, b3)
    
    # Начальные точки для тестирования
    start_points = [
        np.array([4.0, 4.0]),   # Далеко от минимума
        np.array([-3.0, 2.0]),  # Другая сторона
        np.array([0.5, -0.5])   # Близко к минимуму
    ]
    
    # Стратегии поиска шага
    strategies = [
        {'method': 'Constant', 'c': 0.1},
        {'method': 'Constant', 'c': 1.0},
        {'method': 'Armijo', 'c1': 1e-4, 'alpha_0': 1.0},
        {'method': 'Wolfe', 'c1': 1e-4, 'c2': 0.9, 'alpha_0': 1.0}
    ]
    
    strategy_names = ['Const(0.1)', 'Const(1.0)', 'Armijo', 'Wolfe']
    
    # Тестируем все комбинации
    for i, (oracle, oracle_name) in enumerate([(oracle1, "Хорошо обусловленная"),
                                              (oracle2, "Плохо обусловленная"),
                                              (oracle3, "С корреляцией")]):
        
        # Вычисляем собственные значения для анализа
        eigvals = np.linalg.eigvals(oracle.A)
        cond_number = max(eigvals) / min(eigvals)
        print(f"\n{oracle_name} функция:")
        print(f"  Собственные значения: {eigvals}")
        print(f"  Число обусловленности: {cond_number:.2f}")
        
        for j, start_point in enumerate(start_points):
            plt.figure(figsize=(15, 10))
            plt.suptitle(f'{oracle_name} функция, начальная точка {start_point}', fontsize=14)
            
            for k, (strategy, strategy_name) in enumerate(zip(strategies, strategy_names)):
                # Запускаем градиентный спуск
                x_star, message, history = gradient_descent(
                    oracle, start_point, 
                    tolerance=1e-6, 
                    max_iter=1000,
                    line_search_options=strategy,
                    trace=True,
                    display=False
                )
                
                # Рисуем траекторию
                plt.subplot(2, 2, k+1)
                plot_levels(oracle.func, xrange=[-5, 5], yrange=[-5, 5])
                if history and 'x' in history:
                    plot_trajectory(oracle.func, history['x'], label=strategy_name)
                plt.title(f'{strategy_name}\nИтераций: {len(history["x"]) if history else 0}')
                plt.legend()
            
            plt.tight_layout()
            plt.show()
            
            # Анализируем сходимость для одной комбинации
            analyze_convergence(oracle, start_point, strategies, strategy_names)

def analyze_convergence(oracle, start_point, strategies, strategy_names):
    """Анализ сходимости для разных стратегий"""
    print(f"\nАнализ сходимости из точки {start_point}:")
    
    convergence_data = []
    
    plt.figure(figsize=(12, 8))
    
    for k, (strategy, name) in enumerate(zip(strategies, strategy_names)):
        x_star, message, history = gradient_descent(
            oracle, start_point, 
            tolerance=1e-6,
            max_iter=1000,
            line_search_options=strategy,
            trace=True,
            display=False
        )
        
        if history:
            iterations = len(history['func'])
            final_grad_norm = history['grad_norm'][-1]
            convergence_data.append((name, iterations, final_grad_norm))
            
            # График сходимости
            plt.subplot(2, 2, k+1)
            plt.semilogy(history['grad_norm'], 'o-', linewidth=2)
            plt.title(f'{name} - {iterations} итераций')
            plt.xlabel('Итерация')
            plt.ylabel('Норма градиента')
            plt.grid(True)
            
            # Анализируем поведение градиента
            if len(history['grad_norm']) > 1:
                grad_decrease = history['grad_norm'][0] / history['grad_norm'][-1]
                print(f"  {name}: {iterations} итераций, градиент уменьшился в {grad_decrease:.1e} раз")
    
    plt.tight_layout()
    plt.show()
    
    return convergence_data

def analyze_condition_number_impact():
    """Анализ влияния числа обусловленности на сходимость"""
    condition_numbers = [1, 10, 100, 1000]
    start_point = np.array([3.0, 3.0])
    
    plt.figure(figsize=(15, 10))
    
    for i, cond in enumerate(condition_numbers):
        # Создаем матрицу с заданным числом обусловленности
        A = np.array([[1.0, 0.0], [0.0, cond]])
        oracle = QuadraticOracle(A, np.array([0.0, 0.0]))
        
        # Запускаем градиентный спуск с Armijo
        x_star, message, history = gradient_descent(
            oracle, start_point,
            line_search_options={'method': 'Armijo', 'c1': 1e-4, 'alpha_0': 1.0},
            trace=True
        )
        
        # Рисуем траекторию
        plt.subplot(2, 2, i+1)
        plot_levels(oracle.func, xrange=[-4, 4], yrange=[-4, 4])
        if history and 'x' in history:
            plot_trajectory(oracle.func, history['x'])
        plt.title(f'Число обусловленности: {cond}')
        
        # Анализируем сходимость
        if history:
            grad_norms = history['grad_norm']
            print(f"Число обусловленности {cond}: {len(grad_norms)} итераций")
    
    plt.tight_layout()
    plt.show()

def compare_strategies_for_ill_conditioned():
    """Сравнение стратегий для плохо обусловленной функции"""
    # Создаем сильно плохо обусловленную функцию
    A = np.array([[1.0, 0.0], [0.0, 1000.0]])
    oracle = QuadraticOracle(A, np.array([0.0, 0.0]))
    start_point = np.array([3.0, 3.0])
    
    strategies = [
        {'method': 'Constant', 'c': 0.001},  # Маленький шаг
        {'method': 'Constant', 'c': 0.1},    # Слишком большой шаг
        {'method': 'Armijo', 'c1': 1e-4, 'alpha_0': 1.0},
        {'method': 'Wolfe', 'c1': 1e-4, 'c2': 0.9, 'alpha_0': 1.0}
    ]
    
    strategy_names = ['Const(0.001)', 'Const(0.1)', 'Armijo', 'Wolfe']
    
    plt.figure(figsize=(15, 10))
    
    for k, (strategy, name) in enumerate(zip(strategies, strategy_names)):
        x_star, message, history = gradient_descent(
            oracle, start_point,
            tolerance=1e-6,
            max_iter=1000,
            line_search_options=strategy,
            trace=True
        )
        
        # График траектории
        plt.subplot(2, 2, k+1)
        plot_levels(oracle.func, xrange=[-4, 4], yrange=[-4, 4])
        if history and 'x' in history:
            plot_trajectory(oracle.func, history['x'])
        plt.title(f'{name} - {len(history["x"]) if history else 0} итераций')
        
        # График сходимости
        if history:
            plt.figure(figsize=(10, 6))
            plt.semilogy(history['grad_norm'], 'o-', label=name, linewidth=2)
            plt.xlabel('Итерация')
            plt.ylabel('Норма градиента')
            plt.title('Сравнение сходимости для плохо обусловленной функции')
            plt.legend()
            plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def summary_analysis():
    """Сводный анализ поведения градиентного спуска"""
    print("=" * 60)
    print("СВОДНЫЙ АНАЛИЗ ПОВЕДЕНИЯ ГРАДИЕНТНОГО СПУСКА")
    print("=" * 60)
    
    # Тестовые функции
    functions = [
        (np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0.0, 0.0]), "Хорошо обусловленная"),
        (np.array([[1.0, 0.0], [0.0, 100.0]]), np.array([0.0, 0.0]), "Плохо обусловленная"),
        (np.array([[2.0, 1.8], [1.8, 2.0]]), np.array([1.0, -1.0]), "С корреляцией")
    ]
    
    start_point = np.array([3.0, 3.0])
    strategy = {'method': 'Armijo', 'c1': 1e-4, 'alpha_0': 1.0}
    
    results = []
    
    for A, b, name in functions:
        oracle = QuadraticOracle(A, b)
        eigvals = np.linalg.eigvals(A)
        cond_number = max(eigvals) / min(eigvals)
        
        x_star, message, history = gradient_descent(
            oracle, start_point,
            line_search_options=strategy,
            trace=True
        )
        
        iterations = len(history['x']) if history else 0
        results.append((name, cond_number, iterations))
        
        print(f"\n{name}:")
        print(f"  Число обусловленности: {cond_number:.2f}")
        print(f"  Итераций до сходимости: {iterations}")
        print(f"  Сообщение: {message}")
    
    print("\n" + "=" * 60)
    print("ВЫВОДЫ:")
    print("1. Хорошо обусловленные функции сходятся быстро и прямо")
    print("2. Плохо обусловленные функции требуют много итераций, траектории зигзагообразные")
    print("3. Функции с корреляцией имеют эллиптические линии уровня")
    print("4. Стратегия Армихо надежна для всех типов функций")
    print("5. Константный шаг требует тщательного подбора")
    print("=" * 60)

# Запуск анализа
if __name__ == "__main__":
    analyze_gradient_descent_cases()
    analyze_condition_number_impact()
    compare_strategies_for_ill_conditioned()
    summary_analysis()
