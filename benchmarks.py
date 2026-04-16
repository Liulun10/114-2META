import numpy as np

def get_function_details(func_name):
    """
    回傳函數的搜尋範圍、維度上限以及理論最優解座標
    """
    details = {
        "F2":  {"range": [-10, 10],   "dim": 30, "target_x": 0},
        "F6":  {"range": [-100, 100],   "dim": 30, "target_x": 0},
        "F9":  {"range": [-5.12, 5.12], "dim": 30, "target_x": 0},
        "F11": {"range": [-600, 600], "dim": 30, "target_x": 0},
        "F13": {"range": [-50, 50],     "dim": 30, "target_x": 1},
        "F15": {"range": [-5, 5],       "dim": 4,  "target_x": None}, # 特殊常數函數
        "F17": {"range": [[-5, 10], [0, 15]], "dim": 2, "target_x": None}
    }
    return details.get(func_name)

# 1. F2: Schwefel 2.22 (Unimodal)
def F2(x):
    # f(x) = sum(|xi|) + prod(|xi|)
    abs_x = np.abs(x)
    return np.sum(abs_x) + np.prod(abs_x)

def F6(x):
    # f(x) = sum([xi+0.5]**2)
    return np.sum(np.floor(x + 0.5)**2)

# 2. F9: Rastrigin (Multimodal, Large number of local minima)
def F9(x):
    # f(x) = sum(xi^2 - 10*cos(2*pi*xi) + 10)
    return np.sum(x**2 - 10 * np.cos(2 * np.pi * x) + 10)

# 3. F11: Griewank (Multimodal)
def F11(x):
    # f(x) = 1/4000 * sum(xi^2) - prod(cos(xi/sqrt(i))) + 1
    sum_part = np.sum(x**2) / 4000
    # i 從 1 開始編號
    i = np.arange(1, len(x) + 1)
    prod_part = np.prod(np.cos(x / np.sqrt(i)))
    return sum_part - prod_part + 1

# 4. F13: Generalized Penalized Function (Complex Multimodal)
def F13(x):
    # 定義內部懲罰函數 u(xi, a, k, m)
    def u(xi, a, k, m):
        # 根據圖片公式：
        # xi > a: k*(xi-a)^m
        # -a <= xi <= a: 0
        # xi < -a: k*(-xi-a)^m
        res = np.zeros_like(xi)
        res[xi > a] = k * (xi[xi > a] - a)**m
        res[xi < -a] = k * (-xi[xi < -a] - a)**m
        return res

    # 維度 D
    D = len(x)
    
    # 第一部分：0.1 * { sin^2(3*pi*x1) + sum + (xD-1)^2 * [1 + sin^2(2*pi*xD)] }
    term1 = np.sin(3 * np.pi * x[0])**2
    
    # sum_{i=1}^{D-1} (xi - 1)^2 * [1 + sin^2(3*pi*xi+1)]
    # 注意：Python index 從 0 開始，所以 xi 是 x[:-1]，xi+1 是 x[1:]
    sum_part = np.sum((x[:-1] - 1)**2 * (1 + np.sin(3 * np.pi * x[1:])**2))
    
    term3 = (x[-1] - 1)**2 * (1 + np.sin(2 * np.pi * x[-1])**2)
    
    main_part = 0.1 * (term1 + sum_part + term3)
    
    # 第二部分：sum_{i=1}^D u(xi, 5, 100, 4)
    penalty_part = np.sum(u(x, 5, 100, 4))
    
    return main_part + penalty_part

# 5. F15: Kowalik's Function (Fixed-dimension, $D=4$)
def F15(x):
    # 這是作業中較特殊的常數函數
    a = np.array([0.1957, 0.1947, 0.1735, 0.1600, 0.0844, 0.0627, 
                  0.0456, 0.0342, 0.0323, 0.0235, 0.0246])
    b = 1 / np.array([0.25, 0.5, 1, 2, 4, 6, 8, 10, 12, 14, 16])
    
    # f(x) = sum( [ai - (x1*(bi^2 + bi*x2) / (bi^2 + bi*x3 + x4))]^2 )
    # 注意：x 只有 4 個維度
    numerator = x[0] * (b**2 + b * x[1])
    denominator = b**2 + b * x[2] + x[3]
    return np.sum((a - (numerator / denominator))**2)

# 7. F17: Branin Function (D=2)
def F17(x):
    # 注意：Branin 固定維度為 2
    x1 = x[0]
    x2 = x[1]
    
    # 定義係數
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    
    # 公式: f(x) = a(x2 - b*x1^2 + c*x1 - r)^2 + s(1 - t)cos(x1) + s
    term1 = a * (x2 - b * x1**2 + c * x1 - r)**2
    term2 = s * (1 - t) * np.cos(x1)
    return term1 + term2 + s