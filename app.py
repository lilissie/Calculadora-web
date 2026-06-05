from flask import Flask, request, jsonify, send_from_directory
import sympy as sp
from flask_cors import CORS
import os
import numpy as np  # Importa NumPy para cálculos numéricos y manejo de arrays
import matplotlib.pyplot as plt  # Importa Matplotlib para graficar funciones
from scipy.optimize import newton  # Importa el método de Newton-Raphson para encontrar raíces
from scipy.interpolate import interp1d  # Importa interpolación para estimar raíces
from concurrent.futures import ThreadPoolExecutor  # Importa para usar hilos y paralelizar cálculos
from flask import Flask, render_template
from flask import Flask, render_template, send_from_directory
from flask_cors import CORS
import os


#----------------Ejecucion------------------
def analizar_funcion(puntos,funcion,a,b):
    x = np.linspace(a, b, puntos)  # Genera un array de 'puntos' valores equidistantes entre a y b
    y = evaluar_funcion(funcion, x)  # Evalúa la función en todos los puntos del array x
    

    if y is None:  # Verifica si hubo un error en la evaluación
        print("No se pudo evaluar la función.")  # Informa del error
        return None  # Sale de la función sin continuar

    f = lambda x_val: evaluar_funcion(funcion, x_val)  # Crea una función lambda evaluable punto por punto

    puntos_criticos = buscar_puntos_criticos(x, y)  # Busca máximos y mínimos locales (puntos críticos)

    print("------Puntos críticos encontrados------")  # Mensaje informativo
    print(puntos_criticos)  # Muestra los puntos críticos

    vect_noconsecutivos = preparar_vector(a, b, puntos_criticos)  # Prepara el vector de puntos para analizar raíces
    print("------Vector de puntos interesantes-----")  # Mensaje informativo
    print(vect_noconsecutivos)  # Muestra el vector preparado

    raices = buscar_raices_concurrente_xy(f, vect_noconsecutivos)  # Busca raíces entre los puntos usando procesamiento paralelo

    print("Raíces encontradas:", raices)  # Muestra las raíces encontradas

    return raices  # Devuelve la lista de raíces encontradas


#-------------------------------------------Definir funciones útiles--------------------------------------
def preparar_funcion(funcion_str):  # Función para reemplazar funciones escritas en español por funciones de NumPy
    for clave, valor in diccionario_funciones.items():  # Recorre cada clave (función en español) y su reemplazo
        funcion_str = funcion_str.replace(clave, valor)  # Reemplaza en el string de la función
    return funcion_str  # Devuelve la función modificada

# Diccionario que mapea funciones comunes en español a funciones válidas de Python/NumPy
diccionario_funciones = {
    "sen": "sin",  # Reemplaza 'sen' por 'sin'
    "sin": "sin",  # Asegura que 'sin' también se mantenga
    "cos": "cos",  # 'cos' se mantiene
    "tan": "tan",  # 'tan' se mantiene
    "tg": "tan",  # Reemplaza 'tg' por 'tan'
    "ln": "log",  # Reemplaza 'ln' por logaritmo natural
    "log": "log10",  # Reemplaza 'log' por log base 10
    "^": "**",  # Reemplaza el símbolo ^ por el operador de potencia de Python
    "e": "np.e",  # Reemplaza la constante 'e' por su versión en NumPy
    "exp": "exp",  # 'exp' se mantiene
    "raiz": "sqrt",  # Reemplaza 'raiz' por raíz cuadrada
    "abs": "abs",  # 'abs' se mantiene
    "pi": "np.pi"
}

def evaluar_funcion(funcion_str, x):  # Evalúa la función en un valor o array de valores x
    funcion_preparada = preparar_funcion(funcion_str)  # Aplica los reemplazos al string de la función
    try:
        import numpy as np  # Importa NumPy dentro del contexto
        import math  # Importa funciones matemáticas estándar
        contexto = {  # Define un contexto seguro con funciones permitidas
            "x": x,
            "np": np,
            "math": math,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "log": np.log,
            "log10": np.log10,
            "exp": np.exp,
            "sqrt": np.sqrt,
            "abs": np.abs
        }
        resultado = eval(funcion_preparada, contexto)  # Evalúa la función en el contexto definido
        return resultado  # Devuelve el resultado
    except Exception as e:  # Si hay error al evaluar
        return None  # Devuelve None

def es_funcion_valida(funcion_str):  # Verifica si la función es segura de ejecutar
    for palabra_prohibida in ['import', 'eval', '__']:  # Palabras peligrosas
        if palabra_prohibida in funcion_str:  # Si están en el string
            return False  # No es válida
    return True  # Si no hay problemas, es válida



#-------------------------------------------Puntos críticos----------------------------------------------
def buscar_puntos_criticos(x, y):  # Busca puntos críticos detectando cambios de signo en la derivada
    derivada = np.gradient(y, x)  # Calcula derivada numérica
    puntos_criticos = []  # Lista vacía para almacenar los puntos críticos
    
    for i in range(1, len(derivada)):  # Recorre los valores de la derivada
        if derivada[i-1] * derivada[i] < 0:  # Si cambia de signo, es un punto crítico
            x_critico = x[i-1] + (x[i] - x[i-1]) * abs(derivada[i-1]) / (abs(derivada[i-1]) + abs(derivada[i]))  # Interpolación lineal para estimar el punto
            y_critico = evaluar_funcion(funcion, x_critico)  # Evalúa la función en el punto crítico
            puntos_criticos.append([x_critico, y_critico])  # Agrega el punto a la lista
    
    return np.array(puntos_criticos)  # Devuelve los puntos como array

def escanear_cambios_signo(a, b, n=10):  # Divide el intervalo y busca cambios de signo
    x_vals = np.linspace(a, b, n)  # Genera n puntos en el intervalo
    y_vals = evaluar_funcion(funcion, x_vals)  # Evalúa la función en esos puntos
    puntos_cambio = []  # Lista para puntos con cambio de signo

    for i in range(len(x_vals) - 1):  # Recorre los puntos
        signo_actual = np.sign(y_vals[i])  # Signo actual
        signo_siguiente = np.sign(y_vals[i+1])  # Signo del siguiente
        if signo_actual != signo_siguiente and signo_actual != 0 and signo_siguiente != 0:  # Si cambia el signo
            puntos_cambio.append([x_vals[i], y_vals[i]])  # Guarda ambos puntos
            puntos_cambio.append([x_vals[i+1], y_vals[i+1]])

    return np.array(puntos_cambio)  # Devuelve los puntos

#-------------------------------------------Limpieza de puntos-------------------------------------------
def limpiarvec(puntos_criticos, tol=1e-6):  # Elimina puntos cercanos repetidos
    if len(puntos_criticos) == 0:  # Si no hay puntos
        return np.array([])  # Devuelve array vacío

    puntos = np.array(puntos_criticos)  # Convierte a array
    filtrados = []  # Lista para los filtrados
    last_sign = None  # Último signo
    last_punto = None  # Último punto

    for i, (x, y) in enumerate(puntos):  # Recorre los puntos
        if abs(y) <= tol:  # Si y ≈ 0
            cero = [x, 0.0]  # Punto raíz
            if len(filtrados) == 0 or not np.allclose(filtrados[-1], cero):  # Si no es duplicado
                filtrados.append(cero)  # Lo agrega
            last_sign = None
            last_punto = None
            continue

        current_sign = np.sign(y)  # Signo actual
        if last_sign is None:  # Si no hay anterior
            filtrados.append([x, y])  # Agrega
            last_sign = current_sign
            last_punto = [x, y]
            continue

        if current_sign != last_sign:  # Si cambia de signo
            if last_punto is not None and not np.allclose(filtrados[-1], last_punto):
                filtrados.append(last_punto)  # Agrega el punto anterior
            filtrados.append([x, y])  # Agrega el actual

        last_sign = current_sign
        last_punto = [x, y]

    if len(filtrados) > 1:  # Elimina duplicados muy cercanos
        unique_filtrados = [filtrados[0]]
        for p in filtrados[1:]:
            if not np.allclose(p, unique_filtrados[-1]):
                unique_filtrados.append(p)
        filtrados = unique_filtrados

    return np.array(filtrados)  # Devuelve puntos filtrados


def preparar_vector(a, b, puntos_criticos):
    """
    Prepara un conjunto de puntos clave para buscar raíces, incluyendo extremos del intervalo.
    """
    extremos = np.array([[a, evaluar_funcion(funcion, a)], [b, evaluar_funcion(funcion, b)]])  # Crea array con los extremos y sus valores

    if puntos_criticos is None or len(puntos_criticos) == 0:  # Si no hay puntos críticos
        print("No hay puntos críticos: escaneando cambios de signo...")  # Mensaje informativo
        vect_final = escanear_cambios_signo(a, b)  # Escanea cambios de signo en el intervalo

    elif len(puntos_criticos) == 1:  # Si hay un solo punto crítico
        print("Hay un solo punto crítico: sumando extremos...")  # Mensaje informativo
        vect_final = np.vstack((puntos_criticos, extremos))  # Agrega los extremos al punto crítico

    else:  # Si hay varios puntos críticos
        print("Hay varios puntos críticos: aplicando limpieza...")  # Mensaje informativo
        puntos_con_extremos = np.vstack((puntos_criticos, extremos))  # Agrega extremos a los puntos críticos
        puntos_ordenados = puntos_con_extremos[puntos_con_extremos[:, 0].argsort()]  # Ordena por coordenada x
        vect_final = limpiarvec(puntos_ordenados)  # Limpia puntos redundantes o cercanos

    return vect_final  # Devuelve el vector final de puntos

#----------------------------------------------Busqueda de raices---------------------
def encontrar_raiz_hibrida(f, a, b, tol=1e-6, max_iter=100):
    """
    Busca una raíz en el intervalo [a, b] usando:
    1. Bisección (garantiza convergencia)
    2. Newton-Raphson (rápido cerca de la raíz)
    3. Interpolación cuadrática como último recurso

    Parámetros:
    - f: función a analizar
    - a, b: extremos del intervalo
    - tol: tolerancia deseada
    - max_iter: máximo de iteraciones

    Retorna:
    - La raíz encontrada o None si falla
    """

    # Comprobamos que haya cambio de signo
    try:
        fa = f(a)  # Evalúa función en a
        fb = f(b)  # Evalúa función en b
    except Exception as e:  # Maneja errores de evaluación
        print(f"Error al evaluar la función: {e}")  # Imprime el error
        return None  # No se puede continuar

    if fa * fb >= 0:  # Si no hay cambio de signo
        print("No hay cambio de signo en el intervalo.")  # Informa
        return None  # Sale sin buscar raíz

    # Paso 1: Bisección
    for _ in range(max_iter):  # Itera hasta max_iter veces
        c = (a + b) / 2  # Punto medio
        fc = f(c)  # Evalúa función en el medio

        if abs(fc) < tol or abs(b - a) < tol:  # Si se cumple la tolerancia
            break  # Termina bucle

        if fa * fc < 0:  # Cambio de signo entre a y c
            b = c  # Nuevo extremo derecho
            fb = fc
        else:  # Cambio de signo entre c y b
            a = c  # Nuevo extremo izquierdo
            fa = fc

    x_init = (a + b) / 2  # Valor inicial para Newton

    def df(x, h=1e-5):  # Derivada numérica centrada
        return (f(x + h) - f(x - h)) / (2 * h)

    # Paso 2: Intento con Newton-Raphson
    try:
        raiz_newton = newton(f, x0=x_init, fprime=df, tol=tol, maxiter=max_iter)  # Aplica Newton
        if a <= raiz_newton <= b:  # Verifica que esté dentro del intervalo
            return raiz_newton  # Devuelve la raíz
    except Exception:
        pass  # Ignora si falla

    # Paso 3: Respaldo con interpolación cuadrática
    try:
        x_vals = np.array([a, x_init, b])  # Puntos x
        y_vals = f(x_vals)  # Valores correspondientes
        interpolador = interp1d(y_vals, x_vals, kind='quadratic', fill_value="extrapolate")  # Crea interpolador
        raiz_interp = float(interpolador(0))  # Calcula valor de x para f(x)=0

        if a <= raiz_interp <= b and abs(f(raiz_interp)) < tol:  # Verifica validez
            return raiz_interp  # Devuelve raíz interpolada
    except Exception:
        pass  # Ignora si falla

    return None  # No se encontró raíz válida

from concurrent.futures import ThreadPoolExecutor
import numpy as np

def buscar_raices_concurrente_xy(f, posiciones_vec, tol=1e-10, tolerancia_cero=1e-12):
    """
    Busca raíces entre pares consecutivos de puntos en paralelo y devuelve coordenadas (x, y) como texto preciso.
    
    Parámetros:
    - f: función a evaluar
    - posiciones_vec: array de puntos [[x0,y0], [x1,y1], ...]
    - tol: tolerancia para raíces
    - tolerancia_cero: umbral para considerar y=0
    
    Devuelve:
    - Lista de strings con formato "(x, y)" para mostrar en tabla
    """
    
    def procesar_par(x0, y0, x1, y1):
        if abs(y0) < tolerancia_cero:
            return (x0, 0.0)
        if abs(y1) < tolerancia_cero:
            return (x1, 0.0)
        if y0 * y1 < 0:
            x_raiz = encontrar_raiz_hibrida(f, x0, x1, tol)
            if x_raiz is not None:
                return (x_raiz, 0.0)
        return None

    posiciones_vec = np.array(posiciones_vec, dtype=np.float64)
    trabajos = [(posiciones_vec[i,0], posiciones_vec[i,1], 
                 posiciones_vec[i+1,0], posiciones_vec[i+1,1]) 
                for i in range(len(posiciones_vec)-1)]
    
    raices = []
    with ThreadPoolExecutor(max_workers=min(32, len(trabajos)+1)) as executor:
        futuros = [executor.submit(procesar_par, *args) for args in trabajos]
        for futuro in futuros:
            if (resultado := futuro.result()) is not None:
                raices.append(resultado)

    # Eliminar duplicados por cercanía numérica sin redondear
    raices_unicas = []
    for raiz in sorted(raices, key=lambda x: x[0]):
        x, y = float(raiz[0]), float(raiz[1])
        if not any(abs(x - r[0]) < tol for r in raices_unicas):
            raices_unicas.append((x, y))

    # Convertir a texto de alta precisión (12 decimales)
    raices_texto = [f"({x:.12f}, {y:.12f})" for x, y in raices_unicas]

    return raices_unicas

def manejar_intervalos(a, b):
    """
    Maneja los intervalos recibidos:
    - Si ambos son números: devuelve (min, max)
    - Si solo uno es número: crea intervalo de ±10 alrededor de ese valor
    - Si ambos son None: devuelve intervalo por defecto (-10, 10)
    """
    if a is not None and b is not None:
        return (min(a, b), max(a, b))
    
    if a is not None:
        return (a - 10, a + 10) if a - 10 < a + 10 else (a + 10, a - 10)
    
    if b is not None:
        return (b - 10, b + 10) if b - 10 < b + 10 else (b + 10, b - 10)
    
    return (-10, 10)  # Caso por defecto cuando ambos son None

#------------------------------------------------Fin Funciones de raices------------------------------------
#------------------------------------------------Comienzo sist e ------------------------------------
import numpy as np

def gauss(A_b):
    """
    Eliminación de Gauss con pivoteo parcial.
    A_b: numpy array (n, n+1) — matriz aumentada [A|b].
    Retorna un único string con todos los pasos y la solución final.
    """
    # Asegurarnos de trabajar sobre floats y copia
    try:
        Ab = A_b.astype(float).copy()
    except:
        return "Error: la matriz debe ser convertible a float."
    n, m = Ab.shape
    if m != n + 1:
        return f"Error: se esperaba matriz aumentada de dimensiones {n}×{n+1}."

    pasos = []
    pasos.append("Matriz aumentada inicial:")
    pasos.append(str(Ab))

    # === Eliminación hacia adelante ===
    for i in range(n):
        # Pivoteo parcial
        max_row = i + np.argmax(np.abs(Ab[i:, i]))
        if abs(Ab[max_row, i]) < 1e-12:
            return "\n".join(pasos) + "\nError: pivote nulo. Sistema singular o infinitas soluciones."
        if max_row != i:
            pasos.append(f"\nIntercambiando fila {i+1} con fila {max_row+1}:")
            Ab[[i, max_row]] = Ab[[max_row, i]]
            pasos.append(str(Ab))

        # Normalizar fila pivote
        pivot = Ab[i, i]
        pasos.append(f"\nDividiendo fila {i+1} por pivote {pivot:.4f}:")
        Ab[i] = Ab[i] / pivot
        pasos.append(str(Ab))

        # Eliminar hacia adelante
        for j in range(i+1, n):
            factor = Ab[j, i]
            if abs(factor) > 1e-12:
                pasos.append(f"\nRestando {factor:.4f}×fila {i+1} de fila {j+1}:")
                Ab[j] -= factor * Ab[i]
                pasos.append(str(Ab))

    # === Sustitución hacia atrás ===
    x = np.zeros(n, dtype=float)
    for i in range(n-1, -1, -1):
        x[i] = Ab[i, -1] - np.dot(Ab[i, i+1:n], x[i+1:n])

    # Formatear la solución
    sol_str = ", ".join([f"x{i+1} = {x[i]:.4f}" for i in range(n)])
    pasos.append("\nSolución final:")
    pasos.append(sol_str)

    return "\n".join(pasos)


def gauss_jordan(A):
    import numpy as np
    n = len(A)
    
    # Verificar matriz aumentada
    if A.shape[1] != n + 1:
        return "Error: La matriz debe ser aumentada [A|b] con dimensiones n x (n+1)"
    
    Ab = A.copy().astype(float)
    pasos = ["Matriz aumentada inicial:\n" + str(Ab)]

    for i in range(n):
        pasos.append(f"\n--- Paso {i + 1}: Pivot en columna {i+1} ---")

        # Pivoteo parcial
        max_row = np.argmax(abs(Ab[i:, i])) + i
        if i != max_row:
            pasos.append(f"Intercambiando fila {i+1} con fila {max_row+1}")
            Ab[[i, max_row]] = Ab[[max_row, i]]
            pasos.append(str(Ab))

        # Verificar pivote cero
        if abs(Ab[i, i]) < 1e-12:
            return "Error: Sistema singular o con infinitas soluciones"

        # Normalizar fila pivote
        pivot = Ab[i, i]
        pasos.append(f"Normalizando fila {i+1} dividiendo por {pivot:.4f}")
        Ab[i] = Ab[i] / pivot
        pasos.append(str(Ab))

        # Eliminación hacia arriba y abajo (Gauss-Jordan)
        for j in range(n):
            if j != i:  # Excluir la fila del pivote
                factor = Ab[j, i]
                if abs(factor) > 1e-12:
                    pasos.append(f"Restando {factor:.4f} × fila {i+1} a fila {j+1}")
                    Ab[j] -= factor * Ab[i]
                    pasos.append(str(Ab))

    # Verificar solución única
    for i in range(n):
        if not np.isclose(Ab[i, i], 1.0) or np.any(~np.isclose(Ab[i, :i], 0.0)):
            return "Error: El sistema no tiene solución única"

    # Extraer soluciones
    soluciones = np.round(Ab[:, -1], 4)
    resultado = "Soluciones: " + ", ".join([f"x{i+1} = {soluciones[i]}" for i in range(n)])
    
    texto_completo = "\n".join(pasos) + "\n\n" + resultado
    return texto_completo



def es_dominante(A):
    """Verifica si la matriz es diagonalmente dominante"""
    for i in range(len(A)):
        if abs(A[i][i]) < sum(abs(A[i][j]) for j in range(len(A)) if j != i):
            return False
    return True

def hacer_dominante(A, b):
    """Intenta reordenar las filas para hacer la matriz diagonalmente dominante"""
    from itertools import permutations
    n = len(A)
    for perm in permutations(range(n)):
        Ap = A[list(perm)]
        bp = b[list(perm)]
        if es_dominante(Ap):
            return Ap, bp, perm
    return None, None, None

def gauss_seidel(matriz_aumentada, tol=1e-6, max_iter=100):
    """Resuelve el sistema usando Gauss-Seidel con matriz aumentada [A|b]"""
    
    # Verificar matriz aumentada
    try:
        A = np.array(matriz_aumentada[:, :-1], dtype=float)
        b = np.array(matriz_aumentada[:, -1], dtype=float)
        n = len(A)
    except:
        return "Error: Formato de matriz incorrecto. Debe ser n x (n+1)"
    
    # Verificar diagonal no nula
    if np.any(np.diag(A) == 0):
        return "Error: La matriz tiene ceros en la diagonal principal"
    
    # Preparar resultado
    resultado = []
    resultado.append("=== Método Gauss-Seidel ===")
    resultado.append("\nMatriz A:")
    resultado.append(str(A))
    resultado.append("\nVector b:")
    resultado.append(str(b))
    
    # Intentar hacer dominante
    A_dom, b_dom, perm = hacer_dominante(A, b)
    if A_dom is not None:
        resultado.append("\n\nSistema reordenado para ser diagonalmente dominante")
        resultado.append(f"Orden de filas: {[i+1 for i in perm]}")
        resultado.append("\nNueva matriz A:")
        resultado.append(str(A_dom))
        resultado.append("\nNuevo vector b:")
        resultado.append(str(b_dom))
        A, b = A_dom, b_dom
    
    # Inicialización
    x = np.zeros(n)
    resultado.append(f"\n\nVector inicial x⁰: {x.round(6)}")
    
    # Iteraciones
    convergencia = False
    for it in range(1, max_iter + 1):
        x_old = x.copy()
        
        for i in range(n):
            suma = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - suma) / A[i][i]
            
        error = np.linalg.norm(x - x_old, np.inf)
        resultado.append(f"\nIteración {it}: x = {x.round(6)}, error = {error:.6f}")
        
        if error < tol:
            convergencia = True
            resultado.append("\nConvergencia alcanzada")
            break
    
    if not convergencia:
        resultado.append("\nAdvertencia: No se alcanzó convergencia en las iteraciones máximas")
    
    # Resultado final
    resultado.append("\n\nSolución aproximada:")
    for i in range(n):
        resultado.append(f"x{i+1} = {x[i]:.6f}")
    
    return "\n".join(map(str, resultado))



import numpy as np
from itertools import permutations

def jacobi(matriz, n_iter=100, tol=1e-6):
    """
    Método de Jacobi sobre matriz aumentada [A|b].
    Devuelve un string con pasos y solución, o advertencia de no convergencia.
    """
    # 1) Convertir y chequear dimensiones
    try:
        Ab = np.array(matriz, dtype=float)
    except Exception as e:
        return f"Error al convertir la matriz a numpy: {e}"
    n, m = Ab.shape
    if m != n + 1:
        return "Error: La matriz debe ser n x (n+1)."
    A = Ab[:, :n]
    b = Ab[:, n]

    # 2) Función de chequeo de dominancia
    def is_diag_dom(M):
        return all(abs(M[i,i]) > sum(abs(M[i,j]) for j in range(n) if j!=i)
                   for i in range(n))

    lines = []
    lines.append("=== Método de Jacobi ===")
    lines.append("Matriz A:\n" + str(A))
    lines.append("Vector b:\n" + str(b))

    # 3) Intentar permutar filas para dominancia diagonal
    if not is_diag_dom(A):
        for perm in permutations(range(n)):
            Ap = A[list(perm),:]
            bp = b[list(perm)]
            if is_diag_dom(Ap):
                A, b = Ap, bp
                lines.append(f"\nSe reordenaron filas para dominancia diagonal: perm {perm}")
                lines.append("Nueva A:\n" + str(A))
                lines.append("Nuevo b:\n" + str(b))
                break
        else:
            lines.append("\nADVERTENCIA: La matriz no es diagonalmente dominante. ¡Puede no converger!")

    # 4) Descomposición D, L, U
    D = np.diag(np.diag(A))
    L = np.tril(A, -1)
    U = np.triu(A, 1)
    try:
        D_inv = np.linalg.inv(D)
    except np.linalg.LinAlgError:
        return "Error: La matriz diagonal D no es invertible."

    lines.append("\nVector inicial x⁰ = " + str([0]*n))

    # 5) Iteraciones
    x_old = np.zeros(n)
    convergió = False
    for k in range(1, n_iter+1):
        x_new = D_inv @ (b - (L+U) @ x_old)
        error = np.linalg.norm(x_new - x_old, np.inf)
        lines.append(f"\nIteración {k}: x = {np.round(x_new,4).tolist()}, error = {error:.6f}")
        if error < tol:
            lines.append("Convergencia alcanzada.")
            convergió = True
            break
        x_old = x_new

    if not convergió:
        lines.append("\nNo se alcanzó convergencia en el número máximo de iteraciones.")

    # 6) Solución final
    sol = np.round(x_new, 4)
    sol_str = ", ".join([f"x{i+1} = {sol[i]:.4f}" for i in range(n)])
    lines.append("\nSolución final: " + sol_str)

    return "\n".join(lines)

# -------------------------------------------- 2do P-----------------------------------
# # --------> INTEGRACION


def integrar(f, a, b, n, metodo):
    h = (b - a) / n
    puntos = []  # ← Lista de pares (x, f(x))

    if metodo == "trapecio":
        suma = f(a) + f(b)
        puntos.append((a, f(a)))
        for i in range(1, n):
            xi = a + i * h
            fxi = f(xi)
            suma += 2 * fxi
            puntos.append((xi, fxi))
        puntos.append((b, f(b)))
        return (h / 2) * suma, puntos

    elif metodo == "simpson13":
        if n % 2 != 0:
            raise ValueError("n debe ser par para Simpson 1/3")
        suma = f(a) + f(b)
        puntos.append((a, f(a)))
        for i in range(1, n):
            xi = a + i * h
            fxi = f(xi)
            if i % 2 == 0:
                suma += 2 * fxi
            else:
                suma += 4 * fxi
            puntos.append((xi, fxi))
        puntos.append((b, f(b)))
        return (h / 3) * suma, puntos

    elif metodo == "simpson38":
        if n % 3 != 0:
            raise ValueError("n debe ser múltiplo de 3 para Simpson 3/8")
        suma = f(a) + f(b)
        puntos.append((a, f(a)))
        for i in range(1, n):
            xi = a + i * h
            fxi = f(xi)
            if i % 3 == 0:
                suma += 2 * fxi
            else:
                suma += 3 * fxi
            puntos.append((xi, fxi))
        puntos.append((b, f(b)))
        return (3 * h / 8) * suma, puntos

    else:
        raise ValueError("Método no reconocido")


# ------------------> EDOs


def euler(f, x0, y0, xf, n):
    h = (xf - x0) / n
    resultados = [(x0, y0)]
    for _ in range(n):
        y0 += h * f(x0, y0)
        x0 += h
        resultados.append((x0, y0))
    return resultados

def runge_kutta_4(f, x0, y0, xf, n):
    h = (xf - x0) / n
    resultados = [(x0, y0)]
    for _ in range(n):
        k1 = f(x0, y0)
        k2 = f(x0 + h / 2, y0 + h * k1 / 2)
        k3 = f(x0 + h / 2, y0 + h * k2 / 2)
        k4 = f(x0 + h, y0 + h * k3)
        y0 += (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        x0 += h
        resultados.append((x0, y0))
    return resultados

def resolver_edo(f, x0, y0, xf, n, metodo):
    if metodo == "euler":
        return euler(f, x0, y0, xf, n)
    elif metodo == "rk4":
        return runge_kutta_4(f, x0, y0, xf, n)
    else:
        raise ValueError("Método no reconocido. Usa 'euler' o 'rk4'.")

# Definimos la función f(x,y) que la EDO usará
def f(x, y):
    return (x**2 - 1) / (y**2 + 1)


#------------------------------------------------Fin------------------------------------

app = Flask(__name__, template_folder='templates')
CORS(app)  # Permite solicitudes CORS desde tu frontend

# Configuración para manejar rutas correctamente
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Desactiva caché para desarrollo
app.static_folder = 'static'

# Ruta principal (página de inicio)
@app.route('/')
def inicio():  # Cambié de 'home' a 'inicio'
    return render_template('inicio.html')

@app.route('/raices')
def raices():
    return render_template('index.html')  # O mejor cambia a 'raices.html'

# Ruta para la página de Teoría
@app.route('/teoria')
def teoria():
    return render_template('teoria.html')

# Ruta para sistemas de ecuaciones
@app.route('/sis_ec')
def sistemas_ecuaciones():
    return render_template('sis_ec.html')

# Manejo de errores 404
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.route("/desarrolladores")
def desarrolladores():
    return render_template("desarrolladores.html")

@app.route('/resolver-raices', methods=['POST'])
def resolver_raices():
    try:
        data = request.get_json()
        global funcion
        funcion = data.get('funcion', '').strip()
        print(f"\nFUNCIÓN RECIBIDA: f(x) = {funcion}")
        if not funcion or not es_funcion_valida(funcion):
            return jsonify({'error': 'Función no válida o vacía'}), 400

        # Procesamiento de intervalos (convierte a float o None si está vacío)
        try:
            a_str = data.get('intervalo1', '').strip()
            b_str = data.get('intervalo2', '').strip()
            
            a = float(a_str) if a_str else None
            b = float(b_str) if b_str else None
            
            # Aquí puedes pasar a otra función que maneje los None
            intervalo = manejar_intervalos(a, b)
            a_final, b_final = intervalo
            
        except ValueError:
            return jsonify({'error': 'Intervalos deben ser números válidos'}), 400

        
        print(f"Intervalo procesado: [{a_final}, {b_final}]")

        puntos = 1000
        raices = analizar_funcion(puntos, funcion, a_final, b_final) or []
        
        return jsonify({
            'raices': raices,
            'intervalo': [a_final, b_final],
            'funcion': funcion
        })
    
    except Exception as e:
        print(f"ERROR: {str(e)}")
        return jsonify({'error': f"Error interno: {str(e)}"}), 500



@app.route('/resolver-sistema', methods=['POST'])
def resolver_sistema():
    data = request.json
    matriz = np.array(data['matriz'])
    print("matrix que llego.-..", matriz)
    metodo = data['metodo']
    print("con el metodo...", metodo)
    
    if metodo == 'gauss':
        soluciones = gauss(matriz)
    elif metodo == 'gauss-jordan':
        soluciones = gauss_jordan(matriz)  # Asegúrate de tener esta función implementada
    elif metodo == 'gauss-seidel':
        soluciones = gauss_seidel(matriz)  # Asegúrate de tener esta función implementada
    elif metodo == 'jacobi':
        soluciones = jacobi(matriz)  # Asegúrate de tener esta función implementada
    else:
        return jsonify({"error": "Método no válido"}), 400
    
    return jsonify({"soluciones": soluciones})

# ---------------------- 2DO P ---------------------------------------------------------------------------


@app.route('/integracion', methods=['GET', 'POST'])
def integracion():
    if request.method == 'POST':
        try:
            data = request.get_json()

            funcion_str = data['funcion']
            a = float(data['a'])
            b = float(data['b'])
            n = int(data['n'])
            metodo = data['metodo']

            # Aquí debes convertir funcion_str en una función Python real
            # Por ejemplo, usando eval con precaución o SymPy.
            # f = lambda x: eval(funcion_str)  # CUIDADO: eval puede ser peligroso

            from sympy import sympify, lambdify, symbols
            x = symbols('x')
            f_expr = sympify(funcion_str)
            f = lambdify(x, f_expr, "math")

            resultado, puntos = integrar(f, a, b, n, metodo)

            return jsonify({
                'resultado': resultado,
                'puntos': [{'x': px, 'fx': fx} for px, fx in puntos]
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 400

    # Si es GET, solo renderiza la plantilla
    return render_template('integracion.html')


# # Ruta para la página de Ecuaciones Diferenciales Ordinarias (EDOs)
# from flask import request, jsonify, render_template

@app.route('/edo', methods=['GET', 'POST'])
def edo():
    if request.method == 'POST':
        data = request.json
        try:
            # Obtener datos del body
            funcion = data.get('funcion', '')
            x0 = float(data.get('x0', 0))
            y0 = float(data.get('y0', 1))
            xf = float(data.get('xf', 1))
            n = int(data.get('n', 10))
            metodo = data.get('metodo', 'euler').lower()

            # Definir f(x, y) en tiempo de ejecución
            def f(x, y):
                return eval(funcion, {"x": x, "y": y, "Math": __import__('math')})

            resultados = resolver_edo(f, x0, y0, xf, n, metodo)
            lista = [{"x": x, "y": y} for x, y in resultados]
            return jsonify({"resultados": lista})

        except Exception as e:
            return jsonify({"error": str(e)}), 400

    # Si es GET, renderizar la plantilla
    return render_template('edo.html')


if __name__ == '__main__':
    app.run(debug=True, port=5000)


