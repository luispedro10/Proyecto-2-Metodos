#Proyecto 2, Metodos Numericos

#Maria Isabel Montoya
#Maria Jose Morales
#Luis Pedro Garcia


#Importacion de paquetes
import numpy as np

#Creacion de la funcion
def gauss_jordan(A, b, method='direct', initial_estimate=None, max_steps=1000, tolerance=1e-6):
    n = len(b)

    if method not in ['direct', 'iterative']:
        raise ValueError("El método debe ser 'direct' o 'iterative'")

    if method == 'iterative' and initial_estimate is None:
        raise ValueError("Debe proporcionar una estimación inicial para el método iterativo")

    if np.linalg.matrix_rank(A) < n:
        raise ValueError("La matriz A no es invertible")

    if method == 'direct':
        aug_matrix = np.hstack((A.astype(np.float64), b.reshape(-1, 1).astype(np.float64)))

        for i in range(n):
            divisor = aug_matrix[i, i]
            for j in range(i+1, n):
                factor = aug_matrix[j, i] / divisor
                aug_matrix[j] -= factor * aug_matrix[i]

        for i in range(n-1, -1, -1):
            divisor = aug_matrix[i, i]
            for j in range(i-1, -1, -1):
                factor = aug_matrix[j, i] / divisor
                aug_matrix[j] -= factor * aug_matrix[i]

        x = np.zeros(n)
        for i in range(n):
            x[i] = aug_matrix[i, -1] / aug_matrix[i, i]

        return x

    
    # Método iterativo
    elif method == 'iterative':
        x = np.array(initial_estimate)
        
        for step in range(max_steps):
            x_new = np.zeros_like(x)
            
            for i in range(n):
                sum_j = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x[i+1:])
                x_new[i] = (b[i] - sum_j) / A[i, i]
            
            # Verificar si la solución cumple con la tolerancia
            if np.linalg.norm(x_new - x, ord=np.inf) < tolerance:
                return x_new
            
            x = x_new
        
        return x
    else:
        raise NotImplementedError("Método no implementado")


    
    
    
####################    RESOLUCION DE EJEMPLOS   #########################
    
#Ejemplo sistema 2x2
  
A1 = np.array([[2, 1], [1, -1]])
b1 = np.array([2, -1])
solution1 = gauss_jordan(A1, b1)
print('Solución sistema 2x2:', solution1)

#Ejemplo sistema 3x3

A2 = np.array([[3, -1, 2], [-2, 4, -2], [0, -1, 3]])
b2 = np.array([5, -2, 8])
solution2 = gauss_jordan(A2, b2)
print('Solución sistema 3x3:', solution2)

#EJEMPLO 100X100

np.random.seed(0)  # Para reproducibilidad
A3 = np.random.rand(100, 100)
b3 = np.random.rand(100)
solution3 = gauss_jordan(A3, b3)
print('Solución sistema 100x100:', solution3)



#################    COMPARACION CON OTROS METODOS:######################

### Comparacion de la matriz A1 ##

#Comparacion con Eliminacion Gaussiana:

print('')
print('Comparaciones matriz ejemplo 1')
print('')


from scipy.linalg import lu_solve, lu_factor

# Utilizando las matrices del ejemplo 1
lu, piv = lu_factor(A1)
solution_gaussian_elimination = lu_solve((lu, piv), b1)
print('Solución por eliminación gaussiana:', solution_gaussian_elimination)


#Factorizacion PA-LU

# Utilizando las matrices del ejemplo 1
solution_pa_lu = lu_solve((lu, piv), b1)
print('Solución por factorización PA-LU:', solution_pa_lu)


#Comparacion con Gauus-Seidel

from scipy.linalg import solve

# Utilizando las matrices del ejemplo 1
solution_gauss_seidel = solve(A1, b1)
print('Solución por Gauss-Seidel:', solution_gauss_seidel)


print('')
print('Comparaciones matriz ejemplo 2')
print('')


### COMPARACION DE LA MATRIZ A2 ##

#Comparacion con Eliminacion Gaussiana:

from scipy.linalg import lu_solve, lu_factor

# Utilizando las matrices del ejemplo 2
lu, piv = lu_factor(A2)
solution_gaussian_elimination = lu_solve((lu, piv), b2)
print('Solución por eliminación gaussiana:', solution_gaussian_elimination)


#Factorizacion PA-LU

# Utilizando las matrices del ejemplo 2
solution_pa_lu = lu_solve((lu, piv), b2)
print('Solución por factorización PA-LU:', solution_pa_lu)


#Comparacion con Gauus-Seidel

from scipy.linalg import solve

# Utilizando las matrices del ejemplo 2
solution_gauss_seidel = solve(A2, b2)
print('Solución por Gauss-Seidel:', solution_gauss_seidel)




#ANALISIS DE ERROR

import time

# Ejemplo con el método gauss_jordan
start_time = time.time()
solution_gj = gauss_jordan(A1, b1)
end_time = time.time()
print('Error Gauss-Jordan:', np.linalg.norm(b1 - A1.dot(solution_gj)))
print('Tiempo Gauss-Jordan:', end_time - start_time)

