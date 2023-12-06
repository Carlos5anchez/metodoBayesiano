import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
import emcee
import corner

# Generación de datos sintéticos
np.random.seed(42)
x = np.sort(10 * np.random.rand(100))

# Modelo 1: y = Ax^n
A1, n = 3.5, 2.0
y1 = A1 * x**n
y1 += 0.5 * np.random.randn(x.size)  # Añadir ruido

# Modelo 2: y = A*sin(wx)
A2, w = 1.5, 2
y2 = A2 * np.sin(w * x)
y2 += 0.5 * np.random.randn(x.size)  # Añadir ruido

# Función para calcular la verosimilitud
def lnlike(theta, x, y, yerr):
    A, n = theta
    model = A * x**n
    inv_sigma2 = 1.0/(yerr**2)
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

# Implementación con emcee
# Utilizamos scipy.optimize para obtener un buen punto de inicio

# Ejemplo para el Modelo 1:
nll = lambda *args: -lnlike(*args)
result = op.minimize(nll, [A1, n], args=(x, y1, 0.5))
A_ml, n_ml = result["x"]

# Configuración de las propiedades del problema
ndim, nwalkers = 2, 100
pos = [result["x"] + 1e-4*np.random.randn(ndim) for i in range(nwalkers)]

# Configuración del muestreador (sampler)
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike, args=(x, y1, 0.5))

# Ejecución del MCMC
sampler.run_mcmc(pos, 500)

# Graficación de los resultados
samples = sampler.chain[:, 50:, :].reshape((-1, ndim))
fig = corner.corner(samples, labels=["$A$", "$n$"], truths=[A1, n])
plt.show()
