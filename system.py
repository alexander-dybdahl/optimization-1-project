import numpy as np

# total external energy
def E_ext(X, mg):
    return np.sum(mg * X[:,2])

# total external energy gradient
def E_ext_grad(X, mg):
    grad = np.zeros_like(X)
    grad[:,2] = mg
    return grad

# cable elastic energy
def E_cab_elast(eij, X, l, k):
    i, j = eij
    d = np.linalg.norm(X[i] - X[j])
    return (0.5 * k * np.maximum(d - l[i,j], 0)**2 / l[i,j]**2)

# cable elastic energy gradient
def E_cab_elast_grad(X, l, p, k, edges_cab):
    N = len(X)
    M = len(p)
    grad = np.zeros_like(X)
    for i in range(M,N):
        for j in range(N):
            if l[i,j] != 0 and ((i,j) in edges_cab or (j,i) in edges_cab):
                d = np.linalg.norm(X[i] - X[j])
                grad[i] += k * np.maximum(d - l[i,j], 0) * (X[i] - X[j]) / (d * l[i,j]**2)
    return grad

# bar elastic energy
def E_bar_elast(eij, X, l, c):
    i, j = eij
    d = np.linalg.norm(X[i] - X[j])
    return 0.5 * c * (d - l[i,j])**2 / l[i,j]**2

# bar elastic energy gradient
def E_bar_elast_grad(X, l, p, c, edges_bar):
    N = len(X)
    M = len(p)
    grad = np.zeros_like(X)
    for i in range(M,N):
        for j in range(N):
            if l[i,j] != 0 and ((i,j) in edges_bar or (j,i) in edges_bar):
                d = np.linalg.norm(X[i] - X[j])
                grad[i] += c * (d - l[i,j]) * (X[i] - X[j]) / (d * l[i,j]**2)
    return grad

# bar gravity
def E_bar_grav(eij, X, l, rhog):
    i, j = eij
    return 0.5 * rhog * l[i,j] * (X[i,2] + X[j,2])

# bar gravity gradient
def E_bar_grav_grad(X, l, p, rhog, edges_bar):
    N = len(X)
    M = len(p)
    grad = np.zeros_like(X)
    grad[:,2] = 0
    for i in range(M, N):
        for j in range(N):
            if (i,j) in edges_bar or (j,i) in edges_bar:
                grad[i,2] += 0.5 * rhog * l[i,j]
    return grad

# total energy
def E(X, edges, l, p, mg, k, c, rhog, mu):
    X = X.reshape(-1,3)
    edges_cab, edges_bar = edges
    if len(edges[1]) != 0:
        E_sum = E_ext(X, mg)
        for eij in edges_cab:
            E_sum += E_cab_elast(eij, X, l, k)
        for eij in edges_bar:
            E_sum += E_bar_elast(eij, X, l, c) + E_bar_grav(eij, X, l, rhog)
    else:
        E_sum = E_ext(X, mg)
        for eij in edges_cab:
            E_sum += E_cab_elast(eij, X, l, k)
    return E_sum

# total energy gradient
def E_grad(X, edges, l, p, mg, k, c, rhog, mu):
    X = X.reshape(-1,3)
    edges_cab, edges_bar = edges
    if len(edges[1]) != 0:
        grad = E_ext_grad(X, mg) + E_cab_elast_grad(X, l, p, k, edges_cab) + E_bar_elast_grad(X, l, p, c, edges_bar) + E_bar_grav_grad(X, l, p, rhog, edges_bar)
    else:
        grad = E_ext_grad(X, mg) + E_cab_elast_grad(X, l, p, k, edges_cab)
    return grad.flatten()

# total energy with penalties
def E_penalty(x, edges, l, p, mg, k, c, rhog, mu, mu1=1e8):
    x = x.reshape(-1,3)
    energy = E(x, edges, l, p, mg, k, c, rhog, mu)
    energy += 0.5 * mu * np.sum(np.maximum(0, -x[:,2])**2)
    energy += 0.5 * mu1 * ((x[0,0] - 1)**2 + (x[0,1] - 1)**2)
    energy += 0.5 * mu1 * (x[0,1] - x[1,1])**2
    return energy

# total energy gradient with penalties
def E_grad_penalty(x, edges, l, p, mg, k, c, rhog, mu, mu1=1e8):
    grad = E_grad(x, edges, l, p, mg, k, c, rhog, mu)
    grad[2::3] -= mu * np.maximum(0, -x[2::3])
    grad[0] += mu1 * (x[0] - 1)
    grad[1] += mu1 * (x[1] - 1)
    grad[1] += mu1 * (x[1] - 1)
    grad[4] += mu1 * (x[4] - 1)
    return grad
