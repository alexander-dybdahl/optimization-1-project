import numpy as np

def strongwolfe(f, fprime, x, d, edges, l, p, mg, k, c, rhog, energy, grad_energy, mu=False, initial_step_length=1.0, c1=1e-2, c2=0.99, rho=2.0, maxit=1000):
    alpha_max = initial_step_length
    alpha_min = 0.0

    x_next = x + alpha_max * d
    if mu:
        energy_next = f(x_next, edges, l, p, mg, k, c, rhog, mu)
        grad_energy_next = fprime(x_next, edges, l, p, mg, k, c, rhog, mu)
    else:
        energy_next = f(x_next, edges, l, p, mg, k, c, rhog)
        grad_energy_next = fprime(x_next, edges, l, p, mg, k, c, rhog)
    armijo = energy_next <= energy + c1 * alpha_max * grad_energy
    grad_d = np.inner(d, grad_energy_next)
    curvature_low = grad_d >= c2 * grad_energy
    curvature_high = grad_d <= -c2 * grad_energy
    numit = 1

    while armijo and (not curvature_low) and numit < maxit:
        alpha_min = alpha_max
        alpha_max *= rho
        x_next = x + alpha_max * d
        if mu:
            energy_next = f(x_next, edges, l, p, mg, k, c, rhog, mu)
            grad_energy_next = fprime(x_next, edges, l, p, mg, k, c, rhog, mu)
        else:
            energy_next = f(x_next, edges, l, p, mg, k, c, rhog)
            grad_energy_next = fprime(x_next, edges, l, p, mg, k, c, rhog)
        armijo = energy_next <= energy + c1 * alpha_max * grad_energy
        grad_d = np.inner(d, grad_energy_next)
        curvature_low = grad_d >= c2 * grad_energy
        curvature_high = grad_d <= -c2 * grad_energy
        numit += 1

    alpha = alpha_max

    while not (armijo and curvature_low and curvature_high) and numit < maxit:
        if armijo and (not curvature_low):
            alpha_min = alpha
        else:
            alpha_max = alpha
        alpha = (alpha_min + alpha_max) / 2
        x_next = x + alpha * d
        if mu:
            energy_next = f(x_next, edges, l, p, mg, k, c, rhog, mu)
            grad_energy_next = fprime(x_next, edges, l, p, mg, k, c, rhog, mu)
        else:
            energy_next = f(x_next, edges, l, p, mg, k, c, rhog)
            grad_energy_next = fprime(x_next, edges, l, p, mg, k, c, rhog)
        armijo = energy_next <= energy + c1 * alpha * grad_energy
        grad_d = np.inner(d, grad_energy_next)
        curvature_low = grad_d >= c2 * grad_energy
        curvature_high = grad_d <= -c2 * grad_energy
        numit += 1

    return x_next, grad_energy_next

def bfgs(f, fprime, x0, edges, l, p, mg, k, c, rhog, X_analytic=np.empty(shape=(0, 0)), mu=False, maxiter=1000, eps=1e-14):
    x = x0.copy().flatten()
    n = len(x)
    B = np.eye(n)
    if mu:
        energy = f(x, edges, l, p, mg, k, c, rhog, mu)
        grad = fprime(x, edges, l, p, mg, k, c, rhog, mu)
    else:
        energy = f(x, edges, l, p, mg, k, c, rhog)
        grad = fprime(x, edges, l, p, mg, k, c, rhog)
    numit = 1
    vals = list()
    
    for i in range(maxiter):
        if len(X_analytic) > 0:
            vals.append(np.linalg.norm(x.reshape(-1,3) - X_analytic))
        else:
            vals.append(np.linalg.norm(grad))
        d = -np.dot(B, grad)
        xnew, gradnew = strongwolfe(f, fprime, x, d, edges, l, p, mg, k, c, rhog, energy, np.inner(d, grad), mu=mu, initial_step_length = 1.0)
        s = xnew - x
        y = gradnew - grad
        rho = 1.0 / np.dot(y, s)
        B = (np.eye(n) - rho * np.outer(s, y)) @ B @ (np.eye(n) - rho * np.outer(y, s)) + rho * np.outer(s, s)
        x = xnew
        if mu:
            grad = fprime(x, edges, l, p, mg, k, c, rhog, mu)
        else:
            grad = fprime(x, edges, l, p, mg, k, c, rhog)
        if np.linalg.norm(grad) < eps:
            break
        numit += 1
    return x.reshape(-1,3), vals, numit


def optimize_system(f, fprime, x, edges, l, p, mg, k, c, rhog, X_analytic=np.empty(shape=(0, 0)), mu=False, maxiter=1000, eps=1e-14):
    mu = 1
    for i in range(10):
        res, vals, numit = bfgs(f, fprime, x, edges, l, p, mg, k, c, rhog, X_analytic=X_analytic, mu=mu, maxiter=maxiter, eps=eps)
        if numit > 1000:
            mu *= 2
        if numit > 500:
            mu *= 2
        if numit > 100:
            mu *= 2
        print(i+1, mu, numit)
    res, vals, numit = bfgs(f, fprime, x, edges, l, p, mg, k, c, rhog, X_analytic=X_analytic, mu=mu, maxiter=maxiter, eps=eps)
    return res, vals