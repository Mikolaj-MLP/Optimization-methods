import numpy as np

def gradient_descent(f_grad, x0, lr=0.01, max_iters=1000, tol=1e-6):
    x = x0.copy()
    trajectory = [x.copy()]
    
    for i in range(max_iters):
        grad = f_grad(x)
        x -= lr * grad
        trajectory.append(x.copy())
        if np.linalg.norm(grad) < tol:
            break
    return x, trajectory


def steepest_descent(f, f_grad, x0, max_iters=1000, tol=1e-6):
    x = x0.copy()
    trajectory = [x.copy()]

    for i in range(max_iters):
        grad = f_grad(x)
        if np.linalg.norm(grad) < tol:
            break

        # Line search using backtracking
        alpha = 1.0
        beta = 0.5
        while f(x - alpha * grad) > f(x) - 0.5 * alpha * np.dot(grad, grad):
            alpha *= beta
        
        x -= alpha * grad
        trajectory.append(x.copy())
    return x, trajectory


def momentum_descent(f_grad, x0, lr=0.01, beta=0.9, max_iters=1000, tol=1e-6):
    x = x0.copy()
    v = np.zeros_like(x)
    trajectory = [x.copy()]

    for i in range(max_iters):
        grad = f_grad(x)
        v = beta * v - lr * grad
        x += v
        trajectory.append(x.copy())
        if np.linalg.norm(grad) < tol:
            break
    return x, trajectory


def nesterov_accelerated_gradient(f_grad, x0, lr=0.01, beta=0.9, max_iters=1000, tol=1e-6):
    x = x0.copy()
    v = np.zeros_like(x)
    trajectory = [x.copy()]

    for i in range(max_iters):
        lookahead = x + beta * v
        grad = f_grad(lookahead)
        v = beta * v - lr * grad
        x += v
        trajectory.append(x.copy())
        if np.linalg.norm(grad) < tol:
            break
    return x, trajectory


def adam_optimizer(f_grad, x0, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, max_iters=1000, tol=1e-6):
    x = x0.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    trajectory = [x.copy()]
    
    for t in range(1, max_iters + 1):
        grad = f_grad(x)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        x -= lr * m_hat / (np.sqrt(v_hat) + eps)
        trajectory.append(x.copy())

        if np.linalg.norm(grad) < tol:
            break
    return x, trajectory

def newton_method(f_grad, f_hess, x0, max_iters=100, tol=1e-6):
    x = x0.copy()
    trajectory = [x.copy()]
    
    for i in range(max_iters):
        grad = f_grad(x)
        hess = f_hess(x)
        
        try:
            delta = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break  # Hessian not invertible
        
        x -= delta
        trajectory.append(x.copy())
        
        if np.linalg.norm(grad) < tol:
            break
            
    return x, trajectory
