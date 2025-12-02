import numpy as np

# --- Funções de Apoio ---
def triangular_superior(U, y):
    n, nc = np.shape(U)
    if y.ndim == 1: y = y.reshape(-1, 1)
    x = np.zeros((n, 1))
    
    for i in range(n - 1, -1, -1):
        if np.abs(U[i, i]) < 1e-15: 
            raise ValueError("Sistema singular: pivô zero na retro-substituição.")
        soma = U[i, i+1:n] @ x[i+1:n, 0]
        x[i, 0] = (y[i, 0] - soma) / U[i, i]
    return x

def triangular_inferior(L, b):
    n, nc = np.shape(L)
    if b.ndim == 1: b = b.reshape(-1, 1)
    y = np.zeros((n, 1))
    
    for i in range(n):
        if np.abs(L[i, i]) < 1e-15:
            raise ValueError("Sistema singular: pivô zero na substituição progressiva.")
        soma = L[i, 0:i] @ y[0:i, 0]
        y[i, 0] = (b[i, 0] - soma) / L[i, i]
    return y

# --- Métodos Diretos ---
def gauss_piv_parcial(A, b):
    A = np.array(A, dtype=float); b = np.array(b, dtype=float).reshape(-1, 1)
    n, m = np.shape(A); 
    if n != m: raise ValueError("A matriz A deve ser quadrada")
    Aa = np.concatenate((A, b), axis=1)

    for j in range(n - 1):
        i_rel = np.argmax(np.abs(Aa[j:, j])); pivo_linha = j + i_rel
        if np.abs(Aa[pivo_linha, j]) < 1e-15: raise ValueError("Sistema singular: pivô zero")
        if pivo_linha != j: Aa[[j, pivo_linha], :] = Aa[[pivo_linha, j], :]
        pivo = Aa[j, j]
        for i in range(j + 1, n):
            fator = Aa[i, j] / pivo; Aa[i, :] = Aa[i, :] - fator * Aa[j, :]

    if np.abs(Aa[n - 1, n - 1]) < 1e-15: raise ValueError("Sistema singular: pivô zero na última etapa.")
    U = Aa[:, :n]; y = Aa[:, n:]
    x = triangular_superior(U, y)
    return x, U, y

def gauss_piv_completo(A, b):
    A = np.array(A, dtype=float); b = np.array(b, dtype=float).reshape(-1, 1)
    n, m = np.shape(A); 
    if n != m: raise ValueError("A matriz A deve ser quadrada")
    Aa = np.concatenate((A, b), axis=1); perm = np.arange(n) 

    for j in range(n - 1):
        submat = np.abs(Aa[j:, j:n]); i_rel, k_rel = np.unravel_index(np.argmax(submat), submat.shape)
        i = j + i_rel; k = j + k_rel
        if np.abs(Aa[i, k]) < 1e-15: raise ValueError("Sistema singular: pivô zero")
        if i != j: Aa[[j, i], :] = Aa[[i, j], :]
        if k != j: 
            Aa[:, [j, k]] = Aa[:, [k, j]]; perm[[j, k]] = perm[[k, j]]
        pivo = Aa[j, j]
        for i in range(j + 1, n):
            fator = Aa[i, j] / pivo; Aa[i, :] = Aa[i, :] - fator * Aa[j, :]

    if np.abs(Aa[n - 1, n - 1]) < 1e-15: raise ValueError("Sistema singular: pivô zero na última etapa.")
    U = Aa[:, :n]; y = Aa[:, n:]
    x = triangular_superior(U, y)
    x_final = np.zeros_like(x)
    for i in range(n): x_final[perm[i], 0] = x[i, 0]
    return x_final, U, y, perm

def fatoracaoLU(A, b):
    A = np.array(A, dtype=float); b = np.array(b, dtype=float).reshape(-1, 1)
    n, m = np.shape(A); 
    if n != m: raise ValueError("A matriz A deve ser quadrada")
        
    U = A.copy(); L = np.eye(n); P = np.eye(n)
    
    for j in range(n): 
        i_rel = np.argmax(np.abs(U[j:, j])); pivo_linha_idx = j + i_rel
        
        if pivo_linha_idx != j:
            U[[j, pivo_linha_idx], :] = U[[pivo_linha_idx, j], :]
            L[[j, pivo_linha_idx], :j] = L[[pivo_linha_idx, j], :j]
            P[[j, pivo_linha_idx], :] = P[[pivo_linha_idx, j], :]
            
        pivo = U[j, j]
        if np.abs(pivo) < 1e-15: raise ValueError("Sistema singular: pivô zero.")
            
        if j < n - 1:
            for i in range(j + 1, n):
                fator = U[i, j] / pivo
                L[i, j] = fator
                U[i, :] = U[i, :] - fator * U[j, :]

    Pb = P @ b
    y = triangular_inferior(L, Pb)
    x = triangular_superior(U, y)
    return x, L, U, P

def fatoracao_cholesky(A):
    A = np.array(A, dtype=float); n, m = np.shape(A)
    if n != m: raise ValueError("A matriz A deve ser quadrada.")
    if not np.allclose(A, A.T): raise ValueError("Cholesky requer uma matriz simétrica.")

    L = np.zeros((n, n))
    for j in range(n):
        soma_diag = np.sum(L[j, 0:j] ** 2)
        termo_diag = A[j, j] - soma_diag
        
        if termo_diag <= 1e-15: raise ValueError("Matriz não é definida positiva (pivô <= 0). Cholesky falhou.")
        L[j, j] = np.sqrt(termo_diag)
        
        for i in range(j + 1, n):
            soma_nao_diag = np.sum(L[i, 0:j] * L[j, 0:j])
            L[i, j] = (A[i, j] - soma_nao_diag) / L[j, j]
    return L

def cholesky_solve(A, b):
    b = np.array(b, dtype=float).reshape(-1, 1)
    L = fatoracao_cholesky(A)
    y = triangular_inferior(L, b)
    U = L.T 
    x = triangular_superior(U, y)
    return x, L, U

# --- Métodos Iterativos ---
def gauss_jacobi(A, b, x0=None, tol=1e-5, max_iter=1000):
    A = np.array(A, dtype=float); b = np.array(b, dtype=float).reshape(-1, 1)
    n, m = np.shape(A); 
    if n != m: raise ValueError("A matriz A deve ser quadrada")
    diag_elements = np.diag(A)
    if np.any(np.abs(diag_elements) < 1e-15): raise ValueError("Pivô zero na diagonal.")
        
    x = np.zeros((n, 1)) if x0 is None else np.array(x0, dtype=float).reshape(-1, 1)
    x_next = np.zeros((n, 1))
    
    for k in range(max_iter):
        for i in range(n):
            soma = A[i, :] @ x[:, 0] - A[i, i] * x[i, 0]
            x_next[i, 0] = (b[i, 0] - soma) / A[i, i]
            
        diff_norm = np.linalg.norm(x_next - x)
        if diff_norm < tol: return x_next, k + 1
        x = x_next.copy()
        
    raise RuntimeError(f"Gauss-Jacobi não convergiu após {max_iter} iterações (Norma: {diff_norm:.2e}).")

def gauss_seidel(A, b, x0=None, tol=1e-5, max_iter=1000):
    A = np.array(A, dtype=float); b = np.array(b, dtype=float).reshape(-1, 1)
    n, m = np.shape(A);
    if n != m: raise ValueError("A matriz A deve ser quadrada")
    diag_elements = np.diag(A)
    if np.any(np.abs(diag_elements) < 1e-15): raise ValueError("Pivô zero na diagonal.")
        
    x = np.zeros((n, 1)) if x0 is None else np.array(x0, dtype=float).reshape(-1, 1)
    
    for k in range(max_iter):
        x_old = x.copy() 
        
        for i in range(n):
            soma = 0.0
            for j in range(n):
                if i != j: soma += A[i, j] * x[j, 0]
            x[i, 0] = (b[i, 0] - soma) / A[i, i]
            
        diff_norm = np.linalg.norm(x - x_old)
        if diff_norm < tol: return x, k + 1
        
    raise RuntimeError(f"Gauss-Seidel não convergiu após {max_iter} iterações (Norma: {diff_norm:.2e}).")