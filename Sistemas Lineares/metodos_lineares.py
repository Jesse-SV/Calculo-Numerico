import numpy as np

# Retro substituição
def triangular_superior(U, y):
    n, nc = np.shape(U)
    if y.ndim == 1: y = y.reshape(-1, 1) 
    
    x = np.zeros((n, 1))
    
    # Loop de retro-substituição: começa da última linha (n-1) e vai até a primeira
    for i in range(n - 1, -1, -1): 
        
        # Verifica se o elemento da diagonal (o pivô U[i, i]) é próximo de zero
        if np.abs(U[i, i]) < 1e-15: 
            raise ValueError("Sistema singular: pivô zero na retro-substituição.")
            
        # Calcula o somatório: U[i, i+1]*x[i+1] + ... + U[i, n-1]*x[n-1]
        # O resultado é a multiplicação da linha i de U pelos x já calculados
        soma = U[i, i+1:n] @ x[i+1:n, 0] 
        
        # Calcula o valor da variável x[i] pela fórmula: x_i = (y_i - soma) / U_ii
        x[i, 0] = (y[i, 0] - soma) / U[i, i]
        
    return x

# Retro substituição também
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

def gauss(A, b):
    A = np.array(A, dtype=float); b = np.array(b, dtype=float).reshape(-1, 1)
    n, m = np.shape(A)
    if n != m: raise ValueError("A matriz A deve ser quadrada")

    Aa = np.concatenate((A, b), axis=1)

    for j in range(n - 1):
        pivo = Aa[j, j]
        
        # Teste de Pivô
        if np.abs(pivo) < 1e-15: 
            raise ValueError(f"Sistema singular: pivô zero (a[{j},{j}]) na eliminação sem pivoteamento. Considere usar pivoteamento.")
            
        # Eliminação
        for i in range(j + 1, n):
            # Calcula o fator de multiplicação
            fator = Aa[i, j] / pivo
            # Zera o elemento Aa[i, j] e atualiza toda a linha i
            Aa[i, :] = Aa[i, :] - fator * Aa[j, :]

    # Teste para o último pivô
    if np.abs(Aa[n - 1, n - 1]) < 1e-15: 
        raise ValueError("Sistema singular: pivô zero na última etapa da eliminação.")
        
    U = Aa[:, :n]
    y = Aa[:, n:]
    
    x = triangular_superior(U, y)
    
    return x, U, y

def gauss_piv_parcial(A, b):
    A = np.array(A, dtype=float); b = np.array(b, dtype=float).reshape(-1, 1)
    
    n, m = np.shape(A); 
    if n != m: raise ValueError("A matriz A deve ser quadrada")
        
    Aa = np.concatenate((A, b), axis=1)

    for j in range(n - 1):
        
        # Encontra a linha com o maior valor absoluto abaixo do pivô atual (coluna j).
        i_rel = np.argmax(np.abs(Aa[j:, j])); 
        pivo_linha = j + i_rel
        
        # Verifica se o pivo máximo encontrado é zero
        if np.abs(Aa[pivo_linha, j]) < 1e-15: raise ValueError("Sistema singular: pivô zero")
            
        # Troca a linha atual j pela linha do pivô máximo 
        if pivo_linha != j: Aa[[j, pivo_linha], :] = Aa[[pivo_linha, j], :]
        
        pivo = Aa[j, j]
        
        # Eliminação
        for i in range(j + 1, n):
            # Calcula o fator de multiplicação
            fator = Aa[i, j] / pivo
            # Zera o elemento Aa[i, j] e atualiza toda a linha i
            Aa[i, :] = Aa[i, :] - fator * Aa[j, :]

    # Checagem final do último pivô
    if np.abs(Aa[n - 1, n - 1]) < 1e-15: raise ValueError("Sistema singular: pivô zero na última etapa.")

    U = Aa[:, :n]; y = Aa[:, n:]
    
    # Resolve o sistema Ux = y usando a Retro-substituição
    x = triangular_superior(U, y)
    
    return x, U, y

def gauss_piv_completo(A, b):
    A = np.array(A, dtype=float); b = np.array(b, dtype=float).reshape(-1, 1)
    
    n, m = np.shape(A); 
    if n != m: raise ValueError("A matriz A deve ser quadrada")
        
    Aa = np.concatenate((A, b), axis=1)

    # Vetor de permutação de colunas
    perm = np.arange(n) 

    for j in range(n - 1):
        
        # Busca o maior elemento em valor absoluto na submatriz
        submat = np.abs(Aa[j:, j:n]); 
        i_rel, k_rel = np.unravel_index(np.argmax(submat), submat.shape)
        
        # Calcula os índices absolutos
        i = j + i_rel; k = j + k_rel
        
        if np.abs(Aa[i, k]) < 1e-15: raise ValueError("Sistema singular: pivô zero")
            
        # Troca de Linhas (Garante o maior pivô na posição (j, k))
        if i != j: Aa[[j, i], :] = Aa[[i, j], :]
        
        # Troca de Colunas (Move o maior pivô para a posição (j, j))
        if k != j: 
            Aa[:, [j, k]] = Aa[:, [k, j]]; # Troca colunas na matriz aumentada A.
            perm[[j, k]] = perm[[k, j]]    # Registra a troca no vetor de permutação de variáveis.
        
        pivo = Aa[j, j]
        
        # Eliminação
        for i in range(j + 1, n):
            # Calcula o fator de eliminação
            fator = Aa[i, j] / pivo 
            # Zera o elemento Aa[i, j] e atualiza a linha i
            Aa[i, :] = Aa[i, :] - fator * Aa[j, :]

    if np.abs(Aa[n - 1, n - 1]) < 1e-15: raise ValueError("Sistema singular: pivô zero na última etapa.")
        
    U = Aa[:, :n]; y = Aa[:, n:]
    
    # Resolve o sistema Ux = y usando a Retro-substituição
    x = triangular_superior(U, y)
    
    # Reordena a solução x para obter o vetor solução final x_final (desfaz a permutação de variáveis).
    x_final = np.zeros_like(x)
    for i in range(n): x_final[perm[i], 0] = x[i, 0]

    return x_final, U, y, perm

def fatoracaoLU(A, b):
    A = np.array(A, dtype=float); b = np.array(b, dtype=float).reshape(-1, 1)
    
    n, m = np.shape(A); 
    if n != m: raise ValueError("A matriz A deve ser quadrada")
        
    # Inicializa U (cópia de A), L (identidade)
    U = A.copy(); L = np.eye(n); P = np.eye(n)
    

    for j in range(n): 
        # Pivoteamento Parcial
        i_rel = np.argmax(np.abs(U[j:, j])); 
        pivo_linha_idx = j + i_rel
        
        # Se houver troca de linha 
        if pivo_linha_idx != j:
            # Troca as linhas de U
            U[[j, pivo_linha_idx], :] = U[[pivo_linha_idx, j], :]
            L[[j, pivo_linha_idx], :j] = L[[pivo_linha_idx, j], :j]
            P[[j, pivo_linha_idx], :] = P[[pivo_linha_idx, j], :]
            
        pivo = U[j, j] 
        if np.abs(pivo) < 1e-15: raise ValueError("Sistema singular: pivô zero.")
            
        # Fatoração
        if j < n - 1:
            # Laço para calcular os multiplicadores e zerar U abaixo do pivô
            for i in range(j + 1, n):
                fator = U[i, j] / pivo # Calcula o multiplicador
                L[i, j] = fator        # Armazena o multiplicador em L
                # Eliminação
                U[i, :] = U[i, :] - fator * U[j, :]

    # Permutação 
    Pb = P @ b

    # Retro-substituição
    y = triangular_inferior(L, Pb)
    x = triangular_superior(U, y)
    
    return x, L, U, P

def fatoracao_cholesky(A):
    A = np.array(A, dtype=float); n, m = np.shape(A)
    
    if n != m: raise ValueError("A matriz A deve ser quadrada.")
    if not np.allclose(A, A.T): raise ValueError("Cholesky requer uma matriz simétrica.")

    L = np.zeros((n, n))
    
    for j in range(n):
        
        # Cálculo da Diagonal L[j, j]
        soma_diag = np.sum(L[j, 0:j] ** 2)
        termo_diag = A[j, j] - soma_diag 
        
        if termo_diag <= 1e-15: raise ValueError("Matriz não é definida positiva (pivô <= 0). Cholesky falhou.")
            
        # Calcula o elemento da diagonal
        L[j, j] = np.sqrt(termo_diag)
        
        # Cálculo dos Elementos Abaixo da Diagonal
        for i in range(j + 1, n):
            # Soma dos produtos
            soma_nao_diag = np.sum(L[i, 0:j] * L[j, 0:j])
            # Calcula o elemento L[i, j] usando a fórmula de Cholesky.
            L[i, j] = (A[i, j] - soma_nao_diag) / L[j, j]
            
    return L

def cholesky_solve(A, b):
    b = np.array(b, dtype=float).reshape(-1, 1)
    
    # Calcula a fatoração L (L*L.T = A)
    L = fatoracao_cholesky(A)
    
    # Resolve o sistema Ly = b
    y = triangular_inferior(L, b)
    
    # Define a matriz triangular superior U como a transposta de L
    U = L.T 
    
    # Retro-substituição
    x = triangular_superior(U, y)
    
    return x, L, U

def criterio_convergencia_jacobi(A):
    n = A.shape[0]
    
    for i in range(n):
        soma_off_diag = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
        if np.abs(A[i, i]) <= soma_off_diag:
            return False
    return True

import numpy as np

def gauss_jacobi(A, b, x0=None, tol=1e-5, max_iter=1000):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)
    n, m = A.shape

    if n != m: raise ValueError("A matriz A deve ser quadrada")
    
    D = np.diag(np.diag(A))
    R = A - D
    
    diag_elements = np.diag(A)
    if np.any(np.abs(diag_elements) < 1e-15):
        raise ValueError("Pivô zero na diagonal. Não é possível dividir por zero.")
    
    if not criterio_convergencia_jacobi(A):
        raise RuntimeError("O critério de Dominância Diagonal Estrita não foi satisfeito. A convergência para Gauss-Jacobi não é garantida para esta matriz.")
        
    x = np.zeros((n, 1)) if x0 is None else np.array(x0, dtype=float).reshape(-1, 1)
    
    D_inv = np.linalg.inv(D) 
    
    for k in range(max_iter):
        
        x_next = D_inv @ (b - np.dot(R, x))
            
        diff_norm = np.linalg.norm(x_next - x)
        if diff_norm < tol: 
            return x_next.flatten(), k + 1
        
        x = x_next.copy()
        
    raise RuntimeError(f"Gauss-Jacobi não convergiu após {max_iter} iterações (Norma: {diff_norm:.2e}).")

def criterio_sassenfeld(A):
    n = A.shape[0]
    beta = np.zeros(n)
    
    for i in range(n):
        soma_termos_ant = 0.0
        for j in range(i):
            soma_termos_ant += np.abs(A[i, j]) * beta[j]
            
        soma_termos_post = np.sum(np.abs(A[i, i+1:]))
        
        # Cálculo do coeficiente beta[i]
        beta[i] = (soma_termos_ant + soma_termos_post) / np.abs(A[i, i])
        
    # O critério de Sassenfeld é satisfeito se o máximo de todos os beta_i for menor que 1
    if np.max(beta) < 1.0:
        return True
    else:
        return False
    
def gauss_seidel(A, b, x0=None, tol=1e-5, max_iter=1000):
    A = np.array(A, dtype=float); b = np.array(b, dtype=float).reshape(-1, 1)
    n, m = np.shape(A);
    
    if n != m: 
        raise ValueError("A matriz A deve ser quadrada")
        
    diag_elements = np.diag(A)
    # Requisito: Checa se há pivôs zero na diagonal
    if np.any(np.abs(diag_elements) < 1e-15): raise ValueError("Pivô zero na diagonal.")
        
    # Verifica a convergência
    if not criterio_sassenfeld(A):
          raise RuntimeError("O Critério de Sassenfeld não foi satisfeito (max(beta_i) >= 1). A convergência de Gauss-Seidel não é garantida para esta matriz.")
        
    # Inicializa o vetor solução x com x0 ou um vetor de zeros
    x = np.zeros((n, 1)) if x0 is None else np.array(x0, dtype=float).reshape(-1, 1)
    
    # Laço principal de iteração.
    for k in range(max_iter):
        x_old = x.copy()
        
        for i in range(n):
            soma = 0.0
            for j in range(n):
                if i != j: soma += A[i, j] * x[j, 0] # Usa os valores DE X MAIS RECENTES
            
            x[i, 0] = (b[i, 0] - soma) / A[i, i]
            
        # Critério de Parada
        diff_norm = np.linalg.norm(x - x_old)
        
        if diff_norm < tol: return x, k + 1
        
    raise RuntimeError(f"Gauss-Seidel não convergiu após {max_iter} iterações (Norma: {diff_norm:.2e}).")