import numpy as np

def condicao_parada_raiz(x_atual, x_anterior, f, tol):
    if np.abs(x_atual - x_anterior) <= tol or np.abs(f(x_atual)) <= tol:
        return True
    return False

def bisseccao(f, a, b, tol, max_iter):
    fa, fb = f(a), f(b)
    if fa * fb > 0: raise ValueError("f(a) e f(b) têm o mesmo sinal.")
    
    k = 0
    x_anterior = (a + b) / 2.0
    
    while k < max_iter:
        k += 1
        meio = 0.5 * (a + b); fmeio = f(meio)
        
        if condicao_parada_raiz(meio, x_anterior, f, tol): return meio, k
            
        x_anterior = meio
            
        if fa * fmeio < 0: b, fb = meio, fmeio
        else: a, fa = meio, fmeio
            
    raise RuntimeError(f"Bisseccao não convergiu após {max_iter} iterações.")


def mil(f, phi, x0, tol, max_iter):
    x_atual = x0; k = 0
    
    if np.abs(f(x0)) < tol: return x0, 0
        
    while k < max_iter:
        x_proximo = phi(x_atual); k += 1
        
        if condicao_parada_raiz(x_proximo, x_atual, f, tol): return x_proximo, k
            
        if np.abs(x_proximo) > 1e15 and k > 1: raise RuntimeError("MIL está divergindo rapidamente.")
             
        x_atual = x_proximo
        
    raise RuntimeError(f"MIL não convergiu após {max_iter} iterações.")


def newton(f, f_linha, x0, tol, max_iter):

    x_atual = x0; k = 0
    
    while k < max_iter:
        fx, fxlinha = f(x_atual), f_linha(x_atual)
        
        if np.abs(fxlinha) < 1e-15: raise ValueError(f"Erro: Derivada (f'(x)) é zero em x = {x_atual:.10f}.")
            
        x_proximo = x_atual - fx / fxlinha; k += 1
        
        if condicao_parada_raiz(x_proximo, x_atual, f, tol): return x_proximo, k
            
        x_atual = x_proximo
        
    raise RuntimeError(f"Newton não convergiu após {max_iter} iterações.")


def secante(f, x0, x1, tol, max_iter):
    f0, f1 = f(x0), f(x1); k = 0
    
    while k < max_iter:
        if np.abs(f1 - f0) < 1e-15: raise ValueError("Erro: Divisão por zero.")
            
        x_proximo = x1 - f1 * (x1 - x0) / (f1 - f0); k += 1
        
        if condicao_parada_raiz(x_proximo, x1, f, tol): return x_proximo, k
            
        x0, f0 = x1, f1
        x1, f1 = x_proximo, f(x_proximo)
        
    raise RuntimeError(f"Secante não convergiu após {max_iter} iterações.")


def regula_falsi(f, a, b, tol, max_iter):
    fa, fb = f(a), f(b)
    
    if fa * fb > 0: raise ValueError("f(a) e f(b) têm o mesmo sinal.")
        
    k = 0
    x_proximo = (a * fb - b * fa) / (fb - fa)
    
    while k < max_iter:
        x_anterior = x_proximo
        
        if np.abs(fb - fa) < 1e-15: raise ValueError("Erro: Denominador muito pequeno.")
            
        x_proximo = (a * fb - b * fa) / (fb - fa); fx = f(x_proximo); k += 1
        
        if condicao_parada_raiz(x_proximo, x_anterior, f, tol): return x_proximo, k
            
        if fa * fx < 0: b, fb = x_proximo, fx
        else: a, fa = x_proximo, fx
            
    raise RuntimeError(f"Regula Falsi não convergiu após {max_iter} iterações.")