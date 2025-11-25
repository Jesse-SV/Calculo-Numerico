import sympy as sp

MAX_ITER = 100

# ===================== Funções globais =====================
x = sp.symbols('x')
func_expr = None
deriv_expr = None
phi_expr = None
precisao = None
arquivo = None

def func(val):
    return float(func_expr.subs(x, val))

def funclinha(val):
    return float(deriv_expr.subs(x, val))

def phi(val):
    return float(phi_expr.subs(x, val))

def condicaoParada(x0, x1):
    erroRelativo = abs(x1 - x0) / abs(x1) if x1 != 0 else abs(x1 - x0)
    if abs(x1 - x0) <= precisao or abs(func(x1)) <= precisao or erroRelativo <= precisao:
        arquivo.write("\nCritério de parada atingido (Precisao = {}):\n".format(precisao))
        arquivo.write("|{} - {}| = {}\n".format(x1, x0, abs(x1 - x0)))
        arquivo.write("|f({})| = {}\n".format(x1, abs(func(x1))))
        arquivo.write("Erro relativo = {}\n".format(erroRelativo))
        return True
    return False

# ===================== Métodos numéricos =====================
def bisseccao(a, b):
    arquivo.write("---------| MÉTODO DA BISSEÇÃO |---------\n")
    k = 0
    fa, fb = func(a), func(b)
    meio = None
    if fa * fb > 0:
        arquivo.write("Erro: f(a) e f(b) têm o mesmo sinal.\n\n")
        return
    while not condicaoParada(a, b) and k < MAX_ITER:
        k += 1
        meio = 0.5 * (a + b)
        fmeio = func(meio)
        arquivo.write(f"Iteração {k}: x = {meio}, f(x) = {fmeio}, intervalo = [{a}, {b}]\n")
        if fa * fmeio < 0:
            b, fb = meio, fmeio
        else:
            a, fa = meio, fmeio
    arquivo.write(f"Raiz aproximada: {meio} após {k} iterações.\n\n")

def newton(x0):
    arquivo.write("---------| MÉTODO DE NEWTON |---------\n")
    k = 0
    while k < MAX_ITER:
        fx, fxlinha = func(x0), funclinha(x0)
        if fxlinha == 0:
            arquivo.write(f"Erro: derivada zero em x = {x0}. Método de Newton não pode continuar.\n")
            return
        x1 = x0 - fx / fxlinha
        k += 1
        arquivo.write(f"Iteração {k}: x = {x1}\n")
        if condicaoParada(x0, x1):
            break
        x0 = x1
    arquivo.write(f"Raiz aproximada: {x1} após {k} iterações\n\n")

def mil(x0):
    arquivo.write("---------| MÉTODO ITERATIVO LINEAR (MIL) |---------\n")
    k = 0
    while k < MAX_ITER:
        x1 = phi(x0)
        k += 1
        arquivo.write(f"Iteração {k}: x = {x1}\n")
        if condicaoParada(x0, x1):
            break
        x0 = x1
    arquivo.write(f"Raiz aproximada: {x1} após {k} iterações\n\n")

def secante(x0, x1):
    arquivo.write("---------| MÉTODO DA SECANTE |---------\n")
    k = 0
    f0, f1 = func(x0), func(x1)
    while k < MAX_ITER:
        if f1 - f0 == 0:
            arquivo.write(f"Erro: divisão por zero na iteração {k+1}. Método da secante não pode continuar.\n")
            return
        x2 = x1 - (f1 * (x1 - x0)) / (f1 - f0)
        k += 1
        arquivo.write(f"Iteração {k}: x = {x2}\n")
        if condicaoParada(x1, x2):
            break
        x0, f0 = x1, f1
        x1, f1 = x2, func(x2)
    arquivo.write(f"Raiz aproximada: {x2} após {k} iterações\n\n")

def regulaFalsi(a, b):
    arquivo.write("---------| MÉTODO DA REGULA FALSI |---------\n")
    fa, fb = func(a), func(b)
    if fa * fb > 0:
        arquivo.write("Erro: f(a) e f(b) têm o mesmo sinal. Intervalo inválido.\n\n")
        return
    k = 0
    while k < MAX_ITER:
        if fb - fa == 0:
            arquivo.write(f"Erro: divisão por zero na iteração {k+1}. Método Regula Falsi não pode continuar.\n")
            return
        x = (a * fb - b * fa) / (fb - fa)
        fx = func(x)
        k += 1
        arquivo.write(f"Iteração {k}: x = {x}, f(x) = {fx}, intervalo = [{a}, {b}]\n")
        if condicaoParada(a, x):
            break
        if fa * fx < 0:
            b, fb = x, fx
        else:
            a, fa = x, fx
    arquivo.write(f"Raiz aproximada: {x} após {k} iterações\n\n")

# ===================== Função principal =====================
def main():
    global func_expr, deriv_expr, phi_expr, precisao, arquivo
    try:
        with open("entrada.txt", "r") as f:
            lines = f.read().splitlines()
            func_expr = sp.sympify(lines[0])
            deriv_expr = sp.sympify(lines[1])
            phi_expr = sp.sympify(lines[2])
            a, b, x0, x1, precisao = map(float, lines[3].split())
    except Exception as e:
        print("Erro ao ler arquivo de entrada:", e)
        return

    arquivo = open("saida.txt", "w")
    arquivo.write("PARÂMETROS DE ENTRADA:\n")
    arquivo.write(f"Intervalo inicial [a, b] = [{a}, {b}]\n")
    arquivo.write(f"Chute inicial x0 = {x0}\n")
    arquivo.write(f"Segundo ponto x1 = {x1}\n")
    arquivo.write(f"Precisão requerida = {precisao}\n\n")

    bisseccao(a, b)
    mil(x0)
    newton(x0)
    secante(x0, x1)
    regulaFalsi(a, b)

    arquivo.close()
    print("Cálculos concluídos. Resultados gravados em 'saida.txt'")

if __name__ == "__main__":
    main()
