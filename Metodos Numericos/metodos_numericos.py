import sympy as sp

MAX_ITER = 100

# Formato do txt:
# função
# função_phi
# a b x0 x1 precisao

#Variáveis globais
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
    if abs(x1 - x0) <= precisao or abs(func(x1)) <= precisao:
        arquivo.write("\nCriterio de parada atingido (Precisao = {}):\n".format(precisao))
        arquivo.write("|{} - {}| = {}\n".format(x1, x0, abs(x1 - x0)))
        arquivo.write("|f({})| = {}\n".format(x1, abs(func(x1))))
        arquivo.write("\n")
        return True
    return False

# MÉTODO DA BISSECÇÃO (isso nao foi gerado por GPT)
# 1. Calcula o ponto médio entre de [a, b]
# 2. Verifica-se os sinais de f(m)
# 3. Dependendo do sinal de f(m), substitui-se o antigo intervalo pelo novo
# 4. Repita o processo até atingir o critério de parada
def bisseccao(a, b):
    arquivo.write("---------| METODO DA BISSECAO |---------\n")
    k = 0
    fa, fb = func(a), func(b)
    meio = None
    # Se não existir a troca de sinal, não existe um ponto entre a e b que é zero (raiz)
    if fa * fb > 0:
        arquivo.write("Erro: f(a) e f(b) tem o mesmo sinal.\n\n")
        return
    while not condicaoParada(a, b) and k < MAX_ITER:
        k += 1
        meio = 0.5 * (a + b)
        fmeio = func(meio)
        arquivo.write(f"Iteracao {k}: x = {meio}, f(x) = {fmeio}, intervalo = [{a}, {b}]\n")
        if fa * fmeio < 0:
            b, fb = meio, fmeio
        else:
            a, fa = meio, fmeio
    arquivo.write(f"Raiz aproximada: {meio} apos {k} iteracoes.\n\n")

# MÉTODO ITERATIVO LINEAR (agora nao pode comentar nos codigo pq é tudo GPT hj em dia)
# 1. Lê a função Phi
# 2. Utiliza um valor inicial x0
# 3. Calcula xk+1 = phi(xk)
# 4. Repita o processo até atingir o critério de parada
def mil(x0):
    arquivo.write("---------| METODO ITERATIVO LINEAR (MIL) |---------\n")
    k = 0
    if abs(func(x0)) < precisao:
        return
    while k < MAX_ITER:
        x1 = phi(x0)
        k += 1
        arquivo.write(f"Iteracao {k}: x = {x1}\n")
        if condicaoParada(x0, x1):
            break
        x0 = x1
    arquivo.write(f"Raiz aproximada: {x1} apos {k} iteracoes\n\n")

# MÉTODO DE NEWTON
# 1. Escolhe um valor inicial x0
# 2. Calcula f(x0) e f'(x0)
# 3. Calcula x1 = x0 - f(x0)/f'(x0)
# 4. Repita o processo até atingir o critério de parada
def newton(x0):
    arquivo.write("---------| METODO DE NEWTON |---------\n")
    k = 0
    while k < MAX_ITER:
        fx, fxlinha = func(x0), funclinha(x0)
        if fxlinha == 0:
            arquivo.write(f"Erro: derivada zero em {x0}\n")
            return
        x1 = x0 - fx/fxlinha
        k += 1
        arquivo.write(f"Iteracao {k}: x = {x1}\n")
        if condicaoParada(x0, x1):
            break
        x0 = x1
    arquivo.write(f"Raiz aproximada: {x1} apos {k} iteracoes\n\n")

# MÉTODO DA SECANTE
# 1. Escolhe dois valores iniciais x0 e x1
# 2. Calcula f(x0) e f(x1)
# 3. Calcula x2 = x1 - f(x1)*(x1 - x0)/(f(x1) - f(x0))
# 4. Repita o processo até atingir o critério de parada
def secante(x0, x1):
    arquivo.write("---------| METODO DA SECANTE |---------\n")
    k = 0
    f0, f1 = func(x0), func(x1)
    while k < MAX_ITER:
        if f1 - f0 == 0:
            arquivo.write(f"Erro: divisao por zero\n")
            return
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        k += 1
        arquivo.write(f"Iteracao {k}: x = {x2}\n")
        if condicaoParada(x1, x2):
            break
        x0, f0 = x1, f1
        x1, f1 = x2, func(x2)
    arquivo.write(f"Raiz aproximada: {x2} apos {k} iteracoes\n\n")

# MÉTODO DA REGULA FALSI
# 1. Escolhe um intervalo [a, b]
# 2. Calcula o ponto de interseção x = (a*fb - b*fa)/(fb - fa)
# 3. Verifica o critério de parada (|x - a| ou |f(x)|)
# 4. Atualiza o intervalo:
#    - se f(a)*f(x) < 0, então b = x
#    - senão, a = x
# 5. Repete o processo até convergir
def regulaFalsi(a, b):
    arquivo.write("---------| METODO DA REGULA FALSI |---------\n")
    fa, fb = func(a), func(b)
    if fa * fb > 0:
        arquivo.write("Erro: f(a) e f(b) tem o mesmo sinal\n\n")
        return
    k = 0
    while k < MAX_ITER:
        x = (a * fb - b * fa) / (fb - fa)
        fx = func(x)
        k += 1
        arquivo.write(f"Iteracao {k}: x = {x}, f(x) = {fx}, intervalo = [{a}, {b}]\n")
        if condicaoParada(a, x):
            break
        if fa * fx < 0:
            b, fb = x, fx
        else:
            a, fa = x, fx
    arquivo.write(f"Raiz aproximada: {x} apos {k} iteracoes\n\n")

def main():
    global func_expr, deriv_expr, phi_expr, precisao, arquivo
    try:
        with open("entrada.txt", "r") as f:
            lines = f.read().splitlines()
            locals_dict = {"log": lambda arg: sp.log(arg, 10)}

            # Função
            func_expr = sp.sympify(lines[0], locals=locals_dict)

            # Phi
            phi_expr = sp.sympify(lines[1], locals=locals_dict)

            #Calcula derivada
            deriv_expr = sp.diff(func_expr, x)

            #Lê os parâmetros 
            a, b, x0, x1, precisao = map(float, lines[2].split())

    except Exception as e:
        print("Erro ao ler entrada.txt:", e)
        return

    arquivo = open("saida.txt", "w")
    arquivo.write("FUNCAO: f(x) = {}\n".format(func_expr))
    arquivo.write("DERIVADA: f'(x) = {}\n".format(deriv_expr))
    arquivo.write("PHI(x): {}\n\n".format(phi_expr))
    arquivo.write(f"Intervalo [a,b] = [{a},{b}]\n")
    arquivo.write(f"x0 = {x0}, x1 = {x1}, precisao = {precisao}\n\n")

    bisseccao(a, b)
    mil(x0)
    newton(x0)
    secante(x0, x1)
    regulaFalsi(a, b)

    arquivo.close()
    print("Cálculo concluído. Veja saida.txt")

if __name__ == "__main__":
    main()