import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import time
import os

from metodos_lineares import *
from metodos_numericos import *


# Interface
class LinearSolverApp:
    def __init__(self, master):
        self.master = master
        master.title("Solucionador de Métodos Numéricos")
        
        self.A = None
        self.b = None
        self.n = 0
        
        self.notebook = ttk.Notebook(master)
        self.notebook.pack(pady=10, padx=10, expand=True, fill="both")

        self.tab_linear = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_linear, text="1️⃣ Sistemas Lineares Ax=b")
        self.create_linear_tab(self.tab_linear)

        self.tab_raizes = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_raizes, text="2️⃣ Zeros de Funções f(x)=0")
        self.create_outros_tab(self.tab_raizes)

    
    def load_data_from_file(self, mode):
        filepath = filedialog.askopenfilename(
            defaultextension=".txt",
            filetypes=[("Arquivos de Dados", "*.txt *.csv"), ("Todos os arquivos", "*.*")]
        )
        if not filepath: return
            
        try:
            data = np.loadtxt(filepath, dtype=float)
            
            if mode == 'A':
                if data.ndim != 2 or data.shape[0] != data.shape[1]: raise ValueError("A matriz A deve ser quadrada.")
                self.A = data; self.n = self.A.shape[0]
                messagebox.showinfo("Sucesso", f"Matriz A ({self.n}x{self.n}) carregada.")
                
            elif mode == 'b':
                if data.ndim == 1: data = data.reshape(-1, 1)
                if self.A is not None and data.shape[0] != self.A.shape[0]:
                    raise ValueError(f"O vetor b ({data.shape[0]} linhas) não corresponde à ordem da matriz A ({self.A.shape[0]} linhas).")
                self.b = data
                if self.A is None: self.n = self.b.shape[0]
                messagebox.showinfo("Sucesso", f"Vetor b ({self.b.shape[0]}x1) carregado.")
                
            elif mode == 'A|b':
                if data.ndim != 2: raise ValueError("O arquivo deve conter a matriz aumentada [A|b] em formato 2D.")
                self.A = data[:, :-1]; self.b = data[:, -1].reshape(-1, 1)
                if self.A.shape[0] != self.A.shape[1]: raise ValueError("A matriz A deve ser quadrada.")
                self.n = self.A.shape[0]
                messagebox.showinfo("Sucesso", f"Matriz A e Vetor b (Ordem {self.n}) carregados.")
                
        except Exception as e:
            self.A = None; self.b = None; self.n = 0
            messagebox.showerror("Erro ao Carregar Dados", str(e))
            
        self.update_matrix_display()

    def update_matrix_display(self):
        # Limpa o campo de visualização
        self.matrix_display.delete(1.0, tk.END) 
        
        # --- Lógica de Status e Botão ---
        if self.A is not None and self.b is not None and self.A.shape[0] == self.b.shape[0]:
            status_text = f"Sistema Carregado (Ordem {self.n}x{self.n})\nPronto para resolver."
            self.solve_button.config(state=tk.NORMAL)
            
            # --- Exibição da Matriz A e B ---
            A_str = np.array2string(self.A, precision=4, suppress_small=True, max_line_width=80)
            b_str = np.array2string(self.b, precision=4, suppress_small=True, max_line_width=80)
            
            display_content = (
                f"Matriz A ({self.n}x{self.n}):\n{A_str}\n\n"
                f"Vetor b ({self.n}x1):\n{b_str}"
            )
            self.matrix_display.insert(tk.END, display_content)
            
        elif self.A is not None or self.b is not None:
            status_text = "Sistema Incompleto. Carregue A e b."
            self.solve_button.config(state=tk.DISABLED)
        else:
            status_text = "Nenhum Sistema Linear Carregado."
            self.solve_button.config(state=tk.DISABLED)
            
        self.status_label.config(text=status_text)
        

    def run_solver(self):
        if self.A is None or self.b is None:
            messagebox.showerror("Erro", "Carregue a matriz A e o vetor b primeiro.")
            return

        method_name = self.method_var.get()
        is_iterative = method_name in ["Gauss-Jacobi", "Gauss-Seidel"]
        
        try:
            if is_iterative:
                tol = float(self.tol_entry.get())
                max_iter = int(self.max_iter_entry.get())
                x0_str = self.x0_entry.get().strip()
                x0 = None
                if x0_str:
                    x0 = np.array([float(val) for val in x0_str.split(',') if val]).reshape(-1, 1)
                    if x0.shape[0] != self.n:
                        raise ValueError(f"Tamanho de x0 ({x0.shape[0]}) deve ser igual à ordem da matriz ({self.n}).")
            
            start_time = time.time()

            if method_name == "Gauss":
                x, U, y = gauss(self.A, self.b)
                output = f"Matriz U:\n{np.array2string(U, precision=6)}\n\nVetor y:\n{np.array2string(y.T, precision=6)}"
            elif method_name == "Gauss (Piv. Parcial)":
                x, U, y = gauss_piv_parcial(self.A, self.b)
                output = f"Matriz U:\n{np.array2string(U, precision=6)}\n\nVetor y:\n{np.array2string(y.T, precision=6)}"
            elif method_name == "Gauss (Piv. Completo)":
                x, U, y, perm = gauss_piv_completo(self.A, self.b)
                output = f"Matriz U:\n{np.array2string(U, precision=6)}\n\nVetor y:\n{np.array2string(y.T, precision=6)}\n\nPermutação de Variáveis:\n{perm}"
            elif method_name == "Fatoração LU":
                x, L, U, P = fatoracaoLU(self.A, self.b)
                output = f"Matriz L:\n{np.array2string(L, precision=6)}\n\nMatriz U:\n{np.array2string(U, precision=6)}\n\nMatriz P:\n{np.array2string(P, precision=6)}"
            elif method_name == "Cholesky":
                x, L, U = cholesky_solve(self.A, self.b)
                output = f"Matriz L (Inferior):\n{np.array2string(L, precision=6)}\n\nMatriz U (L^T - Superior):\n{np.array2string(U, precision=6)}"
            elif method_name == "Gauss-Jacobi":
                x, iterations = gauss_jacobi(self.A, self.b, x0=x0, tol=tol, max_iter=max_iter)
                output = f"Critério: ||x^(k+1) - x^(k)|| < {tol:.1e}\nIterações: {iterations} / {max_iter}"
            elif method_name == "Gauss-Seidel":
                x, iterations = gauss_seidel(self.A, self.b, x0=x0, tol=tol, max_iter=max_iter)
                output = f"Critério: ||x^(k+1) - x^(k)|| < {tol:.1e}\nIterações: {iterations} / {max_iter}"
            else:
                raise ValueError("Método selecionado inválido.")

            end_time = time.time()
            execution_time = end_time - start_time
            
            sol_display = "Solução x (Transposta):\n" + np.array2string(x.T, precision=8, separator=', ', suppress_small=True)
            time_display = f"Tempo de Execução: {execution_time:.6f} segundos"
            
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"===  {method_name} ===\n\n")
            self.result_text.insert(tk.END, sol_display + "\n\n")
            self.result_text.insert(tk.END, output + "\n\n")
            self.result_text.insert(tk.END, time_display + "\n")

        except Exception as e:
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"=== ERRO NA EXECUÇÃO ({method_name}) ===\n\n")
            self.result_text.insert(tk.END, f"Falha ao resolver o sistema:\n{str(e)}")
            messagebox.showerror("Erro de Cálculo", str(e))

    def create_linear_tab(self, tab):
        load_frame = ttk.LabelFrame(tab, text="Carregar Sistema Linear (Ax=b)")
        load_frame.pack(pady=10, padx=10, fill="x")

        ttk.Button(load_frame, text="Carregar Matriz A", command=lambda: self.load_data_from_file('A')).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(load_frame, text="Carregar Vetor b", command=lambda: self.load_data_from_file('b')).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(load_frame, text="Carregar A|b Juntos", command=lambda: self.load_data_from_file('A|b')).pack(side=tk.LEFT, padx=5, pady=5)
        
        self.status_label = ttk.Label(load_frame, text="Nenhum Sistema Linear Carregado.", foreground="red")
        self.status_label.pack(side=tk.RIGHT, padx=10)

        matrix_disp_frame = ttk.LabelFrame(tab, text="Matrizes A e b Recebidas")
        matrix_disp_frame.pack(pady=5, padx=10, fill="x")
        
        scrollbar = ttk.Scrollbar(matrix_disp_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.matrix_display = tk.Text(matrix_disp_frame, wrap=tk.WORD, height=8, font=('Consolas', 10), yscrollcommand=scrollbar.set)
        self.matrix_display.pack(fill="both", expand=True, padx=5, pady=5)
        scrollbar.config(command=self.matrix_display.yview)

        solver_frame = ttk.Frame(tab)
        solver_frame.pack(pady=10, padx=10, fill="x")

        method_group = ttk.LabelFrame(solver_frame, text="Método de Solução")
        method_group.pack(side=tk.LEFT, padx=10, fill="y")
        
        self.methods = [
            "Gauss", "Gauss (Piv. Parcial)", "Gauss (Piv. Completo)", "Fatoração LU", "Cholesky",
            "Gauss-Jacobi", "Gauss-Seidel"
        ]
        self.method_var = tk.StringVar(method_group); self.method_var.set(self.methods[0])
        ttk.Label(method_group, text="Escolha:").pack(pady=5, padx=10)
        method_menu = ttk.Combobox(method_group, textvariable=self.method_var, values=self.methods, state="readonly"); method_menu.pack(pady=5, padx=10)

        iter_group = ttk.LabelFrame(solver_frame, text="Parâmetros Iterativos")
        iter_group.pack(side=tk.LEFT, padx=10, fill="y")

        ttk.Label(iter_group, text="Tolerância (e.g., 1e-5):").pack(pady=2, padx=5)
        self.tol_entry = ttk.Entry(iter_group, width=15); self.tol_entry.insert(0, "1e-5"); self.tol_entry.pack(pady=2, padx=5)

        ttk.Label(iter_group, text="Máx. Iterações:").pack(pady=2, padx=5)
        self.max_iter_entry = ttk.Entry(iter_group, width=15); self.max_iter_entry.insert(0, "1000"); self.max_iter_entry.pack(pady=2, padx=5)
        
        ttk.Label(iter_group, text="Chute Inicial x0 (csv):").pack(pady=2, padx=5)
        self.x0_entry = ttk.Entry(iter_group, width=25); self.x0_entry.insert(0, "0.0"); self.x0_entry.pack(pady=2, padx=5)
        
        self.solve_button = ttk.Button(solver_frame, text="Executar Método", command=self.run_solver, state=tk.DISABLED)
        self.solve_button.pack(side=tk.LEFT, padx=10, fill="y")

        result_frame = ttk.LabelFrame(tab, text="Resultados")
        result_frame.pack(pady=10, padx=10, fill="both", expand=True)

        self.result_text = tk.Text(result_frame, wrap=tk.WORD, height=15, width=80, font=('Consolas', 10))
        self.result_text.pack(fill="both", expand=True, padx=5, pady=5)

    
    # Aba dos métodos numéricos 
    def run_raiz_solver(self):
        method_name = self.raiz_method_var.get()
        
        try:
            # Parsing de Expressões
            safe_dict = {'x': None, 'np': np, 'sin': np.sin, 'cos': np.cos, 
                         'tan': np.tan, 'exp': np.exp, 'log': np.log, 'sqrt': np.sqrt, 
                         'abs': np.abs, 'e': np.e, 'pi': np.pi, 'cbrt': np.cbrt}
            
            f_expr = self.fx_entry.get(); f = lambda x: eval(f_expr, safe_dict.copy(), {'x': x})
            f_linha_expr = self.f_linha_entry.get(); f_linha = lambda x: eval(f_linha_expr, safe_dict.copy(), {'x': x})
            phi_expr = self.phi_entry.get(); phi = lambda x: eval(phi_expr, safe_dict.copy(), {'x': x})

            tol = float(self.raiz_tol_entry.get())
            max_iter = int(self.raiz_max_iter_entry.get())
            
            start_vals_str = [val.strip() for val in self.start_values_entry.get().split(',')]
            start_vals = [float(val) for val in start_vals_str if val]
            
            if not start_vals: raise ValueError("Forneça pelo menos um valor inicial (x0 ou [a, b]).")

            start_time = time.time()
            
            if method_name in ["Bisseção", "Regula Falsi"]:
                if len(start_vals) < 2: raise ValueError(f"O método {method_name} requer dois valores (a, b) para o intervalo.")
                a, b = start_vals[0], start_vals[1]
                raiz, iteracoes = bisseccao(f, a, b, tol, max_iter) if method_name == "Bisseção" else regula_falsi(f, a, b, tol, max_iter)
                
            elif method_name in ["Newton", "Iterativo Linear (MIL)"]:
                if len(start_vals) < 1: raise ValueError(f"O método {method_name} requer um valor inicial (x0).")
                x0 = start_vals[0]
                raiz, iteracoes = newton(f, f_linha, x0, tol, max_iter) if method_name == "Newton" else mil(f, phi, x0, tol, max_iter)
                
            elif method_name == "Secante":
                if len(start_vals) < 2: raise ValueError("O método da Secante requer dois valores iniciais (x0, x1).")
                x0, x1 = start_vals[0], start_vals[1]
                raiz, iteracoes = secante(f, x0, x1, tol, max_iter)
            else:
                raise ValueError("Método de Raiz selecionado inválido.")

            end_time = time.time()
            execution_time = end_time - start_time
            
            self.raiz_result_text.delete(1.0, tk.END)
            self.raiz_result_text.insert(tk.END, f"=== {method_name} ===\n\n")
            self.raiz_result_text.insert(tk.END, f"Função f(x) utilizada: {f_expr}\n")
            self.raiz_result_text.insert(tk.END, f"Raiz Aproximada: {raiz:.10f}\n")
            self.raiz_result_text.insert(tk.END, f"f({raiz:.10f}) ≈ {f(raiz):.2e}\n")
            self.raiz_result_text.insert(tk.END, f"Iterações: {iteracoes} / {max_iter}\n")
            self.raiz_result_text.insert(tk.END, f"Tempo de Execução: {execution_time:.6f} segundos\n")

        except Exception as e:
            self.raiz_result_text.delete(1.0, tk.END)
            self.raiz_result_text.insert(tk.END, f"=== ERRO NA EXECUÇÃO DE {method_name} ===\n\n")
            self.raiz_result_text.insert(tk.END, f"Erro: {str(e)}")
            messagebox.showerror("Erro de Cálculo de Raiz", str(e))

    def create_outros_tab(self, tab):
        func_frame = ttk.LabelFrame(tab, text="Definição das Funções (Use 'x' como variável)")
        func_frame.pack(pady=10, padx=10, fill="x")

        ttk.Label(func_frame, text="f(x) =").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.fx_entry = ttk.Entry(func_frame, width=50); self.fx_entry.insert(0, "x**2 - 2"); self.fx_entry.grid(row=0, column=1, padx=5, pady=5, sticky="we")

        ttk.Label(func_frame, text="f'(x) =").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.f_linha_entry = ttk.Entry(func_frame, width=50); self.f_linha_entry.insert(0, "2 * x"); self.f_linha_entry.grid(row=1, column=1, padx=5, pady=5, sticky="we")
        
        ttk.Label(func_frame, text="φ(x) =").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.phi_entry = ttk.Entry(func_frame, width=50); self.phi_entry.insert(0, "2 / x"); self.phi_entry.grid(row=2, column=1, padx=5, pady=5, sticky="we")
        
        ttk.Label(func_frame, text="Funções aceitas: np.sin(), np.exp(), x**2, etc.").grid(row=3, column=1, padx=5, pady=5, sticky="w")


        param_method_frame = ttk.Frame(tab)
        param_method_frame.pack(pady=10, padx=10, fill="x")

        param_group = ttk.LabelFrame(param_method_frame, text="Parâmetros de Raiz")
        param_group.pack(side=tk.LEFT, padx=10, fill="y")
        
        ttk.Label(param_group, text="Valores Iniciais (csv, e.g., '1' ou '1, 2'):").pack(pady=2, padx=5)
        self.start_values_entry = ttk.Entry(param_group, width=25); self.start_values_entry.insert(0, "1, 2"); self.start_values_entry.pack(pady=2, padx=5)

        ttk.Label(param_group, text="Tolerância (tol):").pack(pady=2, padx=5)
        self.raiz_tol_entry = ttk.Entry(param_group, width=15); self.raiz_tol_entry.insert(0, "1e-8"); self.raiz_tol_entry.pack(pady=2, padx=5)

        ttk.Label(param_group, text="Máx. Iterações:").pack(pady=2, padx=5)
        self.raiz_max_iter_entry = ttk.Entry(param_group, width=15); self.raiz_max_iter_entry.insert(0, "50"); self.raiz_max_iter_entry.pack(pady=2, padx=5)

        method_group = ttk.LabelFrame(param_method_frame, text="Métodos de Zeros de Funções")
        method_group.pack(side=tk.LEFT, padx=10, fill="y")
        
        self.raiz_methods = ["Bisseção", "Regula Falsi", "Secante", "Newton", "Iterativo Linear (MIL)"]
        self.raiz_method_var = tk.StringVar(method_group); self.raiz_method_var.set(self.raiz_methods[0]) 
        
        method_menu = ttk.Combobox(method_group, textvariable=self.raiz_method_var, 
                                   values=self.raiz_methods, state="readonly", width=20); method_menu.pack(pady=5, padx=10)
        
        ttk.Button(method_group, text="Encontrar Raiz", command=self.run_raiz_solver).pack(pady=10, padx=10)

        raiz_result_frame = ttk.LabelFrame(tab, text="Resultados da Raiz")
        raiz_result_frame.pack(pady=10, padx=10, fill="both", expand=True)

        self.raiz_result_text = tk.Text(raiz_result_frame, wrap=tk.WORD, height=10, width=80, font=('Consolas', 10))
        self.raiz_result_text.pack(fill="both", expand=True, padx=5, pady=5)

if __name__ == "__main__":
    root = tk.Tk()
    app = LinearSolverApp(root)

    root.mainloop()
