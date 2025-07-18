#include <iostream>
#include <fstream>
#include <cmath>
#include <stdexcept>
#define MAX_ITER 100

using namespace std;

double func(double x) {
     return pow(x, 3) - 9 * x + 3;
}

double funclinha(double x) {
    return 3 * pow(x, 2) - 9;
}

double phi(double x) {
    return ((pow(x, 3)) / 9) + (1.0 / 3.0);
}

void bisseccao_Metodo(double a, double b, double precisao, ofstream &arquivo) {
    arquivo << "---------| MÉTODO DA BISSEÇÃO |---------" << endl;
    int k = 0;
    double meio, fmeio, fa = func(a);

    if (fa * func(b) > 0) {
        arquivo << "Erro: f(a) e f(b) têm o mesmo sinal. Método não aplicável no intervalo [" 
                << a << ", " << b << "]." << endl << endl;
        return;
    }

    while ((fabs(b - a) > precisao) && (k < MAX_ITER)) {
        k++;
        meio = (a + b) / 2;
        fmeio = func(meio);

        arquivo << "Iteração " << k << ": x = " << meio << ", f(x) = " << fmeio << ", intervalo = [" 
                << a << ", " << b << "]" << endl;

        if (fmeio == 0) {
            arquivo << "Raiz exata encontrada: " << meio << endl << endl;
            return;
        }

        if (fa * fmeio < 0)
            b = meio;
        else {
            a = meio;
            fa = fmeio;
        }
    }
    arquivo << "Raiz aproximada: " << (a + b) / 2 << " após " << k << " iterações" << endl << endl;
}

void newton_Metodo(double x0, double precisao, ofstream &arquivo) {
    arquivo << "---------| MÉTODO DE NEWTON |---------" << endl;
    int k = 0;
    double x1, fx, fxlinha;

    do {
        fx = func(x0);
        fxlinha = funclinha(x0);
        
        if (fabs(fxlinha) < 1e-12) {
            arquivo << "Erro: Derivada próxima de zero (" << fxlinha << "). Método falhou na iteração " 
                   << k << " com x = " << x0 << endl << endl;
            return;
        }

        x1 = x0 - fx / fxlinha;
        k++;

        arquivo << "Iteração " << k << ": x = " << x1 << ", f(x) = " << func(x1) 
                << ", |xₙ - xₙ₋₁| = " << fabs(x1 - x0) << endl;

        if (fabs(x1 - x0) < precisao || fabs(func(x1)) < precisao)
            break;

        x0 = x1;
    } while (k < MAX_ITER);

    arquivo << "Raiz aproximada: " << x1 << " após " << k << " iterações" << endl << endl;
}

void mil_Metodo(double x0, double precisao, ofstream &arquivo) {
    arquivo << "---------| MÉTODO ITERATIVO LINEAR (MIL) |---------" << endl;
    int k = 0;
    double x1;
    bool convergencia_garantida = fabs(3 * pow(x0, 2) / 9) < 1; // |φ'(x)| < 1?

    if (!convergencia_garantida) {
        arquivo << "Atenção: Convergência não garantida para x0 = " << x0 
                << " (|φ'(x0)| = " << fabs(3 * pow(x0, 2) / 9) << ")" << endl;
    }

    do {
        x1 = phi(x0);
        k++;

        arquivo << "Iteração " << k << ": x = " << x1 << ", f(x) = " << func(x1) 
                << ", |xₙ - xₙ₋₁| = " << fabs(x1 - x0) << endl;

        if (fabs(x1 - x0) < precisao)
            break;

        x0 = x1;
    } while (k < MAX_ITER);

    arquivo << "Raiz aproximada: " << x1 << " após " << k << " iterações" << endl << endl;
}

void secante_Metodo(double x0, double x1, double precisao, ofstream &arquivo) {
    arquivo << "---------| MÉTODO DA SECANTE |---------" << endl;
    int k = 0;
    double x2, f0, f1;

    if (fabs(x1 - x0) < 1e-12) {
        arquivo << "Erro: x0 e x1 muito próximos. Método não pode ser aplicado." << endl << endl;
        return;
    }

    f0 = func(x0);
    f1 = func(x1);

    do {
        if (fabs(f1 - f0) < 1e-12) {
            arquivo << "Erro: Divisão por zero na iteração " << k 
                   << ". Diferença entre f(x1) e f(x0) muito pequena." << endl << endl;
            return;
        }

        x2 = x1 - (f1 * (x1 - x0)) / (f1 - f0);
        k++;

        arquivo << "Iteração " << k << ": x = " << x2 << ", f(x) = " << func(x2) 
                << ", |xₙ - xₙ₋₁| = " << fabs(x2 - x1) << endl;

        if (fabs(x2 - x1) < precisao)
            break;

        x0 = x1; f0 = f1;
        x1 = x2; f1 = func(x1);
    } while (k < MAX_ITER);

    arquivo << "Raiz aproximada: " << x2 << " após " << k << " iterações" << endl << endl;
}

void regulaFalsi_Metodo(double a, double b, double precisao, ofstream &arquivo) {
    arquivo << "---------| MÉTODO DA REGULA FALSI |---------" << endl;
    double fa = func(a), fb = func(b), x, fx;
    int k = 0;

    if (fa * fb > 0) {
        arquivo << "Erro: f(a) e f(b) têm o mesmo sinal. Método não aplicável no intervalo [" 
                << a << ", " << b << "]." << endl << endl;
        return;
    }

    do {
        x = (a * fb - b * fa) / (fb - fa);
        fx = func(x);
        k++;

        arquivo << "Iteração " << k << ": x = " << x << ", f(x) = " << fx 
                << ", intervalo = [" << a << ", " << b << "]" << endl;

        if (fabs(fx) < precisao)
            break;

        if (fa * fx < 0) {
            b = x;
            fb = fx;
        } else {
            a = x;
            fa = fx;
        }
    } while (k < MAX_ITER);

    arquivo << "Raiz aproximada: " << x << " após " << k << " iterações" << endl << endl;
}

// ===================== Função Principal =====================
int main() {
    ifstream entrada("entrada.txt");
    ofstream arquivo("saida.txt");

    if (!entrada.is_open()) {
        cerr << "Erro: Não foi possível abrir o arquivo de entrada 'entrada.txt'" << endl;
        return 1;
    }

    if (!arquivo.is_open()) {
        cerr << "Erro: Não foi possível abrir o arquivo de saída 'saida.txt'" << endl;
        return 1;
    }

    double a, b, x0, x1, precisao;
    entrada >> a >> b >> x0 >> x1 >> precisao;

    if (entrada.fail()) {
        cerr << "Erro: Formato inválido no arquivo de entrada. Esperado: a b x0 x1 precisao" << endl;
        return 1;
    }

    arquivo << "PARÂMETROS DE ENTRADA:" << endl;
    arquivo << "Intervalo inicial [a, b] = [" << a << ", " << b << "]" << endl;
    arquivo << "Chute inicial x0 = " << x0 << endl;
    arquivo << "Segundo ponto x1 = " << x1 << endl;
    arquivo << "Precisão requerida = " << precisao << endl << endl;

    bisseccao_Metodo(a, b, precisao, arquivo);
    mil_Metodo(x0, precisao, arquivo);
    newton_Metodo(x0, precisao, arquivo);
    secante_Metodo(x0, x1, precisao, arquivo);
    regulaFalsi_Metodo(a, b, precisao, arquivo);

    cout << "Cálculos concluídos. Resultados gravados em 'saida.txt'" << endl;

    entrada.close();
    arquivo.close();
    return 0;
}