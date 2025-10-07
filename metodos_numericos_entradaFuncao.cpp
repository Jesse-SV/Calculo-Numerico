#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <stdexcept>
#include "muParser.h"   // Inclua o cabeçalho do muParser
#define MAX_ITER 100

using namespace std;
using namespace mu;

// Variáveis globais
ofstream arquivo;
double precisao;
Parser parserFunc, parserPhi;
string funcExpr, phiExpr;
double x;

// ================== Funções utilitárias ==================
double func(double val) {
    x = val;
    return parserFunc.Eval();
}

double phi(double val) {
    x = val;
    return parserPhi.Eval();
}

// Derivada numérica de f(x)
double funclinha(double val) {
    double h = 1e-6;
    return (func(val + h) - func(val - h)) / (2 * h);
}

bool condicaoParada(double x0, double x1) {
    if ((fabs(x1 - x0) <= precisao) ||
        (fabs(func(x1)) <= precisao) ||
        (fabs(x1 - x0) / fabs(x1) <= precisao)) {

        arquivo << endl;
        arquivo << "Critério de parada atingido (Precisao = " << precisao << "):" << endl;
        arquivo << "|" << x1 << " - " << x0 << "| = " << fabs(x1 - x0) << endl;
        arquivo << "|f(" << x1 << ")| = " << fabs(func(x1)) << endl;
        arquivo << "Erro relativo = " << fabs(x1 - x0) / fabs(x1) << endl;
        return true;
    }
    return false;
}

// ================== Métodos Numéricos ==================
void bisseccao(double a, double b) {
    arquivo << "---------| MÉTODO DA BISSEÇÃO |---------" << endl;
    int k = 0;
    double meio = 0.0, fmeio = 0.0;
    double fa = func(a), fb = func(b);

    if (fa * fb > 0.0) {
        arquivo << "Erro: f(a) e f(b) têm o mesmo sinal." << endl << endl;
        return;
    }

    while (!condicaoParada(a, b) && k < MAX_ITER) {
        k++;
        meio = 0.5 * (a + b);
        fmeio = func(meio);

        arquivo << "Iteração " << k << ": x = " << meio
                << ", f(x) = " << fmeio
                << ", intervalo = [" << a << ", " << b << "]" << endl;

        if (fa * fmeio < 0.0) {
            b = meio;
            fb = fmeio;
        } else {
            a = meio;
            fa = fmeio;
        }
    }

    arquivo << "Raiz aproximada: " << meio
            << " após " << k << " iterações." << endl << endl;
}

void newton(double x0) {
    arquivo << "---------| MÉTODO DE NEWTON |---------" << endl;
    int k = 0;
    double x1, fx, fxlinha;

    do {
        fx = func(x0);
        fxlinha = funclinha(x0);
        x1 = x0 - fx / fxlinha;
        k++;

        arquivo << "Iteração " << k
                << ": x = " << x1
                << ", f(x) = " << func(x1)
                << ", |xₙ - xₙ₋₁| = " << fabs(x1 - x0) << endl;

        if (condicaoParada(x0, x1))
            break;

        x0 = x1;
    } while (k < MAX_ITER);

    arquivo << "Raiz aproximada: " << x1
            << " após " << k << " iterações" << endl << endl;
}

void mil(double x0) {
    arquivo << "---------| MÉTODO ITERATIVO LINEAR (MIL) |---------" << endl;
    int k = 0;
    double x1;

    do {
        x1 = phi(x0);
        k++;

        arquivo << "Iteração " << k
                << ": x = " << x1
                << ", f(x) = " << func(x1)
                << ", |xₙ - xₙ₋₁| = " << fabs(x1 - x0) << endl;

        if (condicaoParada(x0, x1))
            break;

        x0 = x1;
    } while (k < MAX_ITER);

    arquivo << "Raiz aproximada: " << x1
            << " após " << k << " iterações" << endl << endl;
}

void secante(double x0, double x1) {
    arquivo << "---------| MÉTODO DA SECANTE |---------" << endl;
    int k = 0;
    double x2, f0, f1;

    f0 = func(x0);
    f1 = func(x1);

    do {
        x2 = x1 - (f1 * (x1 - x0)) / (f1 - f0);
        k++;

        arquivo << "Iteração " << k
                << ": x = " << x2
                << ", f(x) = " << func(x2)
                << ", |xₙ - xₙ₋₁| = " << fabs(x2 - x1) << endl;

        if (condicaoParada(x1, x2))
            break;

        x0 = x1; f0 = f1;
        x1 = x2; f1 = func(x1);
    } while (k < MAX_ITER);

    arquivo << "Raiz aproximada: " << x2
            << " após " << k << " iterações" << endl << endl;
}

void regulaFalsi(double a, double b) {
    arquivo << "---------| MÉTODO DA REGULA FALSI |---------" << endl;
    double fa = func(a), fb = func(b), x, fx;
    int k = 0;

    if (fa * fb > 0) {
        arquivo << "Erro: f(a) e f(b) têm o mesmo sinal. Intervalo inválido." << endl << endl;
        return;
    }

    do {
        x = (a * fb - b * fa) / (fb - fa);
        fx = func(x);
        k++;

        arquivo << "Iteração " << k
                << ": x = " << x
                << ", f(x) = " << fx
                << ", intervalo = [" << a << ", " << b << "]" << endl;

        if (condicaoParada(a, x))
            break;

        if (fa * fx < 0) {
            b = x;
            fb = fx;
        } else {
            a = x;
            fa = fx;
        }
    } while (k < MAX_ITER);

    arquivo << "Raiz aproximada: " << x
            << " após " << k << " iterações" << endl << endl;
}

// ================== Função principal ==================
int main() {
    ifstream entrada("entrada.txt");
    arquivo.open("saida.txt");

    if (!entrada.is_open()) {
        cerr << "Erro: Não foi possível abrir 'entrada.txt'" << endl;
        return 1;
    }

    entrada >> funcExpr >> phiExpr;
    double a, b, x0, x1;
    entrada >> a >> b >> x0 >> x1 >> precisao;

    if (entrada.fail()) {
        cerr << "Erro: Formato inválido no arquivo de entrada." << endl;
        return 1;
    }

    // Configura os parsers
    parserFunc.DefineVar("x", &x);
    parserPhi.DefineVar("x", &x);
    parserFunc.SetExpr(funcExpr);
    parserPhi.SetExpr(phiExpr);

    arquivo << "FUNÇÃO LIDA: f(x) = " << funcExpr << endl;
    arquivo << "FUNÇÃO PHI: φ(x) = " << phiExpr << endl;
    arquivo << "Intervalo: [" << a << ", " << b << "], x0 = " << x0
            << ", x1 = " << x1 << ", precisão = " << precisao << endl << endl;

    bisseccao(a, b);
    mil(x0);
    newton(x0);
    secante(x0, x1);
    regulaFalsi(a, b);

    cout << "Cálculos concluídos. Resultados gravados em 'saida.txt'" << endl;

    return 0;
}
