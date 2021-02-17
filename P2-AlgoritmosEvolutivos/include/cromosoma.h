#ifndef CROMOSOMA_H
#define CROMOSOMA_H

#include <vector>
#include <algorithm>
#include "datos.h"

using namespace std;

class Cromosoma {
public:
    vector<int> solution;
    int fitness;

    // Constructor
    Cromosoma();
    // Establece la semilla
    static void SetSemilla(int &semilla);
    // Genera un número en un rango
    static unsigned GenerateNumber(unsigned low, unsigned high);
    // Calcula el coste
    void CalculateFitness(Datos &datos);
    // Calcula la diferencia de costes. Un coste negativo -> la solución actual es peor.
    int CalculateDifFitness(Datos &datos, unsigned i1, unsigned i2);
    // Inicializa la solución
    void Init(Datos &datos);
    // Devuelve un cromosoma donde los dos genes han sido intercambiados
    Cromosoma SimulateSwapGens(Datos &datos, unsigned gen1, unsigned gen2);
    // Intercambia dos genes
    void SwapGens(unsigned gen1, unsigned gen2);
    // Imprime la información del cromosoma
    void Show();
};

#endif
