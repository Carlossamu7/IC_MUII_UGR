#ifndef GENETICOS_H
#define GENETICOS_H

#include "cromosoma.h"
#include "datos.h"
#include "busquedaLocal.h"
#include "greedy.h"

class Geneticos {
  public:
    vector<Cromosoma> population;
    Datos datos;
    float pCross, pMut;
    unsigned crossNumber, ngens, iters;
    unsigned mode;  // 0:standard, 1:baldwin, 2:lamarck.
    Greedy algGreedy;

    // Constructor
    Geneticos(Datos &dat, float pCr, float pMu, unsigned size, unsigned mod, bool initGreedy);
    // Ordenamos la población teniendo el mejor en la primera posición
    void Order(vector<Cromosoma> &pop);
    // Operador de selección -> Torneo Binario
    unsigned BinaryTournament();
    // Operador de cruce -> posición, mantiene los genes coincidentes y completa de forma aleatoria sin repetir.
    Cromosoma CrossP(Cromosoma c1, Cromosoma c2);
    // Operador de cruce -> OX, subcadena de padre 1 y completa con padre 2.
    Cromosoma CrossOX(Cromosoma c1, Cromosoma c2);
    // Operador de mutación -> Dos genes elegidos al azar son intercambiados
    void Mutation(vector<Cromosoma> &pop);
    // Solamente se mutan dos genes de forma aleatoria.
    void StationaryMutation(vector<Cromosoma> &pop);
    // Variante Baldwiniana: Greedy 2-opt y no susituye el cromosoma
    void Baldwiniana(vector<Cromosoma> &pop);
    // Variante Lamarckiana: Greedy 2-opt y sustituye el cromosoma
    void Lamarckiana(vector<Cromosoma> &pop);
    // Algoritmo Genético Generacional: cruce posición y OX. Normal, Lamarckiana y Baldwiniana
    void AGG(unsigned maxIter, unsigned cruce);
    // Algoritmo Genético Estacionario: cruce posición y OX. Normal, Lamarckiana y Baldwiniana
    void AGE(unsigned maxIter, unsigned cruce);
};

#endif
