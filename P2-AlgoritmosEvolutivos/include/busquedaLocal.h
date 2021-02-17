#ifndef BL_H
#define BL_H

#include "cromosoma.h"

class BusquedaLocal {
  public:
    Datos datos;
    Cromosoma solBL;
    int numIter;

    // Constructores
    BusquedaLocal(Datos &dat);
    BusquedaLocal(Datos &dat, Cromosoma &c);
    // Algoritmo de Búsqueda Local
    void BL(int iteraciones);
};

#endif
