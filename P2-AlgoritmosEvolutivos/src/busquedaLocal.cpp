#include "busquedaLocal.h"

/******************************************************************************/

BusquedaLocal::BusquedaLocal(Datos &dat) {
  datos = dat;
  solBL.Init(datos);
}

/******************************************************************************/

BusquedaLocal::BusquedaLocal(Datos &dat, Cromosoma &c) {
  datos = dat;
  solBL = c;
}

/******************************************************************************/

void BusquedaLocal::BL(int iteraciones) {
  numIter = 0;
  vector<int> mask;
  mask.resize(datos.nInstalaciones, 0);
  bool improveFlag=false, mejorSol = false;

  while (!mejorSol && numIter < iteraciones) {
    mejorSol = false;
    for (unsigned i=0; i<datos.nInstalaciones && numIter<iteraciones; ++i) {
      bool hayMejora = true;
      for (unsigned j=0; j<mask.size() && hayMejora; ++j) {
        if (mask[j] == 0) {
          hayMejora = false;
        }
      }
      if (hayMejora)   mejorSol = true;
      if (mask[i] == 0) {
        improveFlag = false;
        for (unsigned j=0; j<datos.nInstalaciones; ++j) {
          // Se comprueba si al intercambiar se mejora
          if (solBL.CalculateDifFitness(datos, i, j) < 0) {
            solBL.SwapGens(i, j);
            mejorSol = improveFlag = true;
            mask[i] = mask[j] = 0;
          }
          numIter++;
        }
      }
      if (!improveFlag) mask[i] = 1;
    }
  }
  solBL.CalculateFitness(datos);
}

/******************************************************************************/
