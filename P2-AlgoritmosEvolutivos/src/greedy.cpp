#include "greedy.h"

/******************************************************************************/

Greedy::Greedy() {}

/******************************************************************************/

void Greedy::AlgGreedy(Datos &datos) {
  unsigned flujoMax, flujoPos=0;
  unsigned distMin, distPos;
  vector<unsigned> potencialDistancia(datos.nInstalaciones, 0);
  vector<unsigned> potencialFlujo(datos.nInstalaciones, 0);
  vector<bool> localizaciones(datos.nInstalaciones, false);
  vector<bool> unidades(datos.nInstalaciones, false);
  solutionGreedy.solution.resize(datos.nInstalaciones, -1);

  for (unsigned i=0; i<datos.nInstalaciones; ++i) {
    for (unsigned j=0; j<datos.nInstalaciones; ++j) {
      potencialDistancia[i] += datos.distancias[i][j];
      potencialFlujo[i] += datos.flujos[i][j];
    }
  }

  for (unsigned i=0; i<datos.nInstalaciones; ++i) {
    flujoMax = 0;
    distMin = INT_MAX;

    // Minimizando distnacias
    for (unsigned j=0; j<datos.nInstalaciones; ++j) {
      if (potencialDistancia[j] < distMin && !localizaciones[j]) {
        distMin = potencialDistancia[j];
        distPos = j;
      }
    }

    // Maximizando flujos
    for (unsigned j=0; j<datos.nInstalaciones; ++j) {
      if (potencialFlujo[j] > flujoMax && !unidades[j]) {
        flujoMax = potencialFlujo[j];
        flujoPos = j;
      }
    }

    solutionGreedy.solution[flujoPos] = distPos;
    unidades[flujoPos] = true;
    localizaciones[distPos] = true;
  }

  solutionGreedy.CalculateFitness(datos);
}

/******************************************************************************/

void Greedy::Alg2opt(Cromosoma crom, Datos &datos) {
  Cromosoma bestCrom = crom;
  unsigned numIter = 10;
  bool isBetter = false;
  iter = 0;

  do {
    isBetter = false;
    for (unsigned i=0; i<datos.nInstalaciones; ++i) {
      for (unsigned j=i+1; j<datos.nInstalaciones && !isBetter; ++j) {
        Cromosoma nextCrom = crom.SimulateSwapGens(datos, i, j);
        if (nextCrom.fitness < crom.fitness) crom = nextCrom;
        if (crom.fitness < bestCrom.fitness) {
          bestCrom = crom;
          isBetter = true;
        }
      }
    }
    ++iter;
  } while( (iter < numIter) && isBetter);

  solutionGreedy = bestCrom;
}

/******************************************************************************/
