#include "cromosoma.h"
#include "pseudoaleatorio.h"

/******************************************************************************/

Cromosoma::Cromosoma() {
  solution = vector<int>(0);
  fitness = 0;
}

/******************************************************************************/

void Cromosoma::SetSemilla(int &semilla) {
  Seed = semilla;
}

/******************************************************************************/

unsigned Cromosoma::GenerateNumber(unsigned low, unsigned high) {
  return Randint(low, high);
}

/******************************************************************************/

void Cromosoma::CalculateFitness(Datos &datos) {
  fitness = 0;
  for(unsigned i=0; i<datos.nInstalaciones; ++i)
    for (unsigned j=0; j<solution.size(); ++j)
      if (i != j)
        fitness += datos.flujos[i][j]*datos.distancias[solution[i]][solution[j]];
}

/******************************************************************************/

int Cromosoma::CalculateDifFitness(Datos &datos, unsigned i1, unsigned i2) {
  int dif = 0;
  for (unsigned i=0; i<datos.nInstalaciones; ++i) {
    if (i != i1 && i != i2) {
      dif += datos.flujos[i1][i]*(datos.distancias[solution[i2]][solution[i]]-datos.distancias[solution[i1]][solution[i]])
        + datos.flujos[i2][i]*(datos.distancias[solution[i1]][solution[i]]-datos.distancias[solution[i2]][solution[i]]) +
        + datos.flujos[i][i1]*(datos.distancias[solution[i]][solution[i2]]-datos.distancias[solution[i]][solution[i1]]) +
        + datos.flujos[i][i2]*(datos.distancias[solution[i]][solution[i1]]-datos.distancias[solution[i]][solution[i2]]);
    }
  }
  return dif;
}

/******************************************************************************/

void Cromosoma::Init(Datos &datos) {
  unsigned rand;
  solution.resize(datos.nInstalaciones, -1);
  for (unsigned i=0; i<datos.nInstalaciones; ++i) {
    rand = Randint(0, datos.nInstalaciones-1);
    // no hay repetición
    while (find(solution.begin(), solution.end(), rand) != solution.end()) {
      rand = Randint(0, datos.nInstalaciones-1);
    }
    solution[i] = rand;
  }
  CalculateFitness(datos);  // nuevo coste
}

/******************************************************************************/

Cromosoma Cromosoma::SimulateSwapGens(Datos &datos, unsigned gen1, unsigned gen2) {
  Cromosoma copiaCromosoma;
  copiaCromosoma.solution = solution;
  swap(copiaCromosoma.solution[gen1], copiaCromosoma.solution[gen2]);
  copiaCromosoma.CalculateFitness(datos);
  return copiaCromosoma;
}

/******************************************************************************/

void Cromosoma::SwapGens(unsigned gen1, unsigned gen2) {
  swap(solution[gen1], solution[gen2]);
}

/******************************************************************************/

void Cromosoma::Show() {
  cout << "Solution: " << endl;
  for (unsigned i=0; i<solution.size(); ++i) {
    cout << solution[i] << " ";
  }
  cout << endl << "Fitness: " << fitness << endl;
}

/******************************************************************************/
