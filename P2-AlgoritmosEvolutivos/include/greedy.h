#ifndef GREEDY_H
#define GREEDY_H

#include "cromosoma.h"
#include "datos.h"
#include <limits.h>
#include <iostream>

class Greedy {
public:
  Cromosoma solutionGreedy;
  unsigned iter;

  // Construtor
  Greedy();
  // Algoritmo greedy
  void AlgGreedy(Datos &datos);
  // 2-opt
  void Alg2opt(Cromosoma crom, Datos &datos);
};

#endif
