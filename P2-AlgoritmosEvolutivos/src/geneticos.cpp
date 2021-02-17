#include "geneticos.h"
#include <time.h>

/******************************************************************************/

Geneticos::Geneticos(Datos &dat, float pCr, float pMu, unsigned size, unsigned mod, bool initGreedy) {
  datos = dat;
  pCross = pCr;
  pMut = pMu;
  mode = mod;

  population.resize(size);
  if (!initGreedy) { // random population
    for (unsigned i=0; i<population.size(); ++i)
      population[i].Init(dat);
  } else {  // greedy population
    for (unsigned i=0; i<population.size(); ++i) {
      Greedy g;
      g.AlgGreedy(datos);
      population[i] = g.solutionGreedy;
    }
  }

  if (mode == 1) {
    Baldwiniana(population);
  } else if (mode == 2) {
    Lamarckiana(population);
  }

  crossNumber = pCross*(population.size()/2) + 1;
  ngens = pMut*(population.size()*datos.nInstalaciones);
}

/******************************************************************************/

void Geneticos::Order(vector<Cromosoma> &pop) {
  Cromosoma aux;
  for (unsigned i=1; i<pop.size(); ++i) {
    for (unsigned j=i; j>0; --j) {
      if (pop[j].fitness < pop[j-1].fitness) {
        aux = pop[j];
        pop[j] = pop[j-1];
        pop[j-1] = aux;
      }
    }
  }
}

/******************************************************************************/

unsigned Geneticos::BinaryTournament() {
  unsigned p1 = Cromosoma::GenerateNumber(0, population.size()-1);
  unsigned p2;
  do{
    p2 = Cromosoma::GenerateNumber(0, population.size()-1);
  } while (p1 == p2);
  // población ordenada -> el mejor es el menor
  if (p1 < p2) return p1;
  else return p2;
}

/******************************************************************************/

Cromosoma Geneticos::CrossP(Cromosoma c1, Cromosoma c2) {
  unsigned gen;
  Cromosoma son;
  son.solution.resize(c1.solution.size(), -1);

  for (unsigned i=0; i<c1.solution.size(); ++i) {
    if (c1.solution[i] == c2.solution[i]) {
      son.solution[i] = c1.solution[i];
    }
  }

  vector<int>::iterator it = son.solution.begin();
  for (unsigned i=0; i<son.solution.size(); ++i) {
    if (son.solution[i] == -1) {
      do {
        gen = Cromosoma::GenerateNumber(0, son.solution.size()-1);
        it = find(son.solution.begin(), son.solution.end(), gen);
      } while (it != son.solution.end()); // Sin genes repetidos
      son.solution[i] = gen;
    }
  }
  son.CalculateFitness(datos);
  return son;
}

/******************************************************************************/

Cromosoma Geneticos::CrossOX(Cromosoma c1, Cromosoma c2) {
  Cromosoma son;
  son.solution.resize(c1.solution.size(), -1);
  unsigned mid = c1.solution.size()/2;
  unsigned beg = mid - (mid/2);
  unsigned en = mid + (mid/2);
  for (unsigned i=beg; i<=en; ++i)
    son.solution[i] = c1.solution[i];

  unsigned parentIndex = en+1, sonIndex = en+1;
  while (find(son.solution.begin(), son.solution.end(),-1) != son.solution.end()) {
    if (parentIndex == c2.solution.size()) parentIndex = 0;
    if (sonIndex == son.solution.size()) sonIndex = 0;

    while (sonIndex < son.solution.size() && parentIndex < c2.solution.size()) {
      if (son.solution[sonIndex] == -1) {
        if (find(son.solution.begin(), son.solution.end(), c2.solution[parentIndex]) == son.solution.end()) {
          son.solution[sonIndex] = c2.solution[parentIndex];
          sonIndex++;
        }
        parentIndex++;
      }
      else sonIndex++;
    }
  }
  son.CalculateFitness(datos);
  return son;
}

/******************************************************************************/

void Geneticos::Mutation(vector<Cromosoma> &pop) {
  unsigned gen1, gen2, sonMut;
  for (unsigned i=0; i<ngens; ++i) {
    ++iters;
    gen1 = Cromosoma::GenerateNumber(0, datos.nInstalaciones-1);
    gen2 = Cromosoma::GenerateNumber(0, datos.nInstalaciones-1);
    while (gen1 == gen2 ) {
      gen2 = Cromosoma::GenerateNumber(0, datos.nInstalaciones-1);
    }
    sonMut = Cromosoma::GenerateNumber(0, pop.size()-1);
    pop[sonMut].SwapGens(gen1, gen2);
    pop[sonMut].CalculateFitness(datos);
  }
}

/******************************************************************************/

void Geneticos::StationaryMutation(vector<Cromosoma> &pop) {
  unsigned gen1, gen2, sonMut;
  ++iters;
  gen1 = Cromosoma::GenerateNumber(0, datos.nInstalaciones-1);
  gen2 = Cromosoma::GenerateNumber(0, datos.nInstalaciones-1);
  while (gen1 == gen2 ) {
    gen2 = Cromosoma::GenerateNumber(0, datos.nInstalaciones-1);
  }
  sonMut = Cromosoma::GenerateNumber(0, pop.size()-1);
  pop[sonMut].SwapGens(gen1, gen2);
  pop[sonMut].CalculateFitness(datos);
}

/******************************************************************************/

void Geneticos::Baldwiniana(vector<Cromosoma> &pop) {
  for (unsigned i=0; i<pop.size(); ++i) {
    algGreedy.Alg2opt(pop[i], datos);
    pop[i].fitness = algGreedy.solutionGreedy.fitness;
  }
}

/******************************************************************************/

void Geneticos::Lamarckiana(vector<Cromosoma> &pop) {
  for (unsigned i=0; i<pop.size(); ++i) {
    algGreedy.Alg2opt(pop[i], datos);
    pop[i].solution = algGreedy.solutionGreedy.solution;
    pop[i].fitness = algGreedy.solutionGreedy.fitness;
  }
}

/******************************************************************************/


void Geneticos::AGG(unsigned maxIter, unsigned cruce) {
  iters = 0;
  int numIterBL = 0;

  // Contamos el tiempo
  clock_t start,end;
  start = clock();

  while (iters < maxIter) {
    Cromosoma s1, s2;
    unsigned parentsCrossed = 0;
    ++numIterBL;
    vector<Cromosoma> parents, sons;
    Order(population);
    int parentSelected;

    for (unsigned i=0; i<population.size(); ++i) {
      parentSelected = BinaryTournament();
      parents.push_back(population[parentSelected]);
    }

    for (unsigned i=0; i<crossNumber; ++i) {
      if (cruce == 0) {
        s1 = CrossP(parents[parentsCrossed], parents[parentsCrossed+1]);
        s2 = CrossP(parents[parentsCrossed], parents[parentsCrossed+1]);
      }
      else if(cruce == 1){
        s1 = CrossOX(parents[parentsCrossed], parents[parentsCrossed+1]);
        s2 = CrossOX(parents[parentsCrossed], parents[parentsCrossed+1]);
      }

      sons.push_back(s1);
      sons.push_back(s2);
      parentsCrossed += 2;
      iters += 2;
    }

    for (unsigned i=parentsCrossed; i<parents.size(); ++i)
      sons.push_back(parents[i]);

    Mutation(sons);   // mutamos

    if (mode == 1 && numIterBL == 10) {
      numIterBL = 0;
      Baldwiniana(sons);
      iters += algGreedy.iter;
    } else if (mode == 2 && numIterBL == 10) {
      numIterBL = 0;
      Lamarckiana(population);
      iters += algGreedy.iter;
    }
    Order(sons);  //ordenamos población

    // Elitismo
    bool bestFather = false;
    for (unsigned i=0; i<sons.size() && !bestFather; ++i)
      if (population[0].solution == sons[i].solution)
        bestFather = true;

    if (!bestFather) {
      sons[sons.size()-1] = population[0];
    }

    population = sons;  // población de la siguiente iteración
  }
  end = clock();
  cout << "Tiempo: " << (end-start)/(double)CLOCKS_PER_SEC << "s" << endl;
}

/******************************************************************************/

// Esquema generacional del Algoritmo Estacionario con cruce basado en posición
void Geneticos::AGE(unsigned maxIter, unsigned cruce) {
  iters = 0;
  unsigned p1, p2;
  Cromosoma s1, s2;
  unsigned numIterBL = 0;
  // Contamos el tiempo
  clock_t start,end;
  start = clock();
  while (iters < maxIter) {
    ++numIterBL;
    vector<Cromosoma> sons;
    Order(population);

    p1 = BinaryTournament();
    p2 = BinaryTournament();

    if (cruce == 0) {
      s1 = CrossP(population[p1], population[p2]);
      s2 = CrossP(population[p1], population[p2]);
    } else if(cruce == 1){
      s1 = CrossOX(population[p1], population[p2]);
      s2 = CrossOX(population[p1], population[p2]);
    }

    sons.push_back(s1);
    sons.push_back(s2);
    iters += 2;

    StationaryMutation(sons); // mutamos

    if (mode == 1 && iters == 10) {
      iters = 0;
      Baldwiniana(sons);
      iters += algGreedy.iter;
    }
    else if (mode == 2 && numIterBL == 10) {
      numIterBL = 0;
      Lamarckiana(population);
      iters += algGreedy.iter;
    }
    // Se eligen los dos mejores entre los dos hijos obtenidos por los dos padres y estos
    sons.push_back(population[population.size()-1]);
    sons.push_back(population[population.size()-2]);
    Order(sons);
    population[population.size()-1] = sons[0];
    population[population.size()-2] = sons[1];
  }
  end = clock();
  cout << "Tiempo: " << (end-start)/(double)CLOCKS_PER_SEC << "s" << endl;
}

/******************************************************************************/
