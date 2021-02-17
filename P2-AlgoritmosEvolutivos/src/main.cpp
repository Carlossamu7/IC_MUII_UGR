#include "datos.h"
#include "geneticos.h"
#include "busquedaLocal.h"
#include "greedy.h"

using namespace std;

int main(int argc, char *argv[]) {
  if (argc < 2) { // No hay argumentos
    cout << "El número de argumentos no es válido" << endl;
    cout << "Ejecute: ./exe ./data/<filename>.dat" << endl;
    return 0;
  } else{ // Hay argumentos
    int semilla=10

    // Número de iteraciones de los algoritmos
    unsigned iterGenetic = 150000;
    unsigned iterBaldwin = 50000;
    unsigned iterLamarck = 1000;
    // Tamaño de la población
    unsigned short populationSize = 20;
    // Probabilidades
    float pCross = 0.7;
    float pMutation = 0.001;

    // Leemos datos
    char* file = argv[1];
    Datos datos(file);

    // Resultados
    cout << "\033[31m" << "----------------------------------------------------" <<  "\033[37m" << endl;
    cout << "\033[31m" << "--------- Inteligencia Computacional - QAP ---------" <<  "\033[37m" << endl;
    cout << "\033[31m" << "----------------------------------------------------" <<  "\033[37m" << endl << endl;

    cout << "\033[33m" << "--------------- AGE P (inicial greedy) --------------" <<  "\033[37m" << endl;
    cout << "Iteraciones: " << iterGenetic << endl;
    Cromosoma::SetSemilla(semilla);
    Geneticos ageP(datos, pCross, pMutation, populationSize, 0, false);
    ageP.AGE(iterGenetic, 0);
    ageP.population[0].Show();
    cout << endl;

    cout << "\033[33m" << "--------------- AGE OX (inicial greedy) --------------" <<  "\033[37m" << endl;
    cout << "Iteraciones: " << iterGenetic << endl;
    Cromosoma::SetSemilla(semilla);
    Geneticos ageOXGreedy(datos, pCross, pMutation, populationSize, 0, false);
    ageOXGreedy.AGE(iterGenetic, 1);
    ageOXGreedy.population[0].Show();
    cout << endl;

    cout << "\033[33m" << "--------------- AGE P Baldwiniano --------------" <<  "\033[37m" << endl;
    cout << "Iteraciones: " << iterBaldwin << endl;
    Cromosoma::SetSemilla(semilla);
    Geneticos ageBP(datos, pCross, pMutation, populationSize, 1, false);
    ageBP.AGE(iterBaldwin, 0);
    ageBP.population[0].Show();
    cout << endl;

    cout << "\033[33m" << "--------------- AGE OX  Baldwiniano --------------" <<  "\033[37m" << endl;
    cout << "Iteraciones: " << iterBaldwin << endl;
    Cromosoma::SetSemilla(semilla);
    Geneticos ageBOX(datos, pCross, pMutation, populationSize, 1, false);
    ageBOX.AGE(iterBaldwin, 1);
    ageBOX.population[0].Show();
    cout << endl;

    cout << "\033[33m" << "--------------- AGE P Larmarckiano --------------" <<  "\033[37m" << endl;
    cout << "Iteraciones: " << iterLamarck << endl;
    Cromosoma::SetSemilla(semilla);
    Geneticos ageL(datos, pCross, pMutation, populationSize, 2, false);
    ageL.AGE(iterLamarck, 0);
    ageL.population[0].Show();
    cout << endl;

    cout << "\033[33m" << "--------------- AGE OX Larmarckiano --------------" <<  "\033[37m" << endl;
    cout << "Iteraciones: " << iterLamarck << endl;
    Cromosoma::SetSemilla(semilla);
    Geneticos ageLOX(datos, pCross, pMutation, populationSize, 2, false);
    ageLOX.AGE(iterLamarck, 1);
    ageLOX.population[0].Show();
    cout << endl;

    ///////////////////////////////////////////////////////////////////////////////////////////////////

    cout << "\033[33m" << "--------------- AGG P (inicial greedy) --------------" <<  "\033[37m" << endl;
    cout << "Iteraciones: " << iterGenetic << endl;
    Cromosoma::SetSemilla(semilla);
    Geneticos aggP(datos, pCross, pMutation, populationSize, 0, false);
    aggP.AGG(iterGenetic, 0);
    aggP.population[0].Show();
    cout << endl;

    cout << "\033[33m" << "--------------- AGG OX (inicial greedy) --------------" <<  "\033[37m" << endl;
    cout << "Iteraciones: " << iterGenetic << endl;
    Cromosoma::SetSemilla(semilla);
    Geneticos aggOXGreedy(datos, pCross, pMutation, populationSize, 0, false);
    aggOXGreedy.AGG(iterGenetic, 1);
    aggOXGreedy.population[0].Show();
    cout << endl;

    cout << "\033[33m" << "--------------- AGG P Baldwiniano --------------" <<  "\033[37m" << endl;
    cout << "Iteraciones: " << iterBaldwin << endl;
    Cromosoma::SetSemilla(semilla);
    Geneticos aggBP(datos, pCross, pMutation, populationSize, 1, false);
    aggBP.AGG(iterBaldwin, 0);
    aggBP.population[0].Show();
    cout << endl;

    cout << "\033[33m" << "--------------- AGG OX  Baldwiniano --------------" <<  "\033[37m" << endl;
    cout << "Iteraciones: " << iterBaldwin << endl;
    Cromosoma::SetSemilla(semilla);
    Geneticos aggBOX(datos, pCross, pMutation, populationSize, 1, false);
    aggBOX.AGG(iterBaldwin, 1);
    aggBOX.population[0].Show();
    cout << endl;

    cout << "\033[33m" << "--------------- AGG P Larmarckiano --------------" <<  "\033[37m" << endl;
    cout << "Iteraciones: " << iterLamarck << endl;
    Cromosoma::SetSemilla(semilla);
    Geneticos aggL(datos, pCross, pMutation, populationSize, 2, false);
    aggL.AGG(iterLamarck, 0);
    aggL.population[0].Show();
    cout << endl;

    cout << "\033[33m" << "--------------- AGG OX Larmarckiano --------------" <<  "\033[37m" << endl;
    cout << "Iteraciones: " << iterLamarck << endl;
    Cromosoma::SetSemilla(semilla);
    Geneticos aggLOX(datos, pCross, pMutation, populationSize, 2, false);
    aggLOX.AGG(iterLamarck, 1);
    aggLOX.population[0].Show();
    cout << endl;

    return 0;
  }
}
