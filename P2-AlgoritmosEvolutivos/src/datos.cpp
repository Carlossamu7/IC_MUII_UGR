#include "datos.h"
#include <iostream>

/******************************************************************************/

Datos::Datos() { }

/******************************************************************************/

Datos::Datos(char* file) {
  ifstream ifs;
  ifs.open(file, ifstream::in);

  // Número de instalaciones.
  ifs >> nInstalaciones;

  // Memoria para un vector de instalaciones
  flujos.resize(nInstalaciones);
  distancias.resize(nInstalaciones);

  // Cada posición es un vector obteniendo una matriz
  for (unsigned i=0; i<nInstalaciones; ++i) {
    flujos[i].resize(nInstalaciones);
    distancias[i].resize(nInstalaciones);
  }

  // Matriz de flujos
  for (unsigned i=0; i<nInstalaciones; ++i) {
    for (unsigned j=0; j<nInstalaciones; ++j) {
      ifs >> flujos[i][j];
    }
  }

  // Matriz de distancias
  for (unsigned i=0; i<nInstalaciones; ++i) {
    for (unsigned j=0; j<nInstalaciones; ++j) {
      ifs >> distancias[i][j];
    }
  }

  ifs.close();
}

/******************************************************************************/
