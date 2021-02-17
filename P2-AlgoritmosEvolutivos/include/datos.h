#ifndef DATOS_H
#define DATOS_H

#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>

using namespace std;

class Datos {
  public:
    unsigned nInstalaciones;
    vector<vector<int>> flujos;
    vector<vector<int>> distancias;

    // Constructores
    Datos();
    Datos(char* file);
};

#endif
