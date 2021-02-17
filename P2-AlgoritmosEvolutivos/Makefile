####### MACROS PARA LOS DIRECTORIOS, LIBRERÍAS Y COMPILADOR
CXX = g++ -O3 -g -std=gnu++11 -Wall -Wextra -I./$(INC)
INC = ./include
SRC = ./src
LIB = ./lib
BIN = ./bin
LDFLAGS = -L$(LIB)
AR = ar
LDLIBS = -lQAP

all: $(BIN)/exe

## Compilando datos ##
$(BIN)/datos.o : $(INC)/datos.h $(SRC)/datos.cpp
	$(CXX) -c -o $(BIN)/datos.o $(SRC)/datos.cpp

## Compilando cromosoma ##
$(BIN)/cromosoma.o : $(INC)/cromosoma.h $(SRC)/cromosoma.cpp
	$(CXX) -c -o $(BIN)/cromosoma.o $(SRC)/cromosoma.cpp

## Compilando busquedaLocal ##
$(BIN)/busquedaLocal.o : $(INC)/busquedaLocal.h $(SRC)/busquedaLocal.cpp
	$(CXX) -c -o $(BIN)/busquedaLocal.o $(SRC)/busquedaLocal.cpp

## Compilando geneticos ##
$(BIN)/geneticos.o : $(INC)/geneticos.h $(SRC)/geneticos.cpp
	$(CXX) -c -o $(BIN)/geneticos.o $(SRC)/geneticos.cpp

## Compilando greedy ##
$(BIN)/greedy.o : $(INC)/greedy.h $(SRC)/greedy.cpp
	$(CXX) -c -o $(BIN)/greedy.o $(SRC)/greedy.cpp

## Compilando objetos ##
$(BIN)/practica2.o : $(INC)/datos.h $(INC)/cromosoma.h $(INC)/geneticos.h $(INC)/busquedaLocal.h $(INC)/greedy.h
	$(CXX) -c -o $(BIN)/practica2.o $(SRC)/main.cpp

## Ejecutable ##
$(BIN)/exe : $(BIN)/practica2.o $(LIB)/libQAP.a
	$(CXX) -o $(BIN)/exe $(BIN)/practica2.o $(LDFLAGS) $(LDLIBS)

## Librería ##
$(LIB)/libQAP.a : $(BIN)/datos.o $(BIN)/cromosoma.o $(BIN)/geneticos.o $(BIN)/busquedaLocal.o $(BIN)/greedy.o
	$(AR) rvs $(LIB)/libQAP.a $(BIN)/datos.o $(BIN)/cromosoma.o $(BIN)/geneticos.o $(BIN)/busquedaLocal.o $(BIN)/greedy.o

clean:
	-@rm $(BIN)/*.o
	@echo "Limpiando..."
