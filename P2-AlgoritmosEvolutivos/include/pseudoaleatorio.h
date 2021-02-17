#ifndef PSEUDOALEATORIO_H
#define PSEUDOALEATORIO_H

#define MASK 2147483647
#define PRIME 65539
#define SCALE 0.4656612875e-9

long Seed;

#define Rand()  (( Seed = ( (Seed * PRIME) & MASK) ) * SCALE )

#define Randint(low,high) ( (int) (low + (high-(low)+1) * Rand()))

#define Randfloat(low,high) ( (low + (high-(low))*Rand()))

#endif
