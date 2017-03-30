/* headers */
#include <stdlib.h>
#include <time.h>

/* auxiliary functions declarations */
char * allocate_lattice(int rows, int columns);

void populate_lattice(double probability, char * lattice, int rows, int columns);

/* main body function */
int main()
{
    int L; /* square lattice size */
    double p; /* occupation probability of each lattice site */

    /* initialize random number generator seed */
    srand(time(NULL));

    /* allocate lattice */
    L = 10;
    char *lattice = allocate_lattice(L, L);

    /* populate lattice with given probability */
    p = 0.4;
    populate_lattice(p, lattice, L, L);

    /* free memory before leaving */
    free(lattice);

    return 0;
}

/* auxiliary functions definitions */
char * allocate_lattice(int rows, int columns)
{
    char *lattice;
    int i, j;

    lattice = (char *) malloc(rows*columns*sizeof(char));
    for (i = 0; i < rows; i++) {
        for (j = 0; j < columns; j++) {
            lattice[i + j*columns] = 0;
        }
    }

    return lattice;
}

void populate_lattice(double probability, char * lattice, int rows, int columns)
{
    int i, j;
    double q;

    for (i = 0; i < rows; i++) {
        for (j = 0; j < columns; j++) {
            q = rand()/RAND_MAX;
            if (q <= probability) {
                lattice[i + j*columns] = 1;
            }
        }
    }

    return;
}
