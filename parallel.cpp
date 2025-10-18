#include <stdio.h>
#include <stdlib.h> // For rand(), srand()
#include <ctime>    // For clock(), time_t, CLOCKS_PER_SEC
#include <math.h>   // For sqrt()
#include <mpi.h>    // For MPI functions

// mpicxx -o serial_mpi serial.cpp
// mpirun -np 8 ./serial_mpi

// MPI process management
int ProcNum = 0;      // Number of available processes
int ProcRank = 0;     // Rank of current process
int GridSize;         // Size of virtual processor grid
MPI_Comm GridComm;    // Grid communicator
MPI_Comm ColComm;     // Column communicator
MPI_Comm RowComm;     // Row communicator
int GridCoords[2];    // Coordinates of current process in grid

// Matrix data
double* pAMatrix; // First argument of matrix multiplication
double* pBMatrix; // Second argument of matrix multiplication
double* pCMatrix; // Result matrix
int Size;         // Size of matrices (total)

// Timing
double Start, Finish, Duration;

// --- Function Declarations ---

// New function for MPI grid setup (Task 2)
void CreateGridCommunicators();

// Functions copied from Serial Lab (to be modified later)
void ProcessInitialization(double* &pAMatrix, double* &pBMatrix, double* &pCMatrix, int &Size);
void RandomDataInitialization(double* pAMatrix, double* pBMatrix, int Size);
void PrintMatrix(double* pMatrix, int Rows, int Cols);
void SerialResultCalculation(double* pAMatrix, double* pBMatrix, double* pCMatrix, int Size);
void ProcessTermination(double* pAMatrix, double* pBMatrix, double* pCMatrix);


// --- Main Program ---

int main(int argc, char* argv[]) {
    // MPI Initialization
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    
    // Disable buffering for stdout
    setvbuf(stdout, 0, _IONBF, 0);

    // Check if the number of processes is a perfect square
    GridSize = sqrt((double)ProcNum);
    if (ProcNum != GridSize * GridSize) {
        if (ProcRank == 0) {
            printf("Number of processes must be a perfect square \n");
        }
    } else {
        if (ProcRank == 0) {
            printf("Parallel matrix multiplication program\n");
        }
        
        // Grid communicator creating (Task 2)
        CreateGridCommunicators();
        
        //
        // --- Future tasks (data distribution, calculation) will go here ---
        //
    }

    // Finalize MPI environment
    MPI_Finalize();
    return 0;
}


// --- Function Definitions ---

/**
 * @brief Creates the Cartesian grid and row/column sub-communicators.
 * (Corresponds to Task 2)
 */
void CreateGridCommunicators() {
    int DimSize[2]; // Number of processes in each dimension
    int Periodic[2]; // =1, if the grid dimension should be periodic
    int Subdims[2]; // Used to create sub-grids (rows/cols)

    // --- Creation of the Cartesian communicator ---
    DimSize[0] = GridSize;
    DimSize[1] = GridSize;
    Periodic[0] = 1; // Periodic shift for Fox algorithm
    Periodic[1] = 1; // Periodic shift for Fox algorithm
    
    // Create the 2D Cartesian grid communicator
    MPI_Cart_create(MPI_COMM_WORLD, 2, DimSize, Periodic, 1, &GridComm);

    // --- Determination of the cartesian coordinates for every process ---
    MPI_Cart_coords(GridComm, ProcRank, 2, GridCoords);

    // --- Creating communicators for rows ---
    // Dimensions are [row, col]
    Subdims[0] = 0; // Dimension 0 (row) is fixed
    Subdims[1] = 1; // Dimension 1 (col) is free
    // Create a communicator for all processes in the same row
    MPI_Cart_sub(GridComm, Subdims, &RowComm);

    // --- Creating communicators for columns ---
    Subdims[0] = 1; // Dimension 0 (row) is free
    Subdims[1] = 0; // Dimension 1 (col) is fixed
    // Create a communicator for all processes in the same column
    MPI_Cart_sub(GridComm, Subdims, &ColComm);
}


// --- Serial Functions (Copied from Lab 1) ---
// These are placeholders to be adapted for parallel execution in later tasks.

void ProcessInitialization(double* &pAMatrix, double* &pBMatrix, double* &pCMatrix, int &Size) {
    if (ProcRank == 0) {
        do {
            printf("\nEnter the size of matrices: ");
            scanf("%d", &Size);
            printf("\nChosen matrices' size = %d\n", Size);
            if (Size <= 0) {
                printf("\nSize of objects must be greater than 0!\n");
            }
        } while (Size <= 0);
    }

    // MPI_Bcast(&variable, count, type, root_rank, communicator);
    MPI_Bcast(&Size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    pAMatrix = new double[Size*Size];
    pBMatrix = new double[Size*Size];
    pCMatrix = new double[Size*Size];

    RandomDataInitialization(pAMatrix, pBMatrix, Size);
    for (int i = 0; i < Size*Size; i++) {
        pCMatrix[i] = 0; 
    }
    // Clear input buffer on rank 0
    if (ProcRank == 0) {
        while (getchar() != '\n');
    }
}

void RandomDataInitialization(double* pAMatrix, double* pBMatrix, int Size) {
    srand(unsigned(clock()));
    for (int i = 0; i < Size*Size; i++) {
        pAMatrix[i] = rand() / double(1000);
        pBMatrix[i] = rand() / double(1000);
    }
}

void PrintMatrix(double* pMatrix, int Rows, int Cols) {
    for (int i = 0; i < Rows; i++) {
        for (int j = 0; j < Cols; j++) {
            printf("%7.4f ", pMatrix[i*Cols + j]);
        }
        printf("\n");
    }
}

void SerialResultCalculation(double* pAMatrix, double* pBMatrix, double* pCMatrix, int Size) {
    for (int i = 0; i < Size; i++) {
        for (int j = 0; j < Size; j++) {
            for (int k = 0; k < Size; k++) {
                pCMatrix[i*Size + j] += pAMatrix[i*Size + k] * pBMatrix[k*Size + j];
            }
        }
    }
}

void ProcessTermination(double* pAMatrix, double* pBMatrix, double* pCMatrix) {
    delete[] pCMatrix;
    delete[] pBMatrix;
    delete[] pAMatrix;
}