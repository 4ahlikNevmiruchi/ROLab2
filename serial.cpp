#include <cstdio> // For C-style I/O
#include <cstdlib> // For rand(), srand()
#include <ctime>   // For clock(), time_t, CLOCKS_PER_SEC

void ProcessInitialization(double* &pAMatrix, double* &pBMatrix, double* &pCMatrix, int &Size);
void DummyDataInitialization(double* pAMatrix, double* pBMatrix, int Size);
void RandomDataInitialization(double* pAMatrix, double* pBMatrix, int Size);
void PrintMatrix(double* pMatrix, int Rows, int Cols);
void SerialMatrixMultiplication(double* pAMatrix, double* pBMatrix, double* pCMatrix, int Size);
void SerialResultCalculation(double* pAMatrix, double* pBMatrix, double* pCMatrix, int Size);
void ProcessTermination(double* pAMatrix, double* pBMatrix, double* pCMatrix);

int main() {
    double* pAMatrix; // First argument of matrix multiplication
    double* pBMatrix; // Second argument of matrix multiplication
    double* pCMatrix; // Result matrix
    int Size;         // Size of matrices

    time_t start, finish;
    double duration;

    printf("Serial matrix multiplication program\n");

    // Get size, allocate memory, and initialize
    ProcessInitialization(pAMatrix, pBMatrix, pCMatrix, Size);


    // NOTE: COMMENT OUT PrintMatrix ON LARGE EXPIREMENTS
    printf("\nInitial A Matrix \n");
    PrintMatrix(pAMatrix, Size, Size);

    printf("\nInitial B Matrix \n");
    PrintMatrix(pBMatrix, Size, Size);
    // UP TO HERE

    // Perform and time the multiplication
    printf("\nCalculating result matrix...\n");
    start = clock();
    SerialResultCalculation(pAMatrix, pBMatrix, pCMatrix, Size);
    finish = clock();
    duration = (finish - start) / double(CLOCKS_PER_SEC);

    // Print the result COMMENT OUT ON BIG CALCULATIONS
    printf("\nResult C Matrix \n");
    PrintMatrix(pCMatrix, Size, Size);

    // Print the execution time
    printf("\nTime of execution: %f sec\n", duration);

    //Free the memory
    ProcessTermination(pAMatrix, pBMatrix, pCMatrix);

    printf("\nPress Enter to exit...");
    getchar();

    return 0;
}

void ProcessInitialization(double* &pAMatrix, double* &pBMatrix, double* &pCMatrix, int &Size) {
    // Setting the size of matrices
    do {
        printf("\nEnter the size of matrices: ");
        scanf("%d", &Size);
        printf("\nChosen matrices' size = %d\n", Size);
        if (Size <= 0) {
            printf("\nSize of objects must be greater than 0!\n");
        }
    } while (Size <= 0);

    // Memory allocation
    pAMatrix = new double[Size*Size];
    pBMatrix = new double[Size*Size];
    pCMatrix = new double[Size*Size];

    // Initialization of matrix elements
    RandomDataInitialization(pAMatrix, pBMatrix, Size);
    //DummyDataInitialization(pAMatrix, pBMatrix, Size);
    for (int i = 0; i < Size*Size; i++) {
        pCMatrix[i] = 0; // Initialize result matrix to zero
    }

    // Clear the input buffer (e.g., the newline left by scanf)
    // This ensures the final getchar() in main() works as expected.
    while (getchar() != '\n');
}

void DummyDataInitialization(double* pAMatrix, double* pBMatrix, int Size) {
    // Fill A and B with 1s
    for (int i = 0; i < Size; i++) {
        for (int j = 0; j < Size; j++) {
            pAMatrix[i*Size + j] = 1;
            pBMatrix[i*Size + j] = 1;
        }
    }
}

void RandomDataInitialization(double* pAMatrix, double* pBMatrix, int Size) {
    int i, j; // Loop variables

    // Seed the random number generator
    srand(unsigned(clock()));

    for (i = 0; i < Size; i++) {
        for (j = 0; j < Size; j++) {
            pAMatrix[i*Size + j] = rand() / double(1000);
            pBMatrix[i*Size + j] = rand() / double(1000);
        }
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
    int i, j, k; // Loop variables
    for (i = 0; i < Size; i++) {
        for (j = 0; j < Size; j++) {
            for (k = 0; k < Size; k++) {
                pCMatrix[i*Size + j] += pAMatrix[i*Size + k] * pBMatrix[k*Size + j];
            }
        }
    }
}

void ProcessTermination(double* pAMatrix, double* pBMatrix, double* pCMatrix) {
    // Deallocate memory
    delete[] pCMatrix;
    delete[] pBMatrix;
    delete[] pAMatrix;
}