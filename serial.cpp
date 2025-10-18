#include <stdio.h> // For C-style I/O: printf, scanf, getchar

// --- Function Declarations ---


void ProcessInitialization(double* &pAMatrix, double* &pBMatrix, double* &pCMatrix, int &Size);
void DummyDataInitialization(double* pAMatrix, double* pBMatrix, int Size);
void PrintMatrix(double* pMatrix, int Rows, int Cols);
void SerialMatrixMultiplication(double* pAMatrix, double* pBMatrix, double* pCMatrix, int Size);
void ProcessTermination(double* pAMatrix, double* pBMatrix, double* pCMatrix);

int main() {
    double* pAMatrix; // First argument of matrix multiplication
    double* pBMatrix; // Second argument of matrix multiplication
    double* pCMatrix; // Result matrix
    int Size;         // Size of matrices

    printf("Serial matrix multiplication program\n");

    // Get size, allocate memory, and initialize
    ProcessInitialization(pAMatrix, pBMatrix, pCMatrix, Size);

    // Print initial matrices
    printf("\nInitial A Matrix \n");
    PrintMatrix(pAMatrix, Size, Size);

    printf("\nInitial B Matrix \n");
    PrintMatrix(pBMatrix, Size, Size);

    // Perform the multiplication
    SerialMatrixMultiplication(pAMatrix, pBMatrix, pCMatrix, Size);

    // Print the result
    printf("\nResult C Matrix \n");
    PrintMatrix(pCMatrix, Size, Size);

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
    DummyDataInitialization(pAMatrix, pBMatrix, Size);
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

void PrintMatrix(double* pMatrix, int Rows, int Cols) {
    for (int i = 0; i < Rows; i++) {
        for (int j = 0; j < Cols; j++) {
            printf("%7.4f ", pMatrix[i*Cols + j]);
        }
        printf("\n");
    }
}

void SerialMatrixMultiplication(double* pAMatrix, double* pBMatrix, double* pCMatrix, int Size) {
    // Standard C = A * B algorithm
    // C[i,j] = sum(A[i,k] * B[k,j]) for k = 0 to Size-1
    for (int i = 0; i < Size; i++) {
        for (int j = 0; j < Size; j++) {
            double temp = 0;
            for (int k = 0; k < Size; k++) {
                temp += pAMatrix[i*Size + k] * pBMatrix[k*Size + j];
            }
            pCMatrix[i*Size + j] = temp;
        }
    }
}

void ProcessTermination(double* pAMatrix, double* pBMatrix, double* pCMatrix) {
    // Deallocate memory
    delete[] pCMatrix;
    delete[] pBMatrix;
    delete[] pAMatrix;
}