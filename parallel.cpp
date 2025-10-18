#include <stdio.h>
#include <stdlib.h> // For rand(), srand()
#include <ctime>    // For clock(), time_t, CLOCKS_PER_SEC
#include <math.h>   // For sqrt(), fabs()
#include <mpi.h>    // For MPI functions

// === COMMANDS TO BUILD AND RUN ===
// mpicxx -o parallel_mpi parallel.cpp
// mpirun -np 4 --use-hwthread-cpus ./parallel_mpi
// mpirun -np 9 --use-hwthread-cpus ./parallel_mpi
// mpirun -np 16 --use-hwthread-cpus ./parallel_mpi

int ProcNum = 0;
int ProcRank = 0;
int GridSize;
MPI_Comm GridComm;
MPI_Comm ColComm;
MPI_Comm RowComm;
int GridCoords[2];

double* pAMatrix; // Full matrix A (on rank 0)
double* pBMatrix; // Full matrix B (on rank 0)
double* pCMatrix; // Full matrix C (on rank 0)
int Size;
int BlockSize;
double* pAblock;  // Current block of A
double* pBblock;  // Current block of B
double* pCblock;  // Block of result C
double* pMatrixABlock; // Initial block of A

double Start, Finish, Duration;
double SerialDuration = 0.0;
int runSerialTest = 0; // 1 to run 0 to not

// --- Parallel Setup Functions ---
void CreateGridCommunicators();
void ProcessInitialization(double* &pAMatrix, double* &pBMatrix, double* &pCMatrix,
                           double* &pAblock, double* &pBblock, double* &pCblock,
                           double* &pMatrixABlock, int &Size, int &BlockSize);
void ProcessTermination(double* pAMatrix, double* pBMatrix, double* &pCMatrix,
                        double* pAblock, double* pBblock, double* pCblock,
                        double* pMatrixABlock);

// --- Data Distribution Functions ---
void CheckerboardMatrixScatter(double* pMatrix, double* pMatrixBlock, int Size, int BlockSize);
void DataDistribution(double* pAMatrix, double* pBMatrix, double* pMatrixABlock,
                      double* pBblock, int Size, int BlockSize);

// --- Parallel Calculation Functions ---
void ParallelResultCalculation(double* pAblock, double* pMatrixABlock, double* pBblock,
                               double* pCblock, int BlockSize);
void ABlockCommunication(int iter, double* pAblock, double* pMatrixABlock, int BlockSize);
void BBlockCommunication(double* pBblock, int BlockSize, MPI_Comm ColComm);
void BlockMultiplication(double* pAblock, double* pBblock, double* pCblock, int BlockSize);

// --- Result Gathering & Testing Functions ---
void ResultCollection(double* pCMatrix, double* pCblock, int Size, int BlockSize); // Task 10
void TestResult(double* pAMatrix, double* pBMatrix, double* pCMatrix, int Size); // Task 11

// --- Debug & Serial Helper Functions ---
void TestBlocks(double* pBlock, int BlockSize, const char* str);
void RandomDataInitialization(double* pAMatrix, double* pBMatrix, int Size); // Task 11
void PrintMatrix(double* pMatrix, int Rows, int Cols);
void SerialResultCalculation(double* pAMatrix, double* pBMatrix, double* pCMatrix, int Size);


int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    setvbuf(stdout, 0, _IONBF, 0);

    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    GridSize = sqrt((double)ProcNum);
    if (ProcNum != GridSize * GridSize) {
        if (ProcRank == 0) {
            printf("Number of processes must be a perfect square \n");
        }
    } else {
        if (ProcRank == 0) {
            printf("Parallel matrix multiplication program\n");
        }

        CreateGridCommunicators(); // Task 2

        ProcessInitialization(pAMatrix, pBMatrix, pCMatrix, pAblock, pBblock,
                              pCblock, pMatrixABlock, Size, BlockSize); // Task 3

        if (ProcRank == 0) {
            printf("\nAll inputs received. Starting calculations...\n");
        }

        //if (ProcRank == 0) {
            //COMMENT IF BIG
            //printf("\nInitial A Matrix \n");
            //PrintMatrix(pAMatrix, Size, Size);
            //printf("\nInitial B Matrix \n");
            //PrintMatrix(pBMatrix, Size, Size);
        //}

        DataDistribution(pAMatrix, pBMatrix, pMatrixABlock, pBblock, Size, BlockSize); // Task 5

        //COMMENT IF BIG
        //TestBlocks(pMatrixABlock, BlockSize, "Initial blocks of matrix A");
        //TestBlocks(pBblock, BlockSize, "Initial blocks of matrix B");

        // Ensure all processes are ready before starting the timer
        MPI_Barrier(MPI_COMM_WORLD);

        // Start timer
        Start = MPI_Wtime();

        ParallelResultCalculation(pAblock, pMatrixABlock, pBblock, pCblock, BlockSize);

        // Stop timer
        Finish = MPI_Wtime();
        Duration = Finish - Start;

        // Gather results
        ResultCollection(pCMatrix, pCblock, Size, BlockSize);

        //COMMENT IF BIG
        //TestBlocks(pCblock, BlockSize, "Result blocks");

        // --- Run Serial Calculation (conditionally) ---
        if (runSerialTest == 1) {
            TestResult(pAMatrix, pBMatrix, pCMatrix, Size);
        } else {
            if (ProcRank == 0) {
                printf("\nVerification: \t\t[SKIPPED]\n");
            }
        }

        //if (ProcRank == 0) {

            //COMMENT IF BIG
            //printf("\nResult C Matrix \n");
            //PrintMatrix(pCMatrix, Size, Size);
        //}

        // --- Print All Experiment Data ---
        if (ProcRank == 0) {
            printf("\n--- EXPERIMENT RESULTS ---\n");
            printf("Matrix Size: \t\t%d\n", Size);

            if (runSerialTest == 1) {
                printf("Serial Time: \t\t%f s\n", SerialDuration);
                printf("Parallel Time (%d ps): \t%f s\n", ProcNum, Duration);
                printf("Speedup (%d ps): \t\t%f\n", ProcNum, SerialDuration / Duration);
            } else {
                // Only print parallel time if serial wasn't run
                printf("Parallel Time (%d ps): \t%f s\n", ProcNum, Duration);
            }
            printf("---------------------------\n");
        }

        ProcessTermination(pAMatrix, pBMatrix, pCMatrix, pAblock, pBblock,
                           pCblock, pMatrixABlock); // Task 4
    }

    MPI_Finalize();
    return 0;
}


// Function Definitions

void CreateGridCommunicators() {
    int DimSize[2];
    int Periodic[2];
    int Subdims[2];

    DimSize[0] = GridSize;
    DimSize[1] = GridSize;
    Periodic[0] = 1;
    Periodic[1] = 1;

    MPI_Cart_create(MPI_COMM_WORLD, 2, DimSize, Periodic, 1, &GridComm);
    MPI_Cart_coords(GridComm, ProcRank, 2, GridCoords);

    Subdims[0] = 0; // Fix row, free column
    Subdims[1] = 1;
    MPI_Cart_sub(GridComm, Subdims, &RowComm);

    Subdims[0] = 1; // Fix column, free row
    Subdims[1] = 0;
    MPI_Cart_sub(GridComm, Subdims, &ColComm);
}

void ProcessInitialization(double* &pAMatrix, double* &pBMatrix, double* &pCMatrix,
                           double* &pAblock, double* &pBblock, double* &pCblock,
                           double* &pMatrixABlock, int &Size, int &BlockSize) {

    if (ProcRank == 0) {
        do {
            printf("\nEnter the size of matrices:\n");
            fflush(stdout);

            scanf(" %d", &Size);
            if (Size <= 0) {
                printf("\nSize of objects must be greater than 0!\n");
            }
            if (Size % GridSize != 0) {
                printf("Size of matrices must be divisible by the grid size!\n");
            }
        } while (Size <= 0 || (Size % GridSize != 0));
        // NEW: Ask to run serial test
        printf("Run serial calculation for verification? (1 for Yes, 0 for No):\n");
        fflush(stdout);

        scanf(" %d", &runSerialTest);
    }

    MPI_Bcast(&Size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // NEW: Broadcast the test flag to all processes
    MPI_Bcast(&runSerialTest, 1, MPI_INT, 0, MPI_COMM_WORLD);

    BlockSize = Size / GridSize;

    pAblock = new double[BlockSize*BlockSize];
    pBblock = new double[BlockSize*BlockSize];
    pCblock = new double[BlockSize*BlockSize];
    pMatrixABlock = new double[BlockSize*BlockSize];

    for (int i=0; i < BlockSize*BlockSize; i++) {
        pCblock[i] = 0;
    }

    if (ProcRank == 0) {
        pAMatrix = new double[Size*Size];
        pBMatrix = new double[Size*Size];
        pCMatrix = new double[Size*Size];

        // Task 11: Use RandomDataInitialization
        RandomDataInitialization(pAMatrix, pBMatrix, Size);
    }

    /*
    if (ProcRank == 0) {
        while (getchar() != '\n');
    }
    */
}

void ProcessTermination(double* pAMatrix, double* pBMatrix, double* &pCMatrix,
                        double* pAblock, double* pBblock, double* pCblock,
                        double* pMatrixABlock) {

    if (ProcRank == 0) {
        delete[] pAMatrix;
        delete[] pBMatrix;
        delete[] pCMatrix;
    }
    delete[] pAblock;
    delete[] pBblock;
    delete[] pCblock;
    delete[] pMatrixABlock;
}

void CheckerboardMatrixScatter(double* pMatrix, double* pMatrixBlock, int Size, int BlockSize) {
    double* pMatrixRow = new double[BlockSize*Size];

    if (GridCoords[1] == 0) { // First column processes
        MPI_Scatter(pMatrix, BlockSize*Size, MPI_DOUBLE,
                    pMatrixRow, BlockSize*Size, MPI_DOUBLE,
                    0, ColComm);
    }

    for (int i=0; i < BlockSize; i++) {
        MPI_Scatter(pMatrixRow + i*Size, BlockSize, MPI_DOUBLE,
                    &(pMatrixBlock[i*BlockSize]), BlockSize, MPI_DOUBLE,
                    0, RowComm);
    }

    delete[] pMatrixRow;
}

void DataDistribution(double* pAMatrix, double* pBMatrix, double* pMatrixABlock,
                      double* pBblock, int Size, int BlockSize) {
    CheckerboardMatrixScatter(pAMatrix, pMatrixABlock, Size, BlockSize);
    CheckerboardMatrixScatter(pBMatrix, pBblock, Size, BlockSize);
}

void ParallelResultCalculation(double* pAblock, double* pMatrixABlock, double* pBblock,
                               double* pCblock, int BlockSize) {

    for (int iter = 0; iter < GridSize; iter++) {
        ABlockCommunication(iter, pAblock, pMatrixABlock, BlockSize);
        BlockMultiplication(pAblock, pBblock, pCblock, BlockSize);
        BBlockCommunication(pBblock, BlockSize, ColComm);

        // Wait for all processes to finish this iteration first
        MPI_Barrier(MPI_COMM_WORLD);

        // Let only the root process print the progress
        if (ProcRank == 0) {
            printf("... Iteration %d of %d complete.\n", iter + 1, GridSize);
        }

        // Debug prints
        //COMMENT IF BIG
        //if (ProcRank == 0) printf("Iteration number %d \n", iter);
        //TestBlocks(pAblock, BlockSize, "Block of A matrix");
        //TestBlocks(pBblock, BlockSize, "Block of B matrix");
    }
}

void ABlockCommunication(int iter, double* pAblock, double* pMatrixABlock, int BlockSize) {
    int Pivot = (GridCoords[0] + iter) % GridSize;

    if (GridCoords[1] == Pivot) {
        for (int i=0; i < BlockSize*BlockSize; i++) {
            pAblock[i] = pMatrixABlock[i];
        }
    }
    MPI_Bcast(pAblock, BlockSize*BlockSize, MPI_DOUBLE, Pivot, RowComm);
}

/**
 * @brief Performs a cyclic UPWARD shift of the B blocks.
 * (This version correctly sends UP and receives from DOWN)
 */
void BBlockCommunication(double* pBblock, int BlockSize, MPI_Comm ColComm) {
    MPI_Status Status;
    int rank_source_FROM_DOWN; // The rank of the process below us
    int rank_dest_TO_UP;       // The rank of the process above us

    // We want to shift UP. A displacement of -1 finds the ranks
    // for an upward shift.
    //
    // MPI_Cart_shift(comm, direction, displacement, *rank_source, *rank_dest)
    //
    // With disp = -1 (upward):
    //   - rank_source_FROM_DOWN gets the rank of the process "below" us
    //   - rank_dest_TO_UP       gets the rank of the process "above" us
    MPI_Cart_shift(ColComm, 0, -1, &rank_source_FROM_DOWN, &rank_dest_TO_UP);

    // MPI_Sendrecv_replace(buf, ..., dest, ..., source, ...)
    //
    // We send to our destination (UP)
    // We receive from our source (DOWN)
    MPI_Sendrecv_replace(
        pBblock,
        BlockSize * BlockSize,
        MPI_DOUBLE,
        rank_dest_TO_UP, 0,        // dest = rank of process to send TO (UP)
        rank_source_FROM_DOWN, 0,  // source = rank of process to receive FROM (DOWN)
        ColComm,
        &Status
    );
}

/*
void BBlockCommunication(double* pBblock, int BlockSize, MPI_Comm ColComm) {
    MPI_Status Status;
    int rank_source_up, rank_dest_down;

    // Shift blocks of B upward (send up, receive from down)
    MPI_Cart_shift(ColComm, 0, -1, &rank_source_up, &rank_dest_down);

    MPI_Sendrecv_replace(
        pBblock,
        BlockSize * BlockSize,
        MPI_DOUBLE,
        rank_source_up, 0,        // send upward
        rank_dest_down, 0,        // receive from below
        ColComm,
        &Status
    );
}
*/

void BlockMultiplication(double* pAblock, double* pBblock, double* pCblock, int BlockSize) {
    SerialResultCalculation(pAblock, pBblock, pCblock, BlockSize);
}

/**
 * @brief Gathers all pCblocks into pCMatrix on rank 0.
 * (Corresponds to Task 10)
 */
void ResultCollection(double* pCMatrix, double* pCblock, int Size, int BlockSize) {
    // Temporary buffer for a horizontal stripe (BlockSize rows)
    double* pResultRow = new double[BlockSize*Size];

    // Gather blocks from each row into a stripe
    // The root (0) of each RowComm (i.e., GridCoords[1] == 0)
    // will receive the data.
    for (int i = 0; i < BlockSize; i++) {
        MPI_Gather(pCblock + i*BlockSize, BlockSize, MPI_DOUBLE,
                   pResultRow + i*Size, BlockSize, MPI_DOUBLE,
                   0, RowComm);
    }

    // Gather the stripes from the first column into the full C matrix
    // The root (0) of ColComm (i.e., ProcRank == 0) will receive.
    if (GridCoords[1] == 0) {
        MPI_Gather(pResultRow, BlockSize*Size, MPI_DOUBLE,
                   pCMatrix, BlockSize*Size, MPI_DOUBLE,
                   0, ColComm);
    }

    delete[] pResultRow;
}

/**
 * @brief Compares parallel result with serial calculation on rank 0.
 * (Corresponds to Task 11)
 */
void TestResult(double* pAMatrix, double* pBMatrix, double* pCMatrix, int Size) {
    if (ProcRank == 0) {
        double* pSerialResult; // Result of serial computation
        double Accuracy = 1.e-6; // Comparison accuracy
        int equal = 1; // =1 if matrices are equal
        int i;

        // Variables for serial timing
        double serial_start, serial_finish;

        pSerialResult = new double[Size*Size];
        for (i = 0; i < Size*Size; i++) {
            pSerialResult[i] = 0;
        }

        printf("\nStarting serial calculation... (this may take a while)\n");
        fflush(stdout); // Force the message to print before the long wait

        serial_start = MPI_Wtime();

        // Run the serial calculation
        SerialResultCalculation(pAMatrix, pBMatrix, pSerialResult, Size);

        serial_finish = MPI_Wtime();
        SerialDuration = serial_finish - serial_start; // Store in global var

        //COMMENT IF BIG
        // Print the correct serial result
        //printf("\n--- Serial Result (Correct) ---\n");
        //PrintMatrix(pSerialResult, Size, Size);

        // --- NEW: Signal that serial is done ---
        printf("...Serial calculation complete.\n");

        // Compare
        for (i = 0; i < Size*Size; i++) {
            if (fabs(pSerialResult[i] - pCMatrix[i]) > Accuracy) {
                equal = 0;
                break;
            }
        }

        if (equal == 1) {
            printf("\nVerification: \t\t[SUCCESS] Results are identical.\n");
        } else {
            printf("\nVerification: \t\t[FAILURE] Results are NOT identical.\n");
        }

        delete[] pSerialResult;
    }
}


// --- Debug & Serial Helper Functions ---

void TestBlocks(double* pBlock, int BlockSize, const char* str) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (ProcRank == 0) {
        printf("\n%s\n", str);
    }
    for (int i = 0; i < ProcNum; i++) {
        if (ProcRank == i) {
            printf("ProcRank = %d\n", ProcRank);
            PrintMatrix(pBlock, BlockSize, BlockSize);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

/**
 * @brief Serial function to fill matrices with random values.
 * (From serial lab, for Task 11)
 */
void RandomDataInitialization(double* pAMatrix, double* pBMatrix, int Size) {
    int i;
    srand(unsigned(clock()));
    for (i = 0; i < Size*Size; i++) {
        pAMatrix[i] = rand() / double(RAND_MAX);
        pBMatrix[i] = rand() / double(RAND_MAX);
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