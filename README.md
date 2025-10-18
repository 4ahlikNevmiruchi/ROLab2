# THE EXPERIMENT TABLE

| Matrix size | Serial algorithm |    4 ps Time | 4 ps Speed up |    9 ps Time | 9 ps Speed up |  16 ps Time | 16 ps Speed up |
|:-----------:|-----------------:|-------------:|--------------:|-------------:|--------------:|------------:|---------------:|
|     12      |         0.000008 |     0.000017 |      0.451750 |     0.000032 |      0.250393 |    0.000095 |       0.083237 |
|     60      |         0.000788 |     0.000353 |      2.233531 |     0.000244 |      3.294699 |    0.000136 |       6.534849 |
|     120     |         0.005829 |     0.002342 |      2.488448 |     0.001123 |      5.299666 |    0.002741 |       2.199603 |
|     600     |         0.740694 |     0.285644 |      2.593069 |     0.125867 |      5.894945 |    0.098234 |       7.471721 |
|    1 200    |         5.869556 |     2.318942 |      2.531135 |     1.048387 |      5.616578 |    0.727710 |       8.067153 |
|    2 400    |        51.650884 |    18.953957 |      2.725071 |     8.500944 |      6.267537 |    5.415359 |       9.614644 |
|    3 600    |       197.439029 |    72.121885 |      2.737574 |    30.574387 |      6.304565 |   19.587224 |      10.079990 |
|    4 800    |       486.373322 |   186.031425 |      2.614468 |    85.498055 |      5.688706 |   49.006880 |       9.924593 |
|    7 200    |     dnf ~ 30 min | dnf ~ 12 min |  ------------ |   289.093441 |  ------------ |  191.988783 |   ------------ |
|    9 600    |     dnf ~ 78 min | dnf ~ 29 min |  ------------ | dnf ~ 14 min |  ------------ |  471.447814 |   ------------ |

The following results were captured on an 8-core, 16-thread system.

Analysis & Key TakeawaysExcellent Scalability:
The speedup factor consistently increases as we add more processes. 
For the $N=3600$ matrix, moving from 4 processes (2.73x speedup) to 16 processes 
(10.07x speedup) shows that the algorithm effectively divides the work.

Computation vs. Overhead:
The speedup gets better as the matrix size increases.
At $N=120$, the 16-process run is only 2.19x faster.

At N=4800, 
the 16-process run is 9.92x faster.

This is because the computational work (O(N3)) 
grows much faster than the communication overhead (O(N2)). For large matrices, the time spent on "work" 
far outweighs the time spent "communicating," leading to high efficiency.

Small Matrix Anomaly: For tiny matrices (e.g., $N=12$), the parallel version is 
slower than the serial one (speedup < 1). This is expected. 
The time to set up the grid, scatter the data, and gather the results is far greater than the 
time to do the tiny calculation.

Amdahl's Law in Action: The speedup is not "perfect" 
(e.g., 16 processes do not give 16x speedup). 

This is due to the non-parallelizable parts of the program:
Data distribution (scattering A and B) Communication (broadcasting A and shifting B)
Gathering the final result (gathering C)

This overhead, which is the "serial" part of the parallel program, 
limits the maximum theoretical speedup.