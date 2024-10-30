#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
    // ---- Input and allocate memory for A, F ----
    int NA, NF;
    scanf("%d %d", &NA, &NF);
    int *A = malloc(sizeof(int) * NA);
    int *F = malloc(sizeof(int) * NF);

    for (int i = 0; i < NA; i++) {
        scanf("%d", &A[i]);
    }
    for (int i = 0; i < NF; i++) {
        scanf("%d", &F[i]);
    }
    // ---- End of input and allocation ----

    // Allocate memory for result arrays
    int *R = malloc(sizeof(int) * (NA - NF + 1));
    int *R2 = malloc(sizeof(int) * (NA - NF + 1));

    // Sequential computation
    double time_sequential_start = omp_get_wtime();
    for (int i = 0; i < NA-NF+1; i++) {
        int conv_sum = 0;
        for (int j = 0; j < NF; j++) {
            conv_sum += A[i+j] * F[NF-1-j];
        }
        R2[i] = conv_sum;
    }
    double time_sequential_end = omp_get_wtime();
    double sequential_time = time_sequential_end - time_sequential_start;
    printf("Sequential time: %.6f\n", sequential_time);

    // Parallel computation
    double time_parallel_start = omp_get_wtime();
    omp_set_num_threads(8);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < NA-NF+1; i++) {
        int conv_sum = 0;
        for (int j = 0; j < NF; j++) {
            conv_sum += A[i+j] * F[NF-1-j];
        }
        R[i] = conv_sum;
    }
    double time_parallel_end = omp_get_wtime();
    double parallel_time = time_parallel_end - time_parallel_start;
    printf("Parallel time: %.6f\n", parallel_time);

    // Calculate speedup
    double speedup = sequential_time / parallel_time;
    printf("Speedup: %.6f\n", speedup);

    // Free allocated memory
    free(R);
    free(R2);
    free(F);
    free(A);

    return 0;
}
