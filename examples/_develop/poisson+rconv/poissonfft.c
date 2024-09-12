#include <fftw3.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

// Function to initialize the right-hand side of the Poisson equation
void initialize_rhs(double* rhs, int N, double L) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double x = (i * L) / N;
            double y = (j * L) / N;
            rhs[i * N + j] = sin(M_PI * x) * sin(M_PI * y); // Example: sin(pi*x)*sin(pi*y)
        }
    }
}

int main() {
    int N = 256; // Size of the grid (must be a power of 2)
    double L = 1.0; // Physical size of the domain
    double* rhs = (double*)fftw_malloc(sizeof(double) * N * N); // Right-hand side of the Poisson equation
    fftw_complex* rhs_hat = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N * (N / 2 + 1)); // Fourier space representation of rhs
    fftw_complex* phi_hat = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N * (N / 2 + 1)); // Fourier space representation of the solution
    double* phi = (double*)fftw_malloc(sizeof(double) * N * N); // Solution in real space

    // Initialize the right-hand side
    initialize_rhs(rhs, N, L);

    // Create FFTW plans
    fftw_plan forward_plan = fftw_plan_dft_r2c_2d(N, N, rhs, rhs_hat, FFTW_ESTIMATE);
    fftw_plan backward_plan = fftw_plan_dft_c2r_2d(N, N, phi_hat, phi, FFTW_ESTIMATE);

    // Perform forward FFT
    fftw_execute(forward_plan);

    // Solve Poisson equation in Fourier space
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N / 2 + 1; j++) {
            double kx = (i <= N / 2) ? i : i - N;
            double ky = j;
            double k_squared = kx * kx + ky * ky;
            if (k_squared == 0) {
                phi_hat[i * (N / 2 + 1) + j][0] = 0.0; // Zero mode should be handled separately (set to 0 or some constant)
                phi_hat[i * (N / 2 + 1) + j][1] = 0.0;
            }
            else {
                phi_hat[i * (N / 2 + 1) + j][0] = rhs_hat[i * (N / 2 + 1) + j][0] / k_squared;
                phi_hat[i * (N / 2 + 1) + j][1] = rhs_hat[i * (N / 2 + 1) + j][1] / k_squared;
            }
        }
    }

    // Perform inverse FFT
    fftw_execute(backward_plan);

    // Normalize the solution
    for (int i = 0; i < N * N; i++) {
        phi[i] /= (N * N);
    }

    // Print some of the solution values
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%f ", phi[i * N + j]);
        }
        printf("\n");
    }

    // Clean up
    fftw_destroy_plan(forward_plan);
    fftw_destroy_plan(backward_plan);
    fftw_free(rhs);
    fftw_free(rhs_hat);
    fftw_free(phi_hat);
    fftw_free(phi);

    return 0;
}
