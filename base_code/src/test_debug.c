#include <stdio.h>
#include <math.h>
#include <stddef.h>

int main() {
    double block[64];
    /* Identity test: X[0][0] = 8, rest 0 */
    for (int i = 0; i < 64; i++) block[i] = 0.0;
    block[0] = 8.0;

    /* Print coefficients */
    extern void idct_2d(double *block);
    idct_2d(block);

    printf("Results:\n");
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            printf("%8.5f ", block[i*8+j]);
        }
        printf("\n");
    }

    /* Also check cos(0) */
    printf("\ncos(0) = %.20f\n", cos(0.0));
    printf("cos(pi/16) = %.20f\n", cos(M_PI/16.0));

    return 0;
}
