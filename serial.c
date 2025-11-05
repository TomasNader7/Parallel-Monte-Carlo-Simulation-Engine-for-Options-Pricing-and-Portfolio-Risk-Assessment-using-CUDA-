
/*
Input that will be hardcoded in serial.c
S0 – stock price today
K – strike price
r – risk-free interest rate
sigma – volatility
T – time to maturity (years)
n_paths – how many simulations (e.g. 100000) 
n_steps – how many time steps in each path (e.g. 252)
Option_type – call or put
*/

#include <stdio.h>      // For input/output operations
#include <stdlib.h>     // For random number generation and memory allocation
#include <time.h>       // For seeding the random number generator
#include <math.h>  

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif



//-------------------------------------------------------------------------
// Standard normal random variable generator using Box-Muller transform
//-------------------------------------------------------------------------

double generate_random_normal() {
    // Box-Muller transform to generate a standard normal random variable
    double u1 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
    double u2 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

//-------------------------------------------------------------------------
// Monte Carlo simulation for European option pricing
//-------------------------------------------------------------------------
double monte_carlo_option(double S0, double K, double r, double sigma, double T, int n_paths, int n_steps, char Option_type) 
{
    double dt = T / n_steps;
    double drift = (r - 0.5 * sigma * sigma) * dt;
    double vol = sigma * sqrt(dt);
    double sum_payoff = 0.0;
    int i, step;
    double S, z, price;

    for (i = 0; i < n_paths; i++) {
        S = S0;

        for (step = 0; step < n_steps; step++) {
            z = generate_random_normal();
            S = S * exp(drift + vol * z);
        }
    double payoff;
    if (Option_type == 'c'|| Option_type == 'C') {
        payoff = fmax(S - K, 0.0);  // Call option payoff
    } else {
        payoff = fmax(K - S, 0.0);  // Put option payoff
    }
        sum_payoff += payoff;
    }

    double mean_payoff = sum_payoff / n_paths;
    price = exp(-r * T) * mean_payoff;  // Discounted expected payoff
    return price;

}


int main() {

srand(time(NULL));

double S0 = 100.0;     // stock price today
double K = 105.0;      // strike price
double r = 0.03;      // risk-free interest rate
double sigma = 0.2;  // volatility
double T = 1.0;      // time to maturity (years)
int n_paths = 1000000;  // how many simulations (e.g. 100000)
int n_steps = 252;  // how many time steps in each path (e.g. 252
char Option_type = 'C';
 // call or put ('C' for call, 'P' for put)

double price = monte_carlo_option(S0, K, r, sigma, T, n_paths, n_steps, Option_type);

printf("Monte Carlo price: %f\n", price);
printf("Inputs: S0=%.2f K=%.2f r=%.2f sigma=%.2f T=%.2f\n", S0, K, r, sigma, T);
return 0;
}