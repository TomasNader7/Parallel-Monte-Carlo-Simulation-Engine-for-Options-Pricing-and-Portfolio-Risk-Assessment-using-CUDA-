
/*
CSC630/730 Advanced Parallel Computing
Enhanced Serial Monte Carlo Simulation for European Option Pricing
Authors: Tomas Nader and Isaac Larbi
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Safe definition of M_PI
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Structure to hold simulation results
typedef struct {
    double price;                   // The calculated option price
    double std_error;               // How uncertain we are about the price
    double conf_interval_lower;     // The range where the true price probably lies (95% confidence)
    double conf_interval_upper;     // The range where the true price probably lies (95% confidence)
    double execution_time;          // How long the calculation took
    double sample_std_dev;          // Standard deviation of all the payoffs
} MonteCarloResult;

//-------------------------------------------------------------------------
// Timer utility using standard C clock()
//-------------------------------------------------------------------------
double get_time() {
    return (double)clock() / CLOCKS_PER_SEC;
}

//-------------------------------------------------------------------------
// Standard normal random variable generator using Box-Muller transform
//-------------------------------------------------------------------------
double generate_random_normal() {
    double u1 = ((double)rand() + 1.0) / ((double)RAND_MAX + 1.0);
    double u2 = ((double)rand() + 1.0) / ((double)RAND_MAX + 1.0);
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

//-------------------------------------------------------------------------
// Monte Carlo simulation with statistical analysis
//-------------------------------------------------------------------------
MonteCarloResult monte_carlo_option(double S0, double K, double r, double sigma, double T, int n_paths, int n_steps, char option_type) 
{
    // Start timer
    MonteCarloResult result;
    double start_time = get_time();

    double dt = T / n_steps;
    double drift = (r - 0.5 * sigma * sigma) * dt;
    double vol = sigma * sqrt(dt);

    double *payoffs = (double*)malloc(n_paths * sizeof(double));
    if (payoffs == NULL) {
    fprintf(stderr, "Memory allocation failed!\n");
    exit(1);
    }

    double sum_payoff = 0.0;
    double sum_payoff_sq = 0.0;

    for (int i = 0; i < n_paths; i++) {
        double S = S0;

        for (int step = 0; step < n_steps; step++) {
            double Z = generate_random_normal();
            S = S * exp(drift + vol * Z);
        }

        double payoff;
        if (option_type == 'c' || option_type == 'C') {
        payoff = fmax(S - K, 0.0);  // Call option
        } else {
        payoff = fmax(K - S, 0.0);  // Put option
        }

        /*
        Example of above code:
        Call Option: Right to BUY at strike price K

        If stock ends at $110 and K=$105, you can buy for $105 and sell for $110 → profit = $5
        If stock ends at $100 and K=$105, you wouldn't use the option → profit = $0
        Formula: max(S - K, 0)

        Put Option: Right to SELL at strike price K

        If stock ends at $100 and K=$105, you can buy for $100 and sell for $105 → profit = $5
        If stock ends at $110 and K=$105, you wouldn't use the option → profit = $0
        Formula: max(K - S, 0)
        */
        
        // Store and accumulate results
        payoffs[i] = payoff;
        sum_payoff += payoff;
        sum_payoff_sq += payoff * payoff;
    }

    // Calculate statistics
    double mean_payoff = sum_payoff / n_paths;
    
    // Sample variance: s² = (1/(N-1)) * Σ(Xi - X̄)²
    double variance = (sum_payoff_sq - (sum_payoff * sum_payoff) / n_paths) / (n_paths - 1);
    double sample_std_dev = sqrt(variance);
    
    // Standard error: SE = s / √N
    double std_error = sample_std_dev / sqrt(n_paths);
    
    // Discounted option price
    double discount_factor = exp(-r * T);
    result.price = discount_factor * mean_payoff;
    result.std_error = discount_factor * std_error;
    result.sample_std_dev = sample_std_dev;
    
    // 95% confidence interval: Price ± 1.96 × SE
    result.conf_interval_lower = result.price - 1.96 * result.std_error;
    result.conf_interval_upper = result.price + 1.96 * result.std_error;
    
    result.execution_time = get_time() - start_time;
    
    free(payoffs);
    return result;
}

//-------------------------------------------------------------------------
// Black-Scholes analytical solution (for validation)
//-------------------------------------------------------------------------
double cumulative_normal(double x) {
    return 0.5 * erfc(-x / sqrt(2.0));
}

double black_scholes_call(double S0, double K, double r, double sigma, double T) {
    double d1 = (log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);
    return S0 * cumulative_normal(d1) - K * exp(-r * T) * cumulative_normal(d2);
}

double black_scholes_put(double S0, double K, double r, double sigma, double T) {
    double d1 = (log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    double d2 = d1 - sigma * sqrt(T);
    return K * exp(-r * T) * cumulative_normal(-d2) - S0 * cumulative_normal(-d1);
}

//-------------------------------------------------------------------------
// Main function with comprehensive testing
//-------------------------------------------------------------------------
int main() {
    srand(time(NULL));
    
    // Input parameters
    double S0 = 100.0;          // Current stock price
    double K = 105.0;           // Strike price
    double r = 0.03;            // Risk-free rate (3%)
    double sigma = 0.2;         // Volatility (20%)
    double T = 1.0;             // Time to maturity (1 year)
    int n_paths = 1000000;      // Number of Monte Carlo paths
    int n_steps = 252;          // Time steps (trading days)
    char option_type = 'P';     // 'C' for Call, 'P' for Put
    
    printf("========================================\n");
    printf("Serial Monte Carlo Option Pricing\n");
    printf("========================================\n\n");
    
    printf("Input Parameters:\n");
    printf("  Stock Price (S0):      $%.2f\n", S0);
    printf("  Strike Price (K):      $%.2f\n", K);
    printf("  Risk-free Rate (r):    %.2f%%\n", r * 100);
    printf("  Volatility (sigma):    %.2f%%\n", sigma * 100);
    printf("  Time to Maturity (T):  %.2f years\n", T);
    printf("  Option Type:           %s\n", option_type == 'C' ? "Call" : "Put");
    printf("  Number of Paths:       %d\n", n_paths);
    printf("  Time Steps per Path:   %d\n\n", n_steps);
    
    // Run Monte Carlo simulation
    MonteCarloResult result = monte_carlo_option(S0, K, r, sigma, T, 
                                                  n_paths, n_steps, option_type);
    
    // Calculate analytical price for comparison
    double analytical_price;
    if (option_type == 'C' || option_type == 'c') {
        analytical_price = black_scholes_call(S0, K, r, sigma, T);
    } else {
        analytical_price = black_scholes_put(S0, K, r, sigma, T);
    }
    
    printf("Results:\n");
    printf("----------------------------------------\n");
    printf("Monte Carlo Price:     $%.6f\n", result.price);
    printf("Standard Error:        $%.6f\n", result.std_error);
    printf("95%% Confidence Int:    [$%.6f, $%.6f]\n", 
           result.conf_interval_lower, result.conf_interval_upper);
    printf("\nAnalytical (B-S):      $%.6f\n", analytical_price);
    printf("Absolute Error:        $%.6f\n", fabs(result.price - analytical_price));
    printf("Relative Error:        %.4f%%\n", 
           100.0 * fabs(result.price - analytical_price) / analytical_price);
    printf("\nExecution Time:        %.4f seconds\n", result.execution_time);
    printf("Paths per Second:      %.2f million\n", n_paths / result.execution_time / 1e6);
    printf("========================================\n");
    
    return 0;
}