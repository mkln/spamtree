#include "RcppArmadillo.h"
#include <omp.h>
using namespace std;

//[[Rcpp::export]]
arma::mat list_mean(const arma::field<arma::mat>& x){
  // all matrices in x must be the same size.
  int n = x.n_elem;
  int nrows = x(0).n_rows;
  int ncols = x(0).n_cols;
  
  arma::mat result = arma::zeros(nrows, ncols);
  
  #pragma omp parallel for
  for(int j=0; j<nrows*ncols; j++){
    //for(int h=0; h<ncols; h++){
    arma::vec slices = arma::zeros(n);
    for(int i=0; i<n; i++){
      slices(i) = x(i)(j);
    }
    result(j) = arma::mean(slices);
    //}
  }
  return result;
}

void prctile_stl(double* in, const int &len, const double &percent, std::vector<double> &range) {
  // Calculates "percent" percentile.
  // Linear interpolation inspired by prctile.m from MATLAB.
  
  double r = (percent / 100.) * len;
  
  double lower = 0;
  double upper = 0;
  double* min_ptr = NULL;
  int k = 0;
  
  if(r >= len / 2.) {     // Second half is smaller
    int idx_lo = max(r - 1, (double) 0.);
    nth_element(in, in + idx_lo, in + len);             // Complexity O(N)
    lower = in[idx_lo];
    if(idx_lo < len - 1) {
      min_ptr = min_element(&(in[idx_lo + 1]), in + len);
      upper = *min_ptr;
    }
    else
      upper = lower;
  } else {                  // First half is smaller
    double* max_ptr;
    int idx_up = ceil(max(r - 1, (double) 0.));
    nth_element(in, in + idx_up, in + len);             // Complexity O(N)
    upper = in[idx_up];
    if(idx_up > 0) {
      max_ptr = max_element(in, in + idx_up);
      lower = *max_ptr;
    }
    else
      lower = upper;
  }
  
  // Linear interpolation
  k = r + 0.5;        // Implicit floor
  r = r - k;
  range[1] = (0.5 - r) * lower + (0.5 + r) * upper;
  
  min_ptr = min_element(in, in + len);
  range[0] = *min_ptr;
}

double cqtile(arma::vec& v, double q){
  int n = v.n_elem;
  double* a = v.memptr();
  std::vector<double> result(2);
  prctile_stl(a, n, q*100.0, result);
  return result.at(1);
}

//[[Rcpp::export]]
arma::mat list_qtile(const arma::field<arma::mat>& x, double q){
  // all matrices in x must be the same size.
  int n = x.n_elem;
  int nrows = x(0).n_rows;
  int ncols = x(0).n_cols;
  
  arma::mat result = arma::zeros(nrows, ncols);
  
#pragma omp parallel for
  for(int j=0; j<nrows*ncols; j++){
    //for(int h=0; h<ncols; h++){
    arma::vec slices = arma::zeros(n);
    for(int i=0; i<n; i++){
      slices(i) = x(i)(j);
    }
    result(j) = cqtile(slices, q);
    //}
  }
  return result;
}
