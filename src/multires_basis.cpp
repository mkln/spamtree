#include <RcppArmadillo.h>


//[[Rcpp::export]]
arma::sp_mat Jmat(const arma::uvec& x, int p){
  arma::vec values = arma::ones(x.n_elem);
  arma::umat locations = arma::zeros<arma::umat>(2, x.n_elem);
  
  for(unsigned int i=0; i<x.n_elem; i++){
    arma::uvec ll = {i, x(i)-1};
    locations.col(i) = ll;
  }
  arma::sp_mat X(locations, values);
  arma::sp_mat D = arma::eye<arma::sp_mat>(p, p);
  return arma::kron(X, D);
}