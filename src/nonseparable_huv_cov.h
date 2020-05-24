#include "RcppArmadillo.h"

using namespace std;

//[[Rcpp::export]]
arma::mat vec_to_symmat(const arma::vec& x);

//[[Rcpp::export]]
double xCovHUV_base(double h, double u, double v, const arma::vec& params, int q, int dim);
  
//[[Rcpp::export]]
arma::mat xCovHUV(const arma::mat& coords, const arma::uvec& ind1, const arma::uvec& ind2, 
                  const arma::vec& cparams, const arma::mat& Dmat, bool same=false);

void xCovHUV_inplace(arma::mat& res,
                     const arma::mat& coords, const arma::uvec& ind1, const arma::uvec& ind2, 
                     const arma::vec& cparams, const arma::mat& Dmat, bool same=false);