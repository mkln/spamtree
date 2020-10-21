#include "RcppArmadillo.h"

using namespace std;

//[[Rcpp::export]]
arma::mat vec_to_symmat(const arma::vec& x);

// phi(x) : exp(-c x^\gamma)
// psi(x) : (a x^\alpha + 1)^\beta

inline double fphi(const double& x, const double& c){
  return exp(-c * x);
}

inline double sqrt_fpsi(const double& x, const double& a, const double& beta){
  // square root of psi function
  // used in conjunction with setting alpha=0.5 and gamma=0.5 in covariances
  return exp(0.5*beta * log1p(a*x));// pow(a * x + 1, 0.5*beta);
}


// exponential covariance
arma::mat cexpcov(const arma::mat& x, const arma::mat& y, const double& sigmasq, const double& phi, bool same=false);


void NonspatialUnivariate_inplace(arma::mat& res,
                         const arma::mat& coords, const arma::uvec& ind1, const arma::uvec& ind2, 
                         const double& sigmasq, const arma::vec& kweights, bool same=false);

arma::mat NonspatialUnivariate(const arma::mat& coords, const arma::uvec& ind1, const arma::uvec& ind2, 
                      const double& sigmasq, const arma::vec& kweights, bool same=false);

// Apanasovich & Genton 2010 
// cross-covariances with same autocovariances 
double xCovHUV_base(const double& h, const double& u, const double& v, const arma::vec& params, const int& q, const int& dim);

//[[Rcpp::export]]
arma::mat xCovHUV(const arma::mat& coords, const arma::uvec& ind1, const arma::uvec& ind2, 
                  const arma::vec& cparams, const arma::mat& Dmat, bool same=false);

void xCovHUV_inplace(arma::mat& res,
                     const arma::mat& coords, const arma::uvec& ind1, const arma::uvec& ind2, 
                     const arma::vec& cparams, const arma::mat& Dmat, bool same=false);



// Apanasovich & Genton 2010 
// cross-covariances with different autocovariances 
// eq. (7) p. 20

double C_base(const double& h, const double& u, const double& v, const arma::vec& params, const int& q, const int& dim);

void mvCovAG20107_inplace(arma::mat& res,
                          const arma::mat& coords, 
                          const arma::uvec& qv_block,
                          const arma::uvec& ind1, const arma::uvec& ind2, 
                          const arma::vec& ai1, const arma::vec& ai2, const arma::vec& phi_i, const arma::vec& thetamv, 
                          const arma::mat& Dmat, bool same=false);

//[[Rcpp::export]]
arma::mat mvCovAG20107(const arma::mat& coords, const arma::uvec& qv_block, 
                       const arma::uvec& ind1, const arma::uvec& ind2, 
                       const arma::vec& ai1, const arma::vec& ai2, const arma::vec& phi_i, const arma::vec& thetamv, 
                       const arma::mat& Dmat, bool same=false);

//[[Rcpp::export]]
arma::mat mvCovAG20107_cx(const arma::mat& coords1,
                          const arma::uvec& qv_block1,
                          const arma::mat& coords2,
                          const arma::uvec& qv_block2,
                          const arma::vec& ai1, const arma::vec& ai2,
                          const arma::vec& phi_i, const arma::vec& thetamv,
                          const arma::mat& Dmat, bool same=false);



void mvCovAG20107anis_inplace(arma::mat& res,
                              const arma::mat& coords, 
                              const arma::uvec& qv_block,
                              const arma::uvec& ind1, const arma::uvec& ind2, 
                              const arma::vec& ai1, const arma::vec& ai2, const arma::vec& phi_i, const arma::vec& thetamv, 
                              const arma::mat& Dmat, const arma::mat& S, bool same);
  
//[[Rcpp::export]]
arma::mat mvCovAG20107anis(const arma::mat& coords, const arma::uvec& qv_block, 
                           const arma::uvec& ind1, const arma::uvec& ind2, 
                           const arma::vec& ai1, const arma::vec& ai2, const arma::vec& phi_i, const arma::vec& thetamv, 
                           const arma::mat& Dmat, const arma::mat& S, bool same);