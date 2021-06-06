#include "RcppArmadillo.h"


using namespace std;


struct CovarianceParams {
  int covariance_model;
  int npars;
  int q;
  
  // for multivariate A+G 2010,7
  int n_cbase;
  arma::vec ai1;
  arma::vec ai2;
  arma::vec phi_i;
  arma::vec thetamv;
  arma::mat Dmat;
  // if elevation is included
  double d_elevation;
  
  // for univariate with several inputs
  double sigmasq;
  arma::vec kweights;
  
  void transform(const arma::vec&);
  
  CovarianceParams(int, int, int);
  CovarianceParams();
  
};


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
                                  const CovarianceParams& covpars, bool same=false);

arma::mat NonspatialUnivariate(const arma::mat& coords, const arma::uvec& ind1, const arma::uvec& ind2, 
                               const CovarianceParams& covpars, bool same=false);


// Apanasovich & Genton 2010 
// cross-covariances with different autocovariances 
// eq. (7) p. 20

double C_base(const double& h, const double& u, const double& v, const arma::vec& params, const int& q, const int& dim);

void mvCovAG20107_inplace(arma::mat& res,
                          const arma::mat& coords, 
                          const arma::uvec& qv_block,
                          const arma::uvec& ind1, const arma::uvec& ind2, 
                          const CovarianceParams& covpars, bool same=false);

arma::mat mvCovAG20107(const arma::mat& coords, const arma::uvec& qv_block, 
                       const arma::uvec& ind1, const arma::uvec& ind2, 
                       const CovarianceParams& covpars, bool same=false);


// like above, adds elevation
void mvWithElevation_inplace(arma::mat& res,
                             const arma::mat& coords, 
                             const arma::uvec& qv_block,
                             const arma::uvec& ind1, const arma::uvec& ind2, 
                             const CovarianceParams& covpars, bool same=false);

// generics

void Covariancef_inplace(arma::mat& res,
                         const arma::mat& coords, const arma::uvec& qv_block, 
                         const arma::uvec& ind1, const arma::uvec& ind2, 
                         const CovarianceParams& covpars, bool same=false);

arma::mat Covariancef(
    const arma::mat& coords, const arma::uvec& qv_block, 
    const arma::uvec& ind1, const arma::uvec& ind2, 
    const CovarianceParams& covpars, bool same=false);


//[[Rcpp::export]]
arma::mat CrossCovarianceAG10(arma::mat coords1,
                              arma::uvec mv1,
                              arma::mat coords2,
                              arma::uvec mv2,
                              arma::vec ai1, 
                              arma::vec ai2,
                              arma::vec phi_i, 
                              arma::vec thetamv,
                              arma::mat Dmat);