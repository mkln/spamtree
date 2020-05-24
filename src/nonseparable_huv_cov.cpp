#include "nonseparable_huv_cov.h"
//#include "RcppArmadillo.h"

using namespace std;

arma::mat cexpcov(const arma::mat& x, const arma::mat& y, double sigmasq, double phi, bool same=false){
  // 0 based indexing
  if(same){
    arma::mat pmag = arma::sum(x % x, 1);
    int np = x.n_rows;
    arma::mat K = sigmasq * exp(-phi * sqrt(abs(arma::repmat(pmag.t(), np, 1) + arma::repmat(pmag, 1, np) - 2 * x * x.t())));
    return K;
  } else {
    arma::mat pmag = arma::sum(x % x, 1);
    arma::mat qmag = arma::sum(y % y, 1);
    int np = x.n_rows;
    int nq = y.n_rows;
    arma::mat K = sigmasq * exp(-phi * sqrt(abs(arma::repmat(qmag.t(), np, 1) + arma::repmat(pmag, 1, nq) - 2 * x * y.t())));
    return K;
  }
}

// phi(x) : exp(-c x^\gamma)
// psi(x) : (a x^\alpha + 1)^\beta

double fphi(double x, double c){
  return exp(-c * x);
}

double sqrt_fpsi(double x, double a, const double beta){
  // square root of psi function
  // used in conjunction with setting alpha=0.5 and gamma=0.5 in covariances
  return exp(0.5*beta * log1p(a*x));// pow(a * x + 1, 0.5*beta);
}

double xCovHUV_base(double h, double u, double v, const arma::vec& params, int q, int dim){
  //Rcpp::Rcout << "h: " << h << " u: " << u << " v: " << v << " | q: " << q << " dim: " << dim << endl;
  // Gneiting 2002 uses phi and psi as above ( beta here is kappa in covariance_functions.cpp )
  // hsq = ||h||^2, u = |u|^2 are squared norms. v = d_ij is "distance" between var. i and var. j
  // d=2
  // we also set 
  // alpha=0.5
  // gamma=0.5 
  if(dim == 3){
    if(q == 1){
      // univariate spacetime -- using Gneiting 2002
      // multivariate but fixing a_psi2=1, beta_psi2=1 ?
      // sigmasq + 3 params
      double sigmasq   = params(0);
      double a_psi1    = params(1);
      double beta_psi1 = params(2);
      double c_phi1    = params(3);
      double psi1_sqrt = sqrt_fpsi(u, a_psi1, beta_psi1); // alpha_psi1=0.5
      double phi1      = fphi(h/psi1_sqrt, c_phi1); // gamma_phi1=0.5
      
      return sigmasq / (psi1_sqrt * psi1_sqrt) * phi1;
      
    } else {
      if(q == 2){
        // multivariate but fixing a_psi2=1, beta_psi2=1 ?
        // sigmasq + 3 params
        double sigmasq   = params(0);
        double a_psi2    = 1.0;
        double beta_psi2 = 1.0;
        double a_psi1    = params(1);
        double beta_psi1 = params(2);
        double c_phi1    = params(3);
        double psi2_sqrt = sqrt(v + 1.0);//sqrt_fpsi(v,           a_psi2, beta_psi2); // alpha_psi2=0.5
        
        double psi1_sqrt = sqrt_fpsi(u/psi2_sqrt, a_psi1, beta_psi1); // alpha_psi1=0.5
        double phi1      = fphi(h/psi1_sqrt, c_phi1); // gamma_phi1=0.5
        
        return sigmasq / (psi1_sqrt * psi1_sqrt) * phi1 / psi2_sqrt;
      } else {
        // full multivariate
        // sigmasq + 5 params
        double sigmasq   = params(0);
        double a_psi2    = params(1);
        double beta_psi2 = params(2);
        double a_psi1    = params(3);
        double beta_psi1 = params(4);
        double c_phi1    = params(5);
        double psi2_sqrt = sqrt_fpsi(v,           a_psi2, beta_psi2); // alpha_psi2=0.5
        double psi1_sqrt = sqrt_fpsi(u/psi2_sqrt, a_psi1, beta_psi1); // alpha_psi1=0.5
        double phi1      = fphi(h/psi1_sqrt, c_phi1); // gamma_phi1=0.5
        
        return sigmasq / (psi1_sqrt * psi1_sqrt) * phi1 / psi2_sqrt;
      }
    }
  } else {
    // no time, only space
    if(q > 1){// number of variables - which we consider q=k
      // space-variable nonseparable psi1=const psi2=const in Apanasovich&Genton2010 (4)
      double sigmasq   = params(0);
      double a_psi4    = params(1);
      double beta_psi4 = params(2);
      double a_psi3    = params(3);
      double beta_psi3 = params(4);
      double c_phi1    = params(5);
      double c_phi3    = 1.0;
      
      double psi4_sqrt = sqrt_fpsi(v, a_psi4, beta_psi4);
      double psi3_sqrt = sqrt_fpsi(h, a_psi3, beta_psi3);
      double phi1      = fphi(h, c_phi1);
      double phi3      = fphi(v/psi3_sqrt, c_phi3);
      
      double psi3_sqrt_powq = psi3_sqrt;
      for(int i=0; i<q-1; i++){
        psi3_sqrt_powq *= psi3_sqrt_powq;
      }
      return sigmasq / (psi3_sqrt_powq * psi4_sqrt) * phi1 * phi3;
    } else {
      // 1 variable no time = exp covariance
      double sigmasq   = params(0);
      double phi       = params(1);
      return sigmasq * fphi(h, phi); 
    }
  }

}


arma::mat vec_to_symmat(const arma::vec& x){
  int k = x.n_elem; // = p(p-1)/2
  int p = ( 1 + sqrt(1 + 8*k) )/2;
  
  arma::mat res = arma::zeros(p, p);
  int start_i=1;
  int ix=0;
  for(int j=0; j<p; j++){
    for(int i=start_i; i<p; i++){
      res(i, j) = x(ix);
      ix ++;
    }
    start_i ++;
  } 
  return arma::symmatl(res);
}

void xCovHUV_inplace(arma::mat& res,
                  const arma::mat& coords, const arma::uvec& ind1, const arma::uvec& ind2, 
                  const arma::vec& cparams, const arma::mat& Dmat, bool same){
  int d = coords.n_cols;
  int p = Dmat.n_cols;
  int rix;
  int cix;
  double h;
  double u;
  double v;
  arma::rowvec delta = arma::zeros<arma::rowvec>(d);
    
  if((d == 2) & (p < 2) & false){
    res = cexpcov(coords.rows(ind1), coords.rows(ind2), cparams(0), cparams(1), same);
  } else {
    if(same){
      for(int i=0; i<ind1.n_elem; i++){
        for(int j=i; j<ind2.n_elem; j++){
          //arma::rowvec 
          delta = coords.row(ind1(i)) - coords.row(ind2(j));
          //double 
            h = arma::norm(delta.subvec(0, 1));
          //double 
            u = d > 2? abs(delta(2)) : 0;
          if(p > 1){
            for(int r=0; r<p; r++){
              for(int s=0; s<p; s++){
                //double 
                v = Dmat(r,s);
                //int 
                  rix = i*p + r;
                //int 
                  cix = j*p + s;
                //printf("i: %d, j: %d, r: %d, s: %d", i, j, r, s);
                res(rix, cix) = xCovHUV_base(h, u, v, cparams, p, d);
              }
            }
          } else {
            res(i, j) = xCovHUV_base(h, u, 0, cparams, p, d);
          }
        }
      }
      res = arma::symmatu(res);
    } else {
      //int cc = 0;
      for(int i=0; i<ind1.n_elem; i++){
        for(int j=0; j<ind2.n_elem; j++){
          //arma::rowvec 
          delta = coords.row(ind1(i)) - coords.row(ind2(j));
          //double 
            h = arma::norm(delta.subvec(0, 1));
          //double 
            u = d > 2? abs(delta(2)) : 0;
          if(p > 1){
            for(int r=0; r<p; r++){
              for(int s=0; s<p; s++){
                //double 
                v = Dmat(r,s);
                //int 
                  rix = i*p + r;
                //int 
                  cix = j*p + s;
                //printf("i: %d, j: %d, r: %d, s: %d", i, j, r, s);
                res(rix, cix) = xCovHUV_base(h, u, v, cparams, p, d);
              }
            }
          } else {
            res(i, j) = xCovHUV_base(h, u, 0, cparams, p, d);
          }
        }
      }
      //return res;
  }
  }
}

arma::mat xCovHUV(const arma::mat& coords, const arma::uvec& ind1, const arma::uvec& ind2, 
                  const arma::vec& cparams, const arma::mat& Dmat, bool same){
  
  int q = Dmat.n_cols;
  int n1 = ind1.n_elem;
  int n2 = ind2.n_elem;
  arma::mat res = arma::zeros(n1 * q, n2 * q);
  xCovHUV_inplace(res, coords, ind1, ind2, cparams, Dmat, same);
  return res;
}
