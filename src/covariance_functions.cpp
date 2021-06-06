#include "covariance_functions.h"
//#include "RcppArmadillo.h"

using namespace std;

CovarianceParams::CovarianceParams(){
  q = 0;
  npars = 0;
}
CovarianceParams::CovarianceParams(int dd, int q_in, int covmodel=-1){
  q = q_in;
  
  covariance_model = covmodel;
  if(covariance_model == -1){
    // auto choice
    if(dd == 2){
      covariance_model = 0;
      n_cbase = q > 2? 3: 1;
      npars = 3*q + n_cbase;
    } else {
      if(q_in > 1){
        Rcpp::Rcout << "Multivariate on many inputs not implemented yet." << endl;
        throw 1;
      }
      covariance_model = 1;
    }
  }
  if(covariance_model == 2){
    n_cbase = q > 2? 3: 1;
    npars = 3*q + n_cbase + 1; // adds elevation
  }
}

void CovarianceParams::transform(const arma::vec& theta){
  if(covariance_model == 0){
    
    // multivariate spatial
    // from vector to all covariance components
    int k = theta.n_elem - npars; // number of cross-distances = p(p-1)/2
    arma::vec cparams = theta.subvec(0, npars - 1);
    ai1 = cparams.subvec(0, q-1);
    ai2 = cparams.subvec(q, 2*q-1);
    phi_i = cparams.subvec(2*q, 3*q-1);
    thetamv = cparams.subvec(3*q, 3*q+n_cbase-1);
    
    if(k>0){
      Dmat = vec_to_symmat(theta.subvec(npars, npars + k - 1));
    } else {
      Dmat = arma::zeros(1,1);
    }
    
  }
  if(covariance_model == 1){
    // univariate with several inputs
    sigmasq = theta(0);
    kweights = theta.subvec(1, theta.n_elem-1);
  }
  if(covariance_model == 2){
    // multivariate spatial
    // from vector to all covariance components
    int k = theta.n_elem - npars; // number of cross-distances = p(p-1)/2
    arma::vec cparams = theta.subvec(0, npars - 1);
    ai1 = cparams.subvec(0, q-1);
    ai2 = cparams.subvec(q, 2*q-1);
    phi_i = cparams.subvec(2*q, 3*q-1);
    thetamv = cparams.subvec(3*q, 3*q+n_cbase-1);
    d_elevation = cparams(3*q+n_cbase);
    
    if(k>0){
      Dmat = vec_to_symmat(theta.subvec(npars, npars + k - 1));
    } else {
      Dmat = arma::zeros(1,1);
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

// exponential covariance
arma::mat cexpcov(const arma::mat& x, const arma::mat& y, const double& sigmasq, const double& phi, bool same){
  // 0 based indexing
  if(same){
    arma::mat pmag = arma::sum(x % x, 1);
    int np = x.n_rows;
    arma::mat K = sigmasq * exp(-phi * sqrt(abs(arma::repmat(pmag.t(), np, 1) + arma::repmat(pmag, 1, np) - 2 * x * x.t())));
    return K;
  } 
  
  arma::mat pmag = arma::sum(x % x, 1);
  arma::mat qmag = arma::sum(y % y, 1);
  int np = x.n_rows;
  int nq = y.n_rows;
  arma::mat K = sigmasq * exp(-phi * sqrt(abs(arma::repmat(qmag.t(), np, 1) + arma::repmat(pmag, 1, nq) - 2 * x * y.t())));
  return K;
  
}

double C_base(const double& h, const double& u, const double& v, const arma::vec& params, const int& q, const int& dim){
  // no time, only space
  if(q > 2){
    // multivariate  space. v plays role of time of Gneiting 2002
    double a_psi1    = params(0);
    double beta_psi1 = params(1);
    double c_phi1    = params(2);
    double psi1_sqrt = sqrt_fpsi(v, a_psi1, beta_psi1); // alpha_psi1=0.5
    return fphi(h/psi1_sqrt, c_phi1) / (psi1_sqrt * psi1_sqrt);
  } else {
    if(q == 2){
      // multivariate  space. v plays role of time of Gneiting 2002
      double c_phi1    = params(0);
      double psi1_sqrt = sqrt(v + 1);//sqrt_fpsi(v, a_psi1, beta_psi1); // alpha_psi1=0.5
      //return fphi(h/psi1_sqrt, c_phi1) / (psi1_sqrt * psi1_sqrt);
      return fphi(h/psi1_sqrt, c_phi1) / (v+1.0);
    } else {
      // 1 variable no time = exp covariance
      double phi       = params(0);
      return fphi(h, phi); 
    }
  }
}


void mvWithElevation_inplace(arma::mat& res,
                             const arma::mat& coords, 
                             const arma::uvec& qv_block,
                             const arma::uvec& ind1, const arma::uvec& ind2, 
                             const CovarianceParams& covpars, bool same){
  int d = coords.n_cols; // 3
  int p = covpars.Dmat.n_cols;
  int v_ix_i;
  int v_ix_j;
  double h;
  double u;
  double v;
  arma::rowvec delta = arma::zeros<arma::rowvec>(d);
  
  // covpars.ai1 params: cparams.subvec(0, p-1);
  // covpars.ai2 params: cparams.subvec(p, 2*p-1);
  // covpars.phi_i params: cparams.subvec(2*p, 3*p-1);
  // C_base params: cparams.subvec(3*p, 3*p + k - 1); // k = q>2? 3 : 1;
  
  arma::rowvec weights = arma::ones<arma::rowvec>(3);
  //weights(2) = covpars.d_elevation;
  
  if(same){
    for(unsigned int i=0; i<ind1.n_elem; i++){
      v_ix_i = qv_block(ind1(i));
      double ai1_sq = covpars.ai1(v_ix_i) * covpars.ai1(v_ix_i);
      double ai2_sq = covpars.ai2(v_ix_i) * covpars.ai2(v_ix_i);
      arma::rowvec cxi = coords.row(ind1(i));
      
      for(unsigned int j=i; j<ind2.n_elem; j++){
        delta = cxi - coords.row(ind2(j));
        h = arma::norm(weights % delta.subvec(0, 2));
        u = d < 4? 0 : abs(delta(3));
        
        v_ix_j = qv_block(ind2(j));
        v = covpars.Dmat(v_ix_i, v_ix_j);
        
        if(v == 0){ // v_ix_i == v_ix_j
          res(i, j) = ai1_sq * C_base(h, u, 0, covpars.thetamv, p, d) + 
            ai2_sq * fphi(h, covpars.phi_i(v_ix_i));
        } else {
          res(i, j) =  covpars.ai1(v_ix_i) * covpars.ai1(v_ix_j) * C_base(h, u, v, covpars.thetamv, p, d);
        }
      }
    }
    res = arma::symmatu(res);
  } else {
    for(unsigned int i=0; i<ind1.n_elem; i++){
      v_ix_i = qv_block(ind1(i));
      double ai1_sq = covpars.ai1(v_ix_i) * covpars.ai1(v_ix_i);
      double ai2_sq = covpars.ai2(v_ix_i) * covpars.ai2(v_ix_i);
      arma::rowvec cxi = coords.row(ind1(i));
      
      for(unsigned int j=0; j<ind2.n_elem; j++){
        delta = cxi - coords.row(ind2(j));
        h = arma::norm(weights % delta.subvec(0, 2));
        u = d < 4? 0 : abs(delta(3));
        
        v_ix_j = qv_block(ind2(j));
        v = covpars.Dmat(v_ix_i, v_ix_j);
        
        if(v == 0){ // v_ix_i == v_ix_j
          res(i, j) = ai1_sq * C_base(h, u, 0, covpars.thetamv, p, d) + 
            ai2_sq * fphi(h, covpars.phi_i(v_ix_i));
        } else {
          res(i, j) =  covpars.ai1(v_ix_i) * covpars.ai1(v_ix_j) * C_base(h, u, v, covpars.thetamv, p, d);
        }
      }
    }
    //return res;
  }
}



void mvCovAG20107_inplace(arma::mat& res,
                          const arma::mat& coords, 
                          const arma::uvec& qv_block,
                          const arma::uvec& ind1, const arma::uvec& ind2, 
                          const CovarianceParams& covpars, bool same){
  int d = coords.n_cols;
  int p = covpars.Dmat.n_cols;
  if((d == 2) & (p < 2)){
    res = cexpcov(coords.rows(ind1), coords.rows(ind2), covpars.ai1(0), covpars.thetamv(0), same);
  } else {
    int v_ix_i;
    int v_ix_j;
    double h;
    double u;
    double v;
    arma::rowvec delta = arma::zeros<arma::rowvec>(d);
    
    // covpars.ai1 params: cparams.subvec(0, p-1);
    // covpars.ai2 params: cparams.subvec(p, 2*p-1);
    // covpars.phi_i params: cparams.subvec(2*p, 3*p-1);
    // C_base params: cparams.subvec(3*p, 3*p + k - 1); // k = q>2? 3 : 1;
    
    if(same){
      for(unsigned int i=0; i<ind1.n_elem; i++){
        v_ix_i = qv_block(ind1(i));
        double ai1_sq = covpars.ai1(v_ix_i) * covpars.ai1(v_ix_i);
        double ai2_sq = covpars.ai2(v_ix_i) * covpars.ai2(v_ix_i);
        arma::rowvec cxi = coords.row(ind1(i));
        
        for(unsigned int j=i; j<ind2.n_elem; j++){
          delta = cxi - coords.row(ind2(j));
          h = arma::norm(delta.subvec(0, 1));
          u = d < 3? 0 : abs(delta(2));
          
          v_ix_j = qv_block(ind2(j));
          v = covpars.Dmat(v_ix_i, v_ix_j);
          
          if(v == 0){ // v_ix_i == v_ix_j
            res(i, j) = ai1_sq * C_base(h, u, 0, covpars.thetamv, p, d) + 
              ai2_sq * fphi(h, covpars.phi_i(v_ix_i));
          } else {
            res(i, j) =  covpars.ai1(v_ix_i) * covpars.ai1(v_ix_j) * C_base(h, u, v, covpars.thetamv, p, d);
          }
        }
      }
      res = arma::symmatu(res);
    } else {
      for(unsigned int i=0; i<ind1.n_elem; i++){
        v_ix_i = qv_block(ind1(i));
        double ai1_sq = covpars.ai1(v_ix_i) * covpars.ai1(v_ix_i);
        double ai2_sq = covpars.ai2(v_ix_i) * covpars.ai2(v_ix_i);
        arma::rowvec cxi = coords.row(ind1(i));
        
        for(unsigned int j=0; j<ind2.n_elem; j++){
          delta = cxi - coords.row(ind2(j));
          h = arma::norm(delta.subvec(0, 1));
          u = d < 3? 0 : abs(delta(2));
          
          v_ix_j = qv_block(ind2(j));
          v = covpars.Dmat(v_ix_i, v_ix_j);
          
          if(v == 0){ // v_ix_i == v_ix_j
            res(i, j) = ai1_sq * C_base(h, u, 0, covpars.thetamv, p, d) + 
              ai2_sq * fphi(h, covpars.phi_i(v_ix_i));
          } else {
            res(i, j) =  covpars.ai1(v_ix_i) * covpars.ai1(v_ix_j) * C_base(h, u, v, covpars.thetamv, p, d);
          }
        }
      }
      //return res;
    }
  }
  
}

arma::mat mvCovAG20107(const arma::mat& coords, const arma::uvec& qv_block, 
                       const arma::uvec& ind1, const arma::uvec& ind2, 
                       const CovarianceParams& covpars, bool same){
  
  int n1 = ind1.n_elem;
  int n2 = ind2.n_elem;
  arma::mat res = arma::zeros(n1, n2);
  mvCovAG20107_inplace(res, coords, qv_block, ind1, ind2, 
                       covpars, same);
  return res;
}


arma::mat CrossCovarianceAG10(arma::mat coords1,
                              arma::uvec mv1,
                              arma::mat coords2,
                              arma::uvec mv2,
                              arma::vec ai1, 
                              arma::vec ai2,
                              arma::vec phi_i, 
                              arma::vec thetamv,
                              arma::mat Dmat){
  
  arma::mat res = arma::zeros(coords1.n_rows, coords2.n_rows);
  mv1 -= 1;
  mv2 -= 1;
  
  int d = coords1.n_cols;
  int p = Dmat.n_cols;
  if((d == 2) & (p < 2)){
    Rcpp::stop("Invalid Dmat for multivariate data");
    return res;
  } else {
    int v_ix_i;
    int v_ix_j;
    double h;
    double u;
    double v;
    arma::rowvec delta = arma::zeros<arma::rowvec>(d);
    // covpars.ai1 params: cparams.subvec(0, p-1);
    // covpars.ai2 params: cparams.subvec(p, 2*p-1);
    // covpars.phi_i params: cparams.subvec(2*p, 3*p-1);
    // C_base params: cparams.subvec(3*p, 3*p + k - 1); // k = q>2? 3 : 1;
    for(unsigned int i=0; i<coords1.n_rows; i++){
      v_ix_i = mv1(i);
      double ai1_sq = ai1(v_ix_i) * ai1(v_ix_i);
      double ai2_sq = ai2(v_ix_i) * ai2(v_ix_i);
      arma::rowvec cxi = coords1.row(i);
      
      for(unsigned int j=0; j<coords2.n_rows; j++){
        delta = cxi - coords2.row(j);
        h = arma::norm(delta.subvec(0, 1));
        u = d < 3? 0 : abs(delta(2));
        
        v_ix_j = mv2(j);
        v = Dmat(v_ix_i, v_ix_j);
        
        if(v == 0){ // v_ix_i == v_ix_j
          res(i, j) = ai1_sq * C_base(h, u, 0, thetamv, p, d) + 
            ai2_sq * fphi(h, phi_i(v_ix_i));
        } else {
          res(i, j) =  ai1(v_ix_i) * ai1(v_ix_j) * C_base(h, u, v, thetamv, p, d);
        }
      }
    }
    return res;
  }
}


void NonspatialUnivariate_inplace(arma::mat& res,
                                  const arma::mat& coords, const arma::uvec& ind1, const arma::uvec& ind2, 
                                  const CovarianceParams& covpars, bool same){

  if(same){
    for(unsigned int i=0; i<ind1.n_elem; i++){
      arma::rowvec cri = coords.row(ind1(i));
      for(unsigned int j=i; j<ind2.n_elem; j++){
        arma::rowvec deltasq = cri - coords.row(ind2(j));
        double weighted = (arma::accu(covpars.kweights.t() % deltasq % deltasq));
        res(i, j) = covpars.sigmasq * exp(-weighted) + (weighted == 0? 1e-3 : 0);
      }
    }
    res = arma::symmatu(res);
  } else {
    //int cc = 0;
    for(unsigned int i=0; i<ind1.n_elem; i++){
      arma::rowvec cri = coords.row(ind1(i));
      for(unsigned int j=0; j<ind2.n_elem; j++){
        arma::rowvec deltasq = cri - coords.row(ind2(j));
        double weighted = (arma::accu(covpars.kweights.t() % deltasq % deltasq));
        res(i, j) = covpars.sigmasq * exp(-weighted) + (weighted == 0? 1e-3 : 0);
      }
    }
  }
  
}

arma::mat NonspatialUnivariate(const arma::mat& coords, const arma::uvec& ind1, const arma::uvec& ind2, 
                               const CovarianceParams& covpars, bool same){
  int n1 = ind1.n_elem;
  int n2 = ind2.n_elem;
  arma::mat res = arma::zeros(n1, n2);
  NonspatialUnivariate_inplace(res, coords, ind1, ind2, covpars, same);
  return res;
}



void Covariancef_inplace(arma::mat& res,
                         const arma::mat& coords, const arma::uvec& qv_block, 
                         const arma::uvec& ind1, const arma::uvec& ind2, 
                         const CovarianceParams& covpars, bool same){
  
  if(covpars.covariance_model == 0){
    mvCovAG20107_inplace(res, coords, qv_block, ind1, ind2, 
                         covpars, same);
  }
  if(covpars.covariance_model == 1){
    NonspatialUnivariate_inplace(res, coords, ind1, ind2, covpars, same);
  }
  if(covpars.covariance_model == 2){
    mvWithElevation_inplace(res, coords, qv_block, ind1, ind2, covpars, same);
  }
}



arma::mat Covariancef(const arma::mat& coords, const arma::uvec& qv_block, 
                      const arma::uvec& ind1, const arma::uvec& ind2, 
                      const CovarianceParams& covpars, bool same){
  int n1 = ind1.n_elem;
  int n2 = ind2.n_elem;
  arma::mat res = arma::zeros(n1, n2);
  if(covpars.covariance_model < 0){
    Rcpp::stop("Covariance model not implemented");
  }
  if(covpars.covariance_model == 0){
    mvCovAG20107_inplace(res, coords, qv_block, ind1, ind2, 
                         covpars, same);
  }
  if(covpars.covariance_model == 1){
    NonspatialUnivariate_inplace(res, coords, ind1, ind2, covpars, same);
  }
  if(covpars.covariance_model == 2){
    mvWithElevation_inplace(res, coords, qv_block, ind1, ind2, covpars, same);
  }
  return res;
}

