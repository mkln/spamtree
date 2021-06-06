#include "mh_adapt.h"

arma::vec par_huvtransf_fwd(arma::vec par, const arma::mat& set_unif_bounds){
  for(unsigned int j=0; j<par.n_elem; j++){
    par(j) = logit(par(j), set_unif_bounds(j, 0), set_unif_bounds(j, 1));
  }
  return par;
}

arma::vec par_huvtransf_back(arma::vec par, const arma::mat& set_unif_bounds){
  for(unsigned int j=0; j<par.n_elem; j++){
    par(j) = logistic(par(j), set_unif_bounds(j, 0), set_unif_bounds(j, 1));
  }
  return par;
}