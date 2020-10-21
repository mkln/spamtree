
#include <RcppArmadillo.h>
using namespace std;

// [[Rcpp::export]]
arma::sp_mat dense_to_spatialsparse(const arma::mat& X) {
  arma::umat locations = arma::zeros<arma::umat>(2, X.n_elem);
  arma::vec values = arma::zeros(X.n_elem);
  
  unsigned int ix=0;
  for(unsigned int i=0; i<X.n_rows; i++){
    arma::vec xrow = arma::trans(X.row(i));
    for(int j=0; j<xrow.n_elem; j++){
      arma::uvec xloc({i, ix});
      locations.col(ix) = xloc;
      values(ix) = xrow(j);
      ix ++;
    }
  }
  
  arma::sp_mat res(locations, values);
  return res;
}
