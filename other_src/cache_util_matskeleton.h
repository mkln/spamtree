#include <RcppArmadillo.h>

inline arma::uvec sortixnotie(const arma::vec& x, 
                              const arma::vec& y, 
                              const arma::vec& z) {
  // Order the elements of x by sorting y and z;
  // we order by y unless there's a tie, then order by z.
  // First create a vector of indices
  arma::uvec idx = arma::regspace<arma::uvec>(0, x.size() - 1);
  // Then sort that vector by the values of y and z
  std::sort(idx.begin(), idx.end(), [&](int i, int j){
    if ( y[i] == y[j] ) {
      return z[i] < z[j];
    }
    return y[i] < y[j];
  });
  // And return x in that order
  return idx;
}

//[[Rcpp::export]]
arma::umat lexisorter_rows(const arma::mat& x, double prec= 1e-6);
//arma::umat lexisorter(const arma::mat& x);

//[[Rcpp::export]]
arma::mat matskeleton(const arma::mat& x);
arma::mat matskeleton(const arma::mat& x, const arma::umat& permrows, const arma::umat& permcols);

arma::mat restore_from_skeleton(const arma::mat& skeleton, const arma::umat& permrows, const arma::umat& permcols);
  
//[[Rcpp::export]]
bool matsame(const arma::mat& x, const arma::mat& y, double tol = 1e-5);

//[[Rcpp::export]]
arma::uvec mat_sortix(const arma::mat& x, const arma::urowvec& ascending);

arma::mat arma_matsort(const arma::mat& x, const arma::urowvec& ascending);

arma::vec drowcol_uv(const arma::field<arma::umat>& diag_blocks);

arma::umat field_v_concat_um(arma::field<arma::umat> const& fuv);

arma::umat block_rotation_group(const arma::mat& coords, const arma::field<arma::uvec>& indexing,
                                const arma::field<arma::uvec>& parents, const arma::vec& block_names);

arma::field<arma::umat> parents_indexing_order(const arma::mat& coords, 
                                               const arma::uvec& qmv_id,
                                               const arma::umat& rot_groups, 
                                               const arma::field<arma::uvec>& indexing,
                                               const arma::field<arma::uvec>& indexing_obs,
                                               const arma::field<arma::uvec>& parents,
                                               const arma::vec& block_names);

arma::field<arma::umat> indexing_order(const arma::mat& coords, 
                                       const arma::uvec& qmv_id,
                                       const arma::umat& rot_groups, 
                                       const arma::field<arma::uvec>& indexing,
                                       const arma::field<arma::uvec>& parents,
                                       const arma::vec& block_names);