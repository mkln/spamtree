#include <RcppArmadillo.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "R.h"
#include "find_nan.h"
#include "mh_adapt.h"
#include "field_v_concatm.h"
#include "covariance_functions.h"
#include "tree_utils.h"


// with indexing
// without block extensions (obs with NA are left in)
 
using namespace std;

const double hl2pi = -.5 * log(2 * M_PI);

class SpamTreeMV {
public:
  // meta
  int n;
  int p;
  int q;
  int dd;
  int n_blocks;
  int npars;
  
  int covariance_model;
  
  // data
  arma::vec y;
  arma::mat X;
  arma::mat Z;
  
  arma::mat y_available;
  arma::mat X_available;
  
  arma::mat coords;
  arma::uvec mv_id;
  
  // block membership
  arma::uvec blocking;
  arma::uvec gix_block; 
  arma::uvec res_is_ref;
  
  // Z is compact for Z in paper
  // Zblock(u) is paper-Z for block u
  arma::field<arma::sp_mat> Zblock;
  // qvblock(u) lists for every row which components of the 
  // multivariate spatial process are involved
  arma::uvec qvblock_c;
  
  //arma::cube eta_rpx;
  //arma::vec Zw;
  
  // indexing info
  arma::field<arma::uvec> indexing; 
  arma::field<arma::uvec> parents_indexing; 
  arma::field<arma::uvec> children_indexing;
  
  // NA data
  arma::uvec na_ix_all;
  
  // variable data
  arma::field<arma::uvec> ix_by_q;
  arma::field<arma::uvec> ix_by_q_a; // storing indices using only available data
  
  // regression
  arma::field<arma::mat> XtX;
  arma::mat Vi; 
  arma::mat Vim;
  arma::vec bprim;
  
  // dependence
  //arma::field<arma::sp_mat> Ib;
  arma::field<arma::uvec>   parents; // i = parent block names for i-labeled block (not ith block)
  arma::field<arma::uvec>   children; // i = children block names for i-labeled block (not ith block)
  arma::field<arma::uvec>   this_is_jth_child;
  
  arma::vec                 block_names; //  i = block name (+1) of block i. all dependence based on this name
  arma::vec                 block_groups; // same group = sample in parallel given all others
  arma::uvec                block_ct_obs; // 0 if no available obs in this block, >0=count how many available
  arma::uvec                blocks_not_empty; // list i indices of non-empty blocks
  arma::uvec                blocks_predicting;
  arma::uvec                block_is_reference;
  
  int                       n_gibbs_groups;
  int                       n_actual_groups;
  arma::field<arma::vec>    u_by_block_groups;
  arma::vec                 block_groups_labels;
  // for each block's children, which columns of parents of c is u? and which instead are of other parents
  arma::field<arma::field<arma::field<arma::uvec> > > u_is_which_col_f; 
  // for each block's parents, which columns of children of c is u? and which instead are of other parents
  //arma::field<arma::field<arma::field<arma::uvec> > > u_is_which_child; 
  
  arma::field<arma::vec> dim_by_parent;
  //arma::field<arma::vec> dim_by_child;
  
  // params
  arma::vec bigrnorm;
  arma::vec w;
  arma::mat Bcoeff; // sampled
  arma::vec XB;
  arma::vec tausq_inv; // tausq for the l=q variables
  arma::vec tausq_inv_long; // storing tausq_inv at all locations

  // params with mh step
  SpamTreeMVData param_data; 
  SpamTreeMVData alter_data;
  
  // setup
  bool predicting;
  bool limited_tree;
  bool verbose;
  bool debug;
  
  // debug var
  Rcpp::List debug_stuff; 
  
  // init / indexing
  void init_indexing();
  void init_finalize();
  void init_model_data(const arma::vec&);
  
  
  CovarianceParams covpars;
  
  void na_study();
  void make_gibbs_groups();
  
  // MCMC
  char use_alg;
  int max_num_threads;
  
  void find_common_descendants();
  arma::field<arma::uvec> block_common_descendants;
  arma::mat Ci_ij(int, int, SpamTreeMVData& data);
  void fill_precision_blocks(SpamTreeMVData& data);
  void decompose_margin_precision(SpamTreeMVData& data);
  
  void get_loglik_w(SpamTreeMVData& data);
  void get_loglik_w_std(SpamTreeMVData& data);
  
  bool get_loglik_comps_w(SpamTreeMVData& data);
  bool get_loglik_comps_w_std(SpamTreeMVData& data);
  
  
  void deal_with_w(bool);
  void deal_with_beta();
  
  void gibbs_sample_w_std(bool);
  
  //void gibbs_sample_w_rpx();
  
  void gibbs_sample_beta();
  void gibbs_sample_tausq();
  
  void predict(bool);
  void predict_std(bool, bool);
  
  // changing the values, no sampling
  void tausq_update(double);
  void theta_update(SpamTreeMVData&, const arma::vec&);
  void beta_update(const arma::mat&);
  
  // avoid expensive copies
  void accept_make_change();
  
  std::chrono::steady_clock::time_point start_overall;
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;
  std::chrono::steady_clock::time_point end_overall;
  
  // empty
  SpamTreeMV();
  
  // build everything
  SpamTreeMV(
    const arma::mat& y_in, 
    const arma::mat& X_in, 
    const arma::mat& Z_in,
    const arma::mat& coords_in, 
    const arma::uvec& mv_id_in,
    
    const arma::uvec& blocking_in,
    const arma::uvec& gix_block_in,
    const arma::uvec& res_is_ref_in,
    
    const arma::field<arma::uvec>& parents_in,
    const arma::field<arma::uvec>& children_in,
    bool limited_tree_in,
    
    const arma::vec& layers_names,
    const arma::vec& block_group_in,
    
    const arma::field<arma::uvec>& indexing_in,
    
    const arma::mat& w_in,
    const arma::vec& beta_in,
    const arma::vec& theta_in,
    double tausq_inv_in,
    
    char use_alg_in,
    int max_num_threads_in,
    bool v,
    bool debugging);
  
};

