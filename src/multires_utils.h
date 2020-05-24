#ifndef MRES_UTILS 
#define MRES_UTILS

#include "RcppArmadillo.h"


using namespace std;

arma::vec armarowsum(const arma::mat& x);

arma::vec armacolsum(const arma::mat& x);

arma::sp_mat Zify(const arma::mat& x);


bool compute_block(bool predicting, int block_ct, bool rfc);

// everything that changes during MCMC
struct MeshData {
  
  arma::vec theta; 
  
  arma::vec wcore; 
  
  arma::field<arma::mat> Kxx_inv;
  arma::field<arma::mat> Kxx_invchol;
  arma::field<arma::mat> Rcc_invchol;
  arma::field<arma::mat> Kxc;
  arma::field<arma::mat> w_cond_mean_K;
  arma::field<arma::mat> w_cond_prec;
  
  arma::vec logdetCi_comps;
  double logdetCi;
  
  arma::vec loglik_w_comps;
  double loglik_w;
  
  arma::uvec track_chol_fails;
  bool cholfail;
  
};

arma::mat join_horiz_mult(const arma::field<arma::mat>& blocks);

arma::mat join_vert_mult(const arma::field<arma::mat>& blocks);

arma::mat invsympd_block(const arma::mat& Ai, const arma::mat& B, const arma::mat& D);

arma::mat invchol_block(const arma::mat& LAi, const arma::mat& B, const arma::mat& D);

void invchol_block_inplace(
    arma::mat& output_reg,
    arma::mat& output_inv,
    const arma::mat& LA, const arma::mat& LAiBt, 
    const arma::mat& cholSchur);

void invchol_block_inplace_direct(
    arma::mat& output_inv,
    const arma::mat& LAi, const arma::mat& C_times_LAi, 
    const arma::mat& invcholSchur);

#endif

