/*
#include <RcppArmadillo.h>
#include <RcppEigen.h>
#include <omp.h>

#include "field_v_concatm.h"
#include "nonseparable_huv_cov.h"

//#define EIGEN_USE_MKL_ALL 1

Eigen::VectorXd armavec_to_vectorxd(arma::vec arma_A) {
  
  Eigen::VectorXd eigen_B = Eigen::Map<Eigen::VectorXd>(arma_A.memptr(),
                                                        arma_A.n_elem);
  return eigen_B;
}

void expand_grid_with_values_(arma::umat& locs,
                              arma::vec& vals,
                              
                              int rowstart, int rowend,
                              const arma::uvec& x1,
                              const arma::uvec& x2,
                              const arma::mat& values){
  
  for(int i=rowstart; i<rowend; i++){
    arma::uvec ix;
    try {
      ix = arma::ind2sub(arma::size(values), i-rowstart);
    } catch (...) {
      Rcpp::Rcout << arma::size(values) << " " << i-rowstart << " " << i <<" " << rowstart << " " << rowend << endl;
      throw 1;
    }
    locs(0, i) = x1(ix(0));
    locs(1, i) = x2(ix(1));
    vals(i) = values(ix(0), ix(1));
  }
}

//[[Rcpp::export]]
Eigen::SparseMatrix<double> spamtree_Cinv(
    const arma::mat& coords, 
    const arma::uvec& blocking,
    
    const arma::field<arma::uvec>& parents,
    
    const arma::vec& block_names,
    
    const arma::field<arma::uvec>& indexing,
    
    const arma::vec& theta,
    const arma::mat& Dmat,
    
    int num_threads = 1,
    
    bool verbose=false,
    bool debug=false){
  
  int n = coords.n_rows;
  
  omp_set_num_threads(num_threads);
  
  int n_blocks = block_names.n_elem;
  
  arma::field<arma::uvec> parents_indexing(n_blocks);
  
  arma::uvec Adims = arma::zeros<arma::uvec>(n_blocks+1);
  arma::uvec Ddims = arma::zeros<arma::uvec>(n_blocks+1);
  
  
//***#pragma omp parallel for
  for(int i=0; i<n_blocks; i++){
    int u = block_names(i)-1;
    if(parents(u).n_elem > 0){
      arma::field<arma::uvec> pixs(parents(u).n_elem);
      for(int pi=0; pi<parents(u).n_elem; pi++){
        pixs(pi) = indexing(parents(u)(pi));//arma::find( blocking == parents(u)(pi)+1 ); // parents are 0-indexed 
      }
      parents_indexing(u) = field_v_concat_uv(pixs);
      Adims(i+1) = indexing(u).n_elem * parents_indexing(u).n_elem;
    }
    Ddims(i+1) = indexing(u).n_elem * indexing(u).n_elem;
  }
  
  int Asize = arma::accu(Adims);
  Adims = arma::cumsum(Adims);
  
  arma::umat Hlocs = arma::zeros<arma::umat>(2, Asize);
  arma::vec Hvals = arma::zeros(Asize);
  
  int Dsize = arma::accu(Ddims);
  Ddims = arma::cumsum(Ddims);
  
  arma::umat Dlocs2 = arma::zeros<arma::umat>(2, Dsize);
  arma::vec Dvals2 = arma::zeros(Dsize);
  
//***#pragma omp parallel for 
  for(int i=0; i<n_blocks; i++){
    int u = block_names(i)-1;
    arma::mat Kcc = xCovHUV(coords, indexing(u), indexing(u), theta, Dmat, true);
    if(parents(u).n_elem > 0){
      arma::mat Kxxi = arma::inv_sympd(xCovHUV(coords, parents_indexing(u), parents_indexing(u), theta, Dmat, true));
      arma::mat Kcx = xCovHUV(coords, indexing(u), parents_indexing(u), theta, Dmat, false);
      arma::mat Hj = Kcx * Kxxi;
      arma::mat Rji = arma::inv( arma::chol( arma::symmatu(Kcc - Kcx * Kxxi * Kcx.t()), "lower"));
      
      expand_grid_with_values_(Hlocs, Hvals, Adims(i), Adims(i+1),
                               indexing(u), parents_indexing(u), Hj);
      
      expand_grid_with_values_(Dlocs2, Dvals2, Ddims(i), Ddims(i+1),
                               indexing(u), indexing(u), Rji);
      
    } else {
      arma::mat Rji = arma::inv( arma::chol( arma::symmatu(Kcc), "lower"));
      expand_grid_with_values_(Dlocs2, Dvals2, Ddims(i), Ddims(i+1),
                               indexing(u), indexing(u), Rji);
      
    }
  }
  
  // EIGEN
  Eigen::SparseMatrix<double> I_eig(n, n);
  I_eig.setIdentity();
  
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList_H;
  std::vector<T> tripletList_Dic2;
  
  tripletList_H.reserve(Hlocs.n_cols);
  for(int i=0; i<Hlocs.n_cols; i++){
    tripletList_H.push_back(T(Hlocs(0, i), Hlocs(1, i), Hvals(i)));
  }
  Eigen::SparseMatrix<double> He(n,n);
  He.setFromTriplets(tripletList_H.begin(), tripletList_H.end());
  
  tripletList_Dic2.reserve(Dlocs2.n_cols);
  for(int i=0; i<Dlocs2.n_cols; i++){
    tripletList_Dic2.push_back(T(Dlocs2(0, i), Dlocs2(1, i), Dvals2(i)));
  }
  Eigen::SparseMatrix<double> Dice2(n,n);
  Dice2.setFromTriplets(tripletList_Dic2.begin(), tripletList_Dic2.end());
  
  Eigen::SparseMatrix<double> L = (I_eig-He).triangularView<Eigen::Lower>().transpose() * Dice2;
  
  return L * L.transpose();
}


//[[Rcpp::export]]
Eigen::VectorXd spamtree_sample(
    const arma::mat& coords, 
    const arma::uvec& blocking,
    
    const arma::field<arma::uvec>& parents,
    
    const arma::vec& block_names,
    
    const arma::field<arma::uvec>& indexing,
    
    const arma::vec& theta,
    const arma::mat& Dmat,
    
    int num_threads = 1,
    
    bool verbose=false,
    bool debug=false){
  
  int n = coords.n_rows;
  
  omp_set_num_threads(num_threads);
  
  int n_blocks = block_names.n_elem;
  
  arma::field<arma::uvec> parents_indexing(n_blocks);
  
  arma::uvec Adims = arma::zeros<arma::uvec>(n_blocks+1);
  arma::uvec Ddims = arma::zeros<arma::uvec>(n_blocks+1);
  
//***#pragma omp parallel for
  for(int i=0; i<n_blocks; i++){
    int u = block_names(i)-1;
    if(parents(u).n_elem > 0){
      arma::field<arma::uvec> pixs(parents(u).n_elem);
      for(int pi=0; pi<parents(u).n_elem; pi++){
        pixs(pi) = indexing(parents(u)(pi));//arma::find( blocking == parents(u)(pi)+1 ); // parents are 0-indexed 
      }
      parents_indexing(u) = field_v_concat_uv(pixs);
      Adims(i+1) = indexing(u).n_elem * parents_indexing(u).n_elem;
    }
    Ddims(i+1) = indexing(u).n_elem * indexing(u).n_elem;
  }
  
  int Asize = arma::accu(Adims);
  Adims = arma::cumsum(Adims);
  
  arma::umat Hlocs = arma::zeros<arma::umat>(2, Asize);
  arma::vec Hvals = arma::zeros(Asize);
  
  int Dsize = arma::accu(Ddims);
  Ddims = arma::cumsum(Ddims);
  
  arma::umat Dlocs2 = arma::zeros<arma::umat>(2, Dsize);
  arma::vec Dvals2 = arma::zeros(Dsize);
  

  #pragma omp parallel for 
  for(int i=0; i<n_blocks; i++){
    int u = block_names(i)-1;
    arma::mat Kcc = xCovHUV(coords, indexing(u), indexing(u), theta, Dmat, true);
    if(parents(u).n_elem > 0){
      arma::mat Kxxi = arma::inv_sympd(xCovHUV(coords, parents_indexing(u), parents_indexing(u), theta, Dmat, true));
      arma::mat Kcx = xCovHUV(coords, indexing(u), parents_indexing(u), theta, Dmat, false);
      arma::mat Hj = Kcx * Kxxi;
      arma::mat Rj = Kcc - Kcx * Kxxi * Kcx.t();
      
      expand_grid_with_values_(Hlocs, Hvals, Adims(i), Adims(i+1),
                               indexing(u), parents_indexing(u), Hj);
      
      expand_grid_with_values_(Dlocs2, Dvals2, Ddims(i), Ddims(i+1),
                               indexing(u), indexing(u), arma::chol(arma::symmatu(Rj), "lower"));
      
    } else {
      expand_grid_with_values_(Dlocs2, Dvals2, Ddims(i), Ddims(i+1),
                               indexing(u), indexing(u), arma::chol(arma::symmatu(Kcc), "lower"));
      
    }
  }
  
  
    
  // EIGEN
  Eigen::SparseMatrix<double> I_eig(n, n);
  I_eig.setIdentity();
  
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList_H;
  std::vector<T> tripletList_Dic2;
  
  tripletList_H.reserve(Hlocs.n_cols);
  for(int i=0; i<Hlocs.n_cols; i++){
    tripletList_H.push_back(T(Hlocs(0, i), Hlocs(1, i), Hvals(i)));
  }
  Eigen::SparseMatrix<double> He(n,n);
  He.setFromTriplets(tripletList_H.begin(), tripletList_H.end());
  
  arma::vec rnorm_sample = arma::randn(n);
  Eigen::VectorXd enormvec = armavec_to_vectorxd(rnorm_sample);
  tripletList_Dic2.reserve(Dlocs2.n_cols);
  for(int i=0; i<Dlocs2.n_cols; i++){
    tripletList_Dic2.push_back(T(Dlocs2(0, i), Dlocs2(1, i), Dvals2(i)));
  }
  Eigen::SparseMatrix<double> Dice2(n,n);
  Dice2.setFromTriplets(tripletList_Dic2.begin(), tripletList_Dic2.end());
  Eigen::MatrixXd enormvecother = Dice2 * enormvec;
  Eigen::VectorXd sampled = (I_eig-He).triangularView<Eigen::Lower>().solve(enormvecother);
  
  return sampled;
}




*/