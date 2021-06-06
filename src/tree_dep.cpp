#include <RcppArmadillo.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "R.h"
#include <stdexcept>
#include <string>

#include "find_nan.h"

using namespace std;

//[[Rcpp::export]]
arma::vec kthresholds(arma::vec x,
                      int k){
  arma::vec res(k-1);
  
  for(unsigned int i=1; i<k; i++){
    unsigned int Q1 = i * x.n_elem / k;
    std::nth_element(x.begin(), x.begin() + Q1, x.end());
    res(i-1) = x(Q1);
  }
  
  return res;
}


//[[Rcpp::export]]
Rcpp::StringMatrix col_to_string(const arma::imat& X){
  Rcpp::StringMatrix S(X.n_rows, X.n_cols);
  
  for(unsigned int i=0; i<X.n_rows; i++){
    for(unsigned int j=0; j<X.n_cols; j++){
      S(i, j) = to_string((X(i,j)-1) % 2);
    }
  }
  return S;
}

arma::vec column_threshold(const arma::vec& col1, const arma::vec& thresholds){
  //col %>% sapply(function(x) as.character(1+sum(x >= thresholds)))
  arma::vec result = arma::zeros(col1.n_elem);
  for(unsigned int i=0; i<col1.n_elem; i++){
    int overthreshold = 1;
    for(unsigned int j=0; j<thresholds.n_elem; j++){
      if(col1(i) >= thresholds(j)){
        overthreshold += 1;
      }
    }
    result(i) = overthreshold;
  }
  return result;
}

//[[Rcpp::export]]
arma::mat part_axis_parallel_lmt(const arma::mat& coords, const arma::field<arma::vec>& thresholds){
  //Rcpp::Rcout << "~ Axis-parallel partitioning for LMTrees... ";
  arma::mat resultmat = arma::zeros(arma::size(coords));
  
  for(unsigned int j=0; j<coords.n_cols; j++){
    resultmat.col(j) = column_threshold(coords.col(j), thresholds(j));
  }
  //Rcpp::Rcout << "done." << endl;
  return resultmat;
}

arma::vec unique_finite(const arma::vec& parchiv){
  return arma::unique(parchiv.elem(arma::find_finite(parchiv)));
}


//[[Rcpp::export]]
Rcpp::List make_edges(const arma::mat& parchimat, const arma::uvec& non_empty_blocks, const arma::uvec& res_is_ref){
  
  // empty blocks have parents but shouldnt be children of parents (as they are to be predicted)
  
  int L = parchimat.n_cols;
  int n_blocks = arma::max(parchimat.col(L-1));
  
  arma::field<arma::uvec> parents(n_blocks);
  arma::field<arma::uvec> children(n_blocks);
  for(int i=0; i<n_blocks; i++){
    parents(i) = arma::zeros<arma::uvec>(0);
    children(i) = arma::zeros<arma::uvec>(0);
  }
  arma::uvec reference_res = arma::find(res_is_ref == 1);
  
  for(int lev=0; lev<L; lev ++){
    // loop the levels of the tree and find the names of the blocks at this level
    arma::vec blocks_this_lev = unique_finite(parchimat.col(lev)); //arma::unique(parchimat.col(lev)); 
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for(unsigned int b=0; b<blocks_this_lev.n_elem; b++){
      int u = blocks_this_lev(b)-1; // index of this block in C
      // zoom into the relevant portion to look into for this block name
      arma::mat sub_parchi = parchimat.rows( arma::find(parchimat.col(lev) == blocks_this_lev(b)) );
      // this block is not among the empty ones. so give it children.
      // this block is a reference set so it can have children
      if((res_is_ref(lev) == 1) & (lev < L-1)){
        // all previous resolutions
        arma::uvec possible_children = arma::conv_to<arma::uvec>::from(
          unique_finite(arma::vectorise(sub_parchi.cols(lev+1, sub_parchi.n_cols-1)))) - 1;
        children(u) = arma::intersect(possible_children, non_empty_blocks-1);
        // just one resolution ahead: children are one column to the right
        //arma::vec block_children = unique_finite(sub_parchi.col(lev+1)) - 1; // minus 1 for indexing
        //children(u) = arma::conv_to<arma::uvec>::from(block_children);
      }
      
      // single parent one column to the left
      if(lev > 0){
        // all previous resolutions
        arma::uvec colselect = reference_res.n_elem > 0 ? 
                arma::find(reference_res<lev) : arma::regspace<arma::uvec>(0, lev-1);
        
        arma::vec possible_parents = unique_finite(arma::vectorise(sub_parchi.cols(colselect)));
        parents(u) = arma::conv_to<arma::uvec>::from(possible_parents) - 1;
        // just the previous resolution
        //parents(u) = sub_parchi(0,lev-1)-1;
      }
    }
  }
  
  return Rcpp::List::create(
    Rcpp::Named("parents") = parents,
    Rcpp::Named("children") = children
  );
}

//[[Rcpp::export]]
Rcpp::List make_edges_limited(const arma::mat& parchimat, const arma::uvec& non_empty_blocks, const arma::uvec& res_is_ref){
  
  // empty blocks have parents but shouldnt be children of parents (as they are to be predicted)
  
  int L = parchimat.n_cols;
  int n_blocks = arma::max(parchimat.col(L-1));
  
  arma::field<arma::uvec> parents(n_blocks);
  arma::field<arma::uvec> children(n_blocks);
  for(int i=0; i<n_blocks; i++){
    parents(i) = arma::zeros<arma::uvec>(0);
    children(i) = arma::zeros<arma::uvec>(0);
  }
  arma::uvec reference_res = arma::find(res_is_ref == 1);
  
  for(int lev=0; lev<L; lev ++){
    // loop the levels of the tree and find the names of the blocks at this level
    arma::vec blocks_this_lev = unique_finite(parchimat.col(lev)); //arma::unique(parchimat.col(lev)); 
    for(unsigned int b=0; b<blocks_this_lev.n_elem; b++){
      int u = blocks_this_lev(b)-1; // index of this block in C
      // zoom into the relevant portion to look into for this block name
      arma::mat sub_parchi = parchimat.rows( arma::find(parchimat.col(lev) == blocks_this_lev(b)) );
      // this block is not among the empty ones. so give it children.
      // this block is a reference set so it can have children
      if((res_is_ref(lev) == 1) & (lev < L-1)){
        // all previous resolutions
        arma::uvec possible_children = arma::conv_to<arma::uvec>::from(
          unique_finite(arma::vectorise(sub_parchi.col(lev+1)))) -1; //, sub_parchi.n_cols-1)))) - 1;
        children(u) = arma::intersect(possible_children, non_empty_blocks-1);
        // just one resolution ahead: children are one column to the right
        //arma::vec block_children = unique_finite(sub_parchi.col(lev+1)) - 1; // minus 1 for indexing
        //children(u) = arma::conv_to<arma::uvec>::from(block_children);
      }
      
      // single parent one column to the left
      if(lev > 0){
        // all previous resolutions
        arma::uvec colselect = reference_res.n_elem > 0 ? 
        arma::find(reference_res<lev) : arma::regspace<arma::uvec>(0, lev-1);
        
        int lastcol = colselect(colselect.n_elem - 1);
        arma::vec possible_parents = unique_finite(arma::vectorise(sub_parchi.col(lastcol)));
        parents(u) = arma::conv_to<arma::uvec>::from(possible_parents) - 1;
        // just the previous resolution
        //parents(u) = sub_parchi(0,lev-1)-1;
      }
    }
  }
  
  return Rcpp::List::create(
    Rcpp::Named("parents") = parents,
    Rcpp::Named("children") = children
  );
}


Rcpp::List make_edges_old(const arma::mat& parchimat, const arma::uvec& non_empty_blocks){
  
  // empty blocks have parents but shouldnt be children of parents (as they are to be predicted)
  
  int L = parchimat.n_cols;
  int n_blocks = arma::max(parchimat.col(L-1));
  
  arma::field<arma::uvec> parents(n_blocks);
  arma::field<arma::uvec> children(n_blocks);
  for(int i=0; i<n_blocks; i++){
    parents(i) = arma::zeros<arma::uvec>(0);
    children(i) = arma::zeros<arma::uvec>(0);
  }
  
  for(int lev=0; lev<L; lev ++){
    // loop the levels of the tree and find the names of the blocks at this level
    arma::vec blocks_this_lev = unique_finite(parchimat.col(lev)); //arma::unique(parchimat.col(lev)); 
    for(unsigned int b=0; b<blocks_this_lev.n_elem; b++){
      int u = blocks_this_lev(b)-1; // index of this block in C
      // zoom into the relevant portion to look into for this block name
      arma::mat sub_parchi = parchimat.rows( arma::find(parchimat.col(lev) == blocks_this_lev(b)) );
      
        // this block is not among the empty ones. so give it children.
        if(lev < L-1){
          // all previous resolutions
          arma::uvec possible_children = arma::conv_to<arma::uvec>::from(
            unique_finite(arma::vectorise(sub_parchi.cols(lev+1, sub_parchi.n_cols-1)))) - 1;
          children(u) = arma::intersect(possible_children, non_empty_blocks-1);
          // just one resolution ahead: children are one column to the right
          //arma::vec block_children = unique_finite(sub_parchi.col(lev+1)) - 1; // minus 1 for indexing
          //children(u) = arma::conv_to<arma::uvec>::from(block_children);
        }
      
      // single parent one column to the left
      if(lev > 0){
        // all previous resolutions
        arma::vec possible_parents = unique_finite(arma::vectorise(sub_parchi.cols(0, lev-1)));
        parents(u) = arma::conv_to<arma::uvec>::from(possible_parents) - 1;
        // just the previous resolution
        //parents(u) = sub_parchi(0,lev-1)-1;
      }
    }
  }
  
  return Rcpp::List::create(
    Rcpp::Named("parents") = parents,
    Rcpp::Named("children") = children
  );
}

//[[Rcpp::export]]
arma::umat number_revalue(const arma::umat& original_mat, const arma::uvec& from_val, const arma::uvec& to_val){
  arma::umat output_mat = original_mat;
  int maxval = to_val.max();
  
  for(unsigned int i=0; i<original_mat.n_rows; i++){
    for(unsigned int c=0; c<original_mat.n_cols; c++){
      unsigned int j=0;
      for(j=0; j<from_val.n_elem; j++){
        if(original_mat(i, c) == from_val(j)){
          output_mat(i, c) = to_val(j);
          break;
        }
      }
      if(output_mat(i,c) > maxval){
        output_mat(i,c) = 0;
      }
    }
  }
  return output_mat;
}
