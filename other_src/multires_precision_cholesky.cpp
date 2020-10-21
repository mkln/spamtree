#include <RcppArmadillo.h>
#include "covariance_functions.h"

using namespace std;


// keeping these for reference


//[[Rcpp::export]]
arma::mat Ci_ij(
    int block_i, int block_j,
    
    const arma::mat& coords, 
    const arma::uvec& blocking,
    
    const arma::field<arma::uvec>& parents,
    const arma::field<arma::uvec>& children,
    
    const arma::vec& block_names,
    
    const arma::field<arma::uvec>& indexing,
    
    const arma::vec& theta,
    const arma::mat& Dmat){
  // this is a (slow) utility to compute the ij block of the precision matrix
  
  arma::uvec oneuv = arma::ones<arma::uvec>(1);
  
  arma::uvec i_descendants = arma::join_vert(oneuv * block_i-1, children(block_i-1));
  arma::uvec j_descendants = arma::join_vert(oneuv * block_j-1, children(block_j-1));
  
  Rcpp::Rcout << "i_desc\n" << i_descendants << endl
              << "j_desc\n" << j_descendants << endl;
  
  arma::uvec common_descendants = arma::intersect(i_descendants, j_descendants);
  
  Rcpp::Rcout << "common " << endl;
  Rcpp::Rcout << common_descendants << endl;
  
  int n_cd = common_descendants.n_elem;
  
  arma::mat result = arma::zeros(indexing(block_i-1).n_elem, indexing(block_j-1).n_elem);
  
  for(int cd=0; cd < n_cd; cd++){
    int block_k = common_descendants(cd);
    Rcpp::Rcout << "descendant: " << block_k << endl;
    
    //Rcpp::Rcout << indexing(block_k).n_elem << " " << indexing(block_i-1).n_elem << " " << indexing(block_j-1).n_elem << endl;
    
    arma::mat Iki = arma::zeros(indexing(block_k).n_elem, indexing(block_i-1).n_elem);
    if(block_k == block_i-1){
      Iki.diag() += 1;
    }
    arma::mat Ikj = arma::zeros(indexing(block_k).n_elem, indexing(block_j-1).n_elem);
    if(block_k == block_j-1){
      Ikj.diag() += 1;
    }
    
    int n_loc_par = 0;
    for(int p = 0; p<parents(block_k).n_elem; p++){
      n_loc_par += indexing( parents(block_k)(p) ).n_elem;
    }
    
    arma::uvec parents_indexing = arma::zeros<arma::uvec>(n_loc_par);
    // whichone: identifies which parent it is
    arma::uvec parents_whichone = arma::zeros<arma::uvec>(n_loc_par);
    int start_ix = 0;
    //Rcpp::Rcout << "building parent locations " << endl;
    for(int p = 0; p<parents(block_k).n_elem; p++){
      Rcpp::Rcout << "parent " << parents(block_k)(p) << " is n. " << p << endl;
      arma::uvec block_here = indexing(parents(block_k)(p));
      //Rcpp::Rcout << "size: " << block_here.n_elem << endl;
      int n_par_block = block_here.n_elem;
      parents_indexing.rows(start_ix, start_ix + n_par_block-1) = block_here;
      parents_whichone.rows(start_ix, start_ix + n_par_block-1) += p;
      start_ix += n_par_block;
    }
    
    arma::mat Kcc = xCovHUV(coords, indexing(block_k), indexing(block_k), theta, Dmat, true);
    arma::mat Kxxi = arma::inv_sympd(xCovHUV(coords, parents_indexing, parents_indexing, theta, Dmat, true));
    arma::mat Kcx = xCovHUV(coords, indexing(block_k), parents_indexing, theta, Dmat, false);
    arma::mat H_k = Kcx * Kxxi;
    arma::mat Ri_k = arma::inv_sympd(Kcc - Kcx * Kxxi * Kcx.t());
    
    //arma::uvec find_k_in_children_i = arma::find(i_descendants == block_k, 1, "first");
    //arma::uvec find_k_in_children_j = arma::find(j_descendants == block_k, 1, "first");
    
    arma::uvec find_i_in_parents_k = arma::find(parents(block_k) == block_i-1, 1, "first");
    arma::uvec find_j_in_parents_k = arma::find(parents(block_k) == block_j-1, 1, "first");
    
    int pn_i_k = find_i_in_parents_k.n_elem > 0 ? arma::conv_to<int>::from( find_i_in_parents_k ) : -1;
    int pn_j_k = find_j_in_parents_k.n_elem > 0 ? arma::conv_to<int>::from( find_j_in_parents_k ) : -1;
    
    arma::uvec pn_i = arma::find(parents_whichone == pn_i_k);
    arma::uvec pn_j = arma::find(parents_whichone == pn_j_k);
    arma::mat H_k_owed_i = H_k.cols(pn_i);
    arma::mat H_k_owed_j = H_k.cols(pn_j);
    
    if(pn_i.n_elem == 0){
      //Rcpp::Rcout << "No i" << endl;
      H_k_owed_i = arma::zeros(arma::size(Iki));
    }
    if(pn_j.n_elem == 0){
      //Rcpp::Rcout << "No j" << endl;
      H_k_owed_j = arma::zeros(arma::size(Ikj));
    }
    
    
    
    //Rcpp::Rcout << cd << " parset size: " << parents_indexing.n_elem << " " << arma::size(H_k) << endl;
    Rcpp::Rcout << parents_whichone.t() << endl;
    Rcpp::Rcout << "pn: " << pn_i_k << " " << pn_j_k << endl;
    Rcpp::Rcout << pn_i << endl << pn_j << endl;
    //Rcpp::Rcout << arma::size(Iki) << " " << arma::size(Ikj) << endl;
    //Rcpp::Rcout << arma::size(H_k_owed_i) << " " << arma::size(H_k_owed_j) << endl;
    Rcpp::Rcout << "--" << endl;
    
    
    
    
    arma::mat IminusH_ki, IminusH_kj;
    if(block_k != block_i-1){
      IminusH_ki = Iki-H_k_owed_i;
    } else {
      IminusH_ki = Iki;
    }
    if(block_k != block_j-1){
      IminusH_kj = Ikj-H_k_owed_j;
    } else {
      IminusH_kj = Ikj;
    }
    
    result += IminusH_ki.t() * Ri_k * IminusH_kj;
    
    //Rcpp::Rcout << Ri_k << endl;
    Rcpp::Rcout << IminusH_ki.t() << endl;
    
    
  }
  
  return result;
}


//[[Rcpp::export]]
Rcpp::List Ci_udu(const arma::sp_mat& Ci,
                  const arma::field<arma::uvec>& parents,
                  const arma::vec& block_names,
                  const arma::field<arma::uvec>& indexing){
  
  int M_S = parents.n_elem;
  
  arma::mat X(Ci);
  arma::mat D = arma::zeros(arma::size(X));
  arma::mat U = arma::eye(arma::size(X));
  
  for(int j=M_S-1; j>=0; j--){
    int u = block_names(j) - 1;
    
    arma::mat Dtemp = X.submat(indexing(u), indexing(u));
    D.submat(indexing(u), indexing(u)) = Dtemp;
    arma::mat Dinv = arma::inv_sympd(Dtemp);
    
    for(int p=0; p<parents(u).n_elem; p++){
      int pj = parents(u)(p);
      U.submat(indexing(pj), indexing(u)) = X.submat(indexing(pj), indexing(u)) * Dinv;
      for(int g=0; g<parents(u).n_elem; g++){
        int gj = parents(u)(g);
        X.submat(indexing(pj), indexing(gj)) = X.submat(indexing(pj), indexing(gj)) - 
          U.submat(indexing(pj), indexing(u)) * Dtemp * arma::trans(U.submat(indexing(gj), indexing(u)));
        X.submat(indexing(gj), indexing(pj)) = arma::trans(X.submat(indexing(pj), indexing(gj)));
      }
    }
  }
  
  return Rcpp::List::create(
    Rcpp::Named("U") = U,
    Rcpp::Named("D") = D
  );
}


//[[Rcpp::export]]
Rcpp::List Ci_UDU_with_H2(const arma::sp_mat& Ci,
                  const arma::field<arma::uvec>& parents,
                  const arma::vec& block_names,
                  const arma::field<arma::uvec>& indexing){
  
  int M_S = parents.n_elem;
  
  arma::mat X(Ci);
  arma::mat D = arma::zeros(arma::size(X));
  arma::mat I = arma::eye(arma::size(X));
  arma::mat H = arma::zeros(arma::size(X));
  
  for(int j=M_S-1; j>=0; j--){
    int u = block_names(j) - 1;
    
    arma::mat Dtemp = X.submat(indexing(u), indexing(u));
    D.submat(indexing(u), indexing(u)) = Dtemp;
    arma::mat Dinv = arma::inv_sympd(Dtemp);
    
    for(int p=0; p<parents(u).n_elem; p++){
      int pj = parents(u)(p);
      H.submat(indexing(u), indexing(pj)) = - Dinv * X.submat(indexing(u), indexing(pj));
      for(int g=0; g<parents(u).n_elem; g++){
        int gj = parents(u)(g);
        arma::mat Hminus = arma::trans(H.submat(indexing(u), indexing(pj))) * Dtemp * H.submat(indexing(u), indexing(gj));
        Rcpp::Rcout << arma::accu(abs(Hminus)) << endl;
        X.submat(indexing(pj), indexing(gj)) = X.submat(indexing(pj), indexing(gj)) - 
          Hminus;
        X.submat(indexing(gj), indexing(pj)) = arma::trans(X.submat(indexing(pj), indexing(gj)));
      }
    }
  }
  
  return Rcpp::List::create(
    Rcpp::Named("H") = H,
    Rcpp::Named("D") = D
  );
}


//[[Rcpp::export]]
Rcpp::List Ci_UDU_with_H(const arma::sp_mat& Ci,
                         const arma::field<arma::uvec>& parents,
                         const arma::vec& block_names,
                         const arma::field<arma::uvec>& indexing){
  
  int M_S = parents.n_elem;
  
  arma::mat X(Ci);
  arma::mat D = arma::zeros(arma::size(X));
  arma::mat I = arma::eye(arma::size(X));
  arma::mat H = arma::zeros(arma::size(X));
  
  for(int j=M_S-1; j>=0; j--){
    int u = block_names(j) - 1;
    
    arma::mat Dtemp = X.submat(indexing(u), indexing(u));
    D.submat(indexing(u), indexing(u)) = Dtemp;
    arma::mat Dinv = arma::inv_sympd(Dtemp);
    
    for(int p=0; p<parents(u).n_elem; p++){
      int pj = parents(u)(p);
      H.submat(indexing(u), indexing(pj)) = - Dinv * X.submat(indexing(u), indexing(pj));
      
      arma::mat Hpj = - arma::trans(X.submat(indexing(u), indexing(pj)));
      
      for(int g=0; g<parents(u).n_elem; g++){
        int gj = parents(u)(g);
        
        //Rcpp::Rcout << arma::accu(abs(Hminus)) << endl;
        X.submat(indexing(pj), indexing(gj)) = X.submat(indexing(pj), indexing(gj)) - Hpj * H.submat(indexing(u), indexing(gj));
        
        X.submat(indexing(gj), indexing(pj)) = arma::trans(X.submat(indexing(pj), indexing(gj)));
      }
    }
  }
  
  return Rcpp::List::create(
    Rcpp::Named("H") = H,
    Rcpp::Named("D") = D
  );
}



//[[Rcpp::export]]
arma::mat ImHinverse(const arma::mat& ImH,
                                  const arma::field<arma::uvec>& parents,
                                  const arma::field<arma::uvec>& children,
                                  const arma::uvec block_names,
                                  const arma::field<arma::uvec>& indexing){
  
  arma::mat ImHinv = ImH;
  
  //start from res 1 as res 0 has identity
  //for(int res=1; res<n_actual_groups; res++){
  //for(int g=0; g<u_by_block_groups(res).n_elem; g++){
  for(int i=0; i<block_names.n_elem; i++){
    
    int u = block_names(i)-1;
    Rcpp::Rcout << "u: " << u << endl;
    
    for(int p=0; p<parents(u).n_elem; p++){
      int pj = parents(u)(p);
      Rcpp::Rcout << "parent: " << pj << endl;
      arma::mat result = arma::zeros(indexing(u).n_elem, indexing(pj).n_elem);
      
      arma::uvec common_gp = arma::intersect(parents(u), arma::join_vert(arma::ones<arma::uvec>(1)*pj, children(pj)));
      Rcpp::Rcout << "common: " << endl << common_gp << endl;
      for(int gp=0; gp<common_gp.n_elem; gp++){
        int gpar = common_gp(gp);
        result -= ImH.submat(indexing(u), indexing(gpar)) * ImHinv.submat(indexing(gpar), indexing(pj)); 
      }
      //Rcpp::Rcout << "operating on " << endl << indexing(u) << endl << indexing(pj) << endl;
      ImHinv.submat(indexing(u), indexing(pj)) = result;
    }
  }
  
  
  return ImHinv;
}