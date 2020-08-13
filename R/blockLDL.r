
blockLDL <- function(X, blist){
  X<-as.matrix(X)
  L <- diag(nrow(X))
  D <- matrix(0, nrow(X), nrow(X))
  
  for(b in 1:length(blist)){
    bhere <- blist[[b]]
    bother <- do.call(c, blist[(b+1):length(blist)])
    D[bhere, bhere] <- X[bhere, bhere]
    L[bother, bhere] <- X[bother, bhere] %*% solve(D[bhere, bhere])
    X[bother, bother] <- X[bother, bother] - L[bother, bhere,drop=F] %*% D[bhere, bhere] %*% t(L[bother, bhere])
  }
  return(list(D=D, L=L))
}


blockrLDL <- function(X, blist){
  X<-as.matrix(X)
  L <- diag(nrow(X))
  D <- matrix(0, nrow(X), nrow(X))
  
  for(b in length(blist):1){
    bhere <- blist[[b]]
    bother <- do.call(c, blist[1:(b-1)])
    D[bhere, bhere] <- X[bhere, bhere]
    L[bother, bhere] <- X[bother, bhere] %*% solve(D[bhere, bhere])
    X[bother, bother] <- X[bother, bother] - L[bother, bhere,drop=F] %*% D[bhere, bhere] %*% t(L[bother, bhere])
  }
  return(list(D=D, L=L))
}


block_ltri_solve <- function(X, blist){
  X <- as.matrix(X)
  nn <- length(blist)
  
  for(i in 1:(nn-1)){
    bherei <- blist[[i]]
    X[bherei, bherei] <- solve(X[bherei, bherei, drop=F])
    for(j in (i+1):nn){
      bherej <- blist[[j]]
      bhereo <- do.call(c, blist[i:(j-1)])
      X[bherej, bherei] <- -X[bherej, bhereo,drop=F] %*% X[bhereo, bherei,drop=F] %*% X[bherei, bherei,drop=F]
    }
  }
  bherei <- blist[[nn]]
  X[bherei, bherei] <- solve(X[bherei, bherei, drop=F])
  return(as(X, "dgCMatrix"))
}

block_ltri_solve2 <- function(X, blist){
  # 1 in block diag
  X <- as.matrix(X)
  nn <- length(blist)
  
  for(i in 1:(nn-1)){
    bherei <- blist[[i]]
    for(j in (i+1):nn){
      bherej <- blist[[j]]
      bhereo <- do.call(c, blist[i:(j-1)])
      X[bherej, bherei] <- -X[bherej, bhereo,drop=F] %*% X[bhereo, bherei,drop=F]
    }
  }
  
  return(as(X, "dgCMatrix"))
}