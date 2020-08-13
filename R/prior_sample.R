#' @export
prior_sample <- function(coords, mv_id, ai1, ai2, phi_i, thetamv, Dmat,
                     cell_size=25, K=rep(2, ncol(coords)),
                     num_threads = 4,
                     verbose=F, debug=F){
  
  if(F){
    rm(list=ls())
    xx <- seq(0, 1, length.out=100)
    coords <- expand.grid(xx, xx)
    q <- 4
    coords <- 1:q %>% lapply(function(x) coords) %>% bind_rows()
    mv_id <- 1:q %x% matrix(1, length(xx)^2)
    
    ai1 <- rep(1, q)
    ai2 <- rep(1, q)
    phi_i <- rep(1, q)
    thetamv <- rep(1.5, ifelse(q>2, 3, 1))
    Dmat <- matrix(.02, q, q)
    diag(Dmat) <- 0
    
    cell_size <- 70
    K <- rep(2, 2)
    num_threads <- 4
    verbose <- F
    debug <- F
    
  }
  
  # cell_size = (approximate) number of location for each block
  # K = number of blocks at resolution L is K^(L-1), with L=1, ... , M.
  cat("Sampling a spatial multiresolution tree.\n")
  
  dd <- ncol(coords)
  na_which <- rep(1, nrow(coords))
  axis_cell_size <- rep(cell_size^(1/dd), dd)
  
  cix <- coords %>% as.data.frame() %>% mutate(ix = 1:n())
  
  
  cat("Partitioning into resolution layers.\n")
  #ptime <- system.time(
  coords_ix <- coords %>% mutate(ix = 1:n())
  #coords <- coords_ix
  mgp_tree <- spamtree:::make_tree(coords_ix, na_which, mv_id, 
                                   axis_cell_size, K, 
                                   max_res <- 3, last_not_ref <- F)
  #)
  #cat("Partitioning total time: ", as.numeric(ptime["elapsed"]), "\n")
  
  parchi_map  <- mgp_tree$parchi_map
  coords_blocking  <- mgp_tree$coords_blocking %>% arrange(ix)
  thresholds <- mgp_tree$thresholds
  res_is_ref <- mgp_tree$res_is_ref
  
  #coords_blocking %>% filter(sort_mv_id==1) %>% 
   # ggplot(aes(Var1, Var2, label=res, color=factor(res))) + 
   # geom_text() + theme(legend.position="none")
  
  coordsnames <- paste0("Var", 1:dd)
  #simdata %<>% arrange(!!!syms(coordsnames), ix)
  
  #coords_blocking %<>% arrange(!!!syms(coordsnames), ix)
  
  #mv_group_ix <- coords_blocking %>% 
  #  group_by(!!!syms(paste0("Var", 1:dd))) %>% 
  #  group_indices()
  
  #coords_blocking %<>%  mutate(gix = mv_group_ix) %>%
  #  group_by(block) #%>% 
    #mutate(gix_block = as.numeric(factor(gix))) %>% as.data.frame()
  coords_blocking %<>% arrange(res, block, ix)
  
  cx_all <- coords_blocking %>% ungroup() %>% dplyr::select(!!!syms(coordsnames)) %>% as.matrix()
  mv_id <- coords_blocking$sort_mv_id
  blocking <- coords_blocking$block
  
  block_info <- coords_blocking %>% mutate(color = res) %>%
    select(block, color) %>% unique()
  
  ###### building graph
  #block_ct_obs_df <- simdata %>% dplyr::select(ix, na_which) %>% 
  #  left_join(coords_blocking %>% dplyr::select(ix, block), by=c("ix"="ix")) %>%
  #  group_by(block) %>% summarise(perc_avail = sum(na_which, na.rm=T)/n(), .groups="drop")
  non_empty_blocks <- blocking %>% unique() #block_ct_obs_df[block_ct_obs_df$perc_avail>0, "block"] %>% pull(block)
  

  cat("Building graph.\n")
  limited_tree <- T
  if(limited_tree){
    parents_children <- spamtree:::make_edges_limited(parchi_map %>% as.matrix(), non_empty_blocks, res_is_ref)
  } else {
    parents_children <- spamtree:::make_edges(parchi_map %>% as.matrix(), non_empty_blocks, res_is_ref)
  }
  
  parents                      <- parents_children[["parents"]] 
  children                     <- parents_children[["children"]] 
  block_names                  <- block_info$block
  block_groups                 <- block_info$color[order(block_names)]
  indexing                     <- (1:nrow(coords_blocking)-1) %>% split(blocking)
  
  cat("Sampling.\n")
  sampled <- spamtree:::spamtree_sample(cx_all, mv_id, blocking,
                            parents, block_names, indexing, 
                            ai1, ai2, phi_i, thetamv, Dmat,
                            num_threads,
                            verbose, debug)
  
  if(0){
  Ciout <- spamtree:::spamtree_Cinv(cx_all, mv_id, blocking,
                                        parents, block_names, indexing, 
                                        ai1, ai2, phi_i, thetamv, Dmat,
                                        num_threads,
                                        verbose, debug)
  image(Ciout$Ci)
  LC <- Ciout$Ci %>% solve() %>% chol() %>% t()
  sampled <- ( LC %*% rnorm(nrow(LC)) ) %>% as.numeric()
  
  sampled_df <- cx_all %>% cbind(mv_id, sampled) %>% 
    as.data.frame()
  
  (plotted <- sampled_df %>% 
    ggplot(aes(Var1, Var2, fill=sampled)) + geom_raster() + facet_wrap(~mv_id, ncol=round(q/2)) +
    scale_fill_viridis_c())
  }
  return(sampled[order(coords_blocking$ix)])
}

