axis_parallel <- function(coordsmat, thresholds, n_threads){
  tessellate_time <- system.time({
  
  cx <- coordsmat %>% 
    as.data.frame() %>% 
    dplyr::select(-dplyr::contains("ix")) %>% 
    as.matrix()
  ncol_coords <- ncol(cx)#-1
  num_blocks_by_coord <- part_axis_parallel_lmt(cx, thresholds)
  blocks_by_coord <- num_blocks_by_coord %>% apply(2, factor)
  dim(blocks_by_coord) <- dim(num_blocks_by_coord)
  colnames(blocks_by_coord) <- colnames(num_blocks_by_coord) <- paste0("L", 1:ncol_coords)
  
  
  if(dim(blocks_by_coord)[1] == 1){
    block <- 1
  } else {
    block <- as.data.frame(blocks_by_coord) %>% 
      as.list() %>% 
      interaction() %>%
      factor() %>% 
      as.numeric()
  }
  
  #partblock <- num_blocks_by_coord %>% col_to_string() %>% #
    ##apply(2, function(x) as.character((x-1) %% 2)) %>%
  #  as.data.frame() %>% as.list() %>% interaction() %>% as.numeric()
  
  #blockdf <- data.frame(num_blocks_by_coord, 
  #                      block = block) 
  
  blockdf <- num_blocks_by_coord %>% cbind(block) %>% as.data.frame()
  
  result <- cbind(coordsmat, blockdf)
  
  })
  
  
  #cat("Partitioning run in ", as.numeric(tessellate_time["elapsed"]), "\n")
  return(result)
}

