axis_parallel <- function(coordsmat, thresholds, n_threads){
  
  tessellate_time <- system.time({
    ncol_coords <- ncol(coordsmat)-1
  num_blocks_by_coord <- spamtree:::part_axis_parallel_lmt(coordsmat[,-ncol(coordsmat)], thresholds)
  blocks_by_coord <- num_blocks_by_coord %>% apply(2, factor)
  colnames(blocks_by_coord) <- colnames(num_blocks_by_coord) <- paste0("L", 1:ncol_coords)
  
  
  block <- as.data.frame(blocks_by_coord) %>% 
    as.list() %>% 
    interaction() %>%
    factor() %>% 
    as.numeric()
  
  #partblock <- num_blocks_by_coord %>% col_to_string() %>% #
    ##apply(2, function(x) as.character((x-1) %% 2)) %>%
  #  as.data.frame() %>% as.list() %>% interaction() %>% as.numeric()
  
  blockdf <- data.frame(num_blocks_by_coord, 
                        block = block#,
                        #part = partblock
                        )
  
  result <- cbind(coordsmat, blockdf)
  })
  
  
  #cat("Partitioning run in ", as.numeric(tessellate_time["elapsed"]), "\n")
  return(result)
}
