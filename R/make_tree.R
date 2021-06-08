make_tree <- function(coords, na_which, sort_mv_id, 
                      axis_cell_size = c(5,5), K = c(2,2), 
                      start_level = 0, tree_depth = Inf,
                      last_not_reference = TRUE,
                      cherrypick_same_margin = TRUE,
                      cherrypick_group_locations = TRUE,
                      mvbias = 0, verbose=T){
  # mvbias: 0 = treat all multivariate margins the same
  # >0 = prefer picking sparser margins for lower levels of the tree
  
  max_res <- start_level + tree_depth 
  
  mv_id_weights <- table(na_which, sort_mv_id) %>% `[`(1,) %>% 
    magrittr::raise_to_power(-mvbias) 
  sum_mv_id_weights <- sum(mv_id_weights)
  
  mv_id_weights %<>% 
    magrittr::divide_by(sum_mv_id_weights)
  
  mv_id_weights <- data.frame(mv_id_weight=mv_id_weights) %>% 
    tibble::rownames_to_column("sort_mv_id") %>% 
    dplyr::mutate(sort_mv_id=as.numeric(sort_mv_id))
  
  coords_allres <- coords %>% as.data.frame() %>% cbind(sort_mv_id)
  coords_missing <- coords_allres[is.na(na_which),] 
  coords_availab <- coords_allres[!is.na(na_which),] %>% as.data.frame()
  
  nr <- nrow(coords_availab)
  dd <- ncol(coords_availab) - 2
  
  joining_coords <- paste0("Var", 1:dd)
  names(joining_coords) <- joining_coords
  
  minx <- coords_availab[,1:dd] %>% min()
  maxx <- coords_availab[,1:dd] %>% max()
  
  cx <- coords_availab
  colmvid <- dd+2#which(grepl("mv_id", colnames(cx)))
  
  coords_refset <- cx %>% 
    dplyr::mutate(block=NA, res=NA) %>% #, part=NA) %>%
    dplyr::filter(.data$Var1 < minx)
  
  cell_size <- prod(axis_cell_size)
  n_u_cells <- nr/cell_size
  
  #LL <- floor( log(n_u_cells)/log(4)  )
  thresholds_list <- list()
  max_block_number <- 0
  res <- start_level+1
  
  timings <- rep(0, 3)
  
  if(verbose){
    cat("Branching the tree")
  }
  
  
  res_ix <- 1
  
  first_steps <- system.time({
    while((res <= max_res) & (nrow(cx) > 0)){
      
      if(verbose){
        cat(" ", res, "(",res_ix,")")
      }
      
      # first, tessellate to find locations that are spread out. select one for each cell.
      # second, with the resulting subsample then actually split into blocks
      
      # generate threshold to make small cells from which we will sample 1 location
      # fixed threshold regularly on each axis
      #thresholds_knots <- 1:dd %>% lapply(function(i) seq(minx-1e-6, maxx+1e-6, length.out=axis_cell_size[i] * 2^(res-1) + 1))
      # based on quantiles -- hopefully more uniform partitioning
      
      timings[1] <- timings[1] + 
        system.time({
          
          thresholds_knots <- 1:dd %>% lapply(function(i) 
            kthresholds(coords_availab[,i], axis_cell_size[i] * K[i]^(res-1)))
          
          grid_size <- thresholds_knots %>% lapply(function(x) length(x)+1) %>% unlist() %>% prod()
          
          if(grid_size < nrow(cx)){
            # if grid is too thin considering available coordinates, skip this and partition directly
            coords_knots <- cx[,-colmvid] %>% as.matrix() %>%
              axis_parallel(thresholds_knots, 4) %>%
              cbind(data.frame(sort_mv_id = cx[,colmvid])) %>% 
              dplyr::left_join(mv_id_weights, by=c("sort_mv_id"="sort_mv_id")) %>%
              #mutate(block = max_block_number + block) %>%
              dplyr::group_by(.data$block) %>% 
              dplyr::summarise(ix = .data$ix[sample(1:dplyr::n(), 1, prob=.data$mv_id_weight)], .groups="keep")
            
            if(cherrypick_group_locations){
              coords_knots <- coords_knots %>% 
                as.data.frame() %>% 
                dplyr::left_join(cx, by=c("ix"="ix")) %>% dplyr::select(-.data$ix, -.data$sort_mv_id) %>%
                dplyr::left_join(cx, by=joining_coords) %>% #by=c("Var1"="Var1", "Var2"="Var2")) %>%
                dplyr::select(dplyr::contains("Var"), .data$ix)
            } else {
              coords_knots <- coords_knots %>% as.data.frame() %>% 
                dplyr::left_join(cx, by=c("ix"="ix")) %>% 
                dplyr::select(dplyr::contains("Var"), .data$ix)
            }
            
          } else {
            coords_knots <- cx 
          }
          
        })["elapsed"]
      
      timings[2] <- timings[2] + 
        system.time({
          # now take the subsample and make the partition
          # fixed threshold regularly on each axis
          #thresholds_res <- 1:dd %>% lapply(function(i) seq(minx-1e-6, maxx+1e-6, length.out=2^(res-1) + 1))
          # based on quantiles -- hopefully more uniform partitioning
          thresholds_res <- 1:dd %>% lapply(function(i) kthresholds(coords_availab[,i], K[i]^(res-1)))
          thresholds_list[[res_ix]] <- thresholds_res
          
          coords_res <- coords_knots[,-colmvid] %>% as.matrix() %>% 
            axis_parallel(thresholds_res, 4) %>%
            dplyr::mutate(block = max_block_number + .data$block,
                   res = res) %>%
            dplyr::select(dplyr::contains("Var"), .data$ix, .data$block, .data$res)#, part)
          coords_res %<>% 
            dplyr::left_join(coords_allres %>% dplyr::select("ix", "sort_mv_id"), by=c("ix"="ix"))
          max_block_number <- coords_res %>% 
            dplyr::pull(.data$block) %>% 
            max()
          
        })["elapsed"]
      
      timings[3] <- timings[3] + 
        system.time({
          # move selected locations to the reference set
          cx %<>% dplyr::filter(!(.data$ix %in% coords_res$ix)) #*
          coords_refset %<>% dplyr::bind_rows(coords_res)
          
          # keep track of overall tessellation
          blockname = sprintf("block_res%02d", res)
          coords_keeptrack <- coords_availab %>% 
            dplyr::select(dplyr::contains("Var"), .data$ix) %>% 
            as.matrix() %>%
            axis_parallel(thresholds_res, 4) %>%
            dplyr::rename(!!rlang::sym(blockname) := .data$block) %>%
            dplyr::select(.data$ix, !!rlang::sym(blockname))
          
          coords_availab %<>% dplyr::left_join(coords_keeptrack, by=c("ix"="ix"))
          
        })["elapsed"]
      res <- res + 1
      res_ix <- res_ix + 1
    }
  })
  
  if(verbose){
    cat(".\n")
  }
  
  
  res_is_ref <- rep(1, res_ix-1)
  if(last_not_reference & (res < max_res)){
    res_is_ref[ length(res_is_ref) ] <- 0
  }
  
  if(verbose){
    cat("Finalizing with leaves.\n")
  }
  
  last_steps <- system.time({
    # now that we have recursive domain partitioning, 
    # we create a table with the parent-child relationships
    
    relate <- coords_availab %>% 
      dplyr::select(dplyr::contains("Var"), .data$ix, dplyr::contains("block_res")) %>%
      dplyr::inner_join(coords_refset %>% dplyr::select(.data$ix, .data$res, .data$block),#, part),
                        by=c("ix"="ix"))
    max_block_number <- 0
    res_from <- coords_refset$res %>% min()
    res_to <- coords_refset$res %>% max()
    for(i in res_from:res_to){
      blockname <- sprintf("block_res%02d", i)
      relate %<>% dplyr::mutate(!!rlang::sym(blockname) := !!rlang::sym(blockname) + max_block_number)
      max_block_number <- max(relate %>% dplyr::pull(!!rlang::sym(blockname)))
    }
    
    parchi_original <- relate[,sprintf("block_res%02d", res_from:res_to), drop=F] %>% 
      dplyr::select(dplyr::contains("block_res")) %>% unique()
    translator <- relate %>% 
      dplyr::mutate(block_unnorm = relate %>% 
               apply(1, function(rr) rr[sprintf("block_res%02d", rr["res"])])) %>%
      dplyr::select(.data$block_unnorm, .data$block) %>% 
      unique() 
    max_block_unnorm <- max(translator$block_unnorm)
    
    translator %<>% dplyr::right_join(data.frame(block_unnorm = 1:max_block_unnorm), by=c("block_unnorm"="block_unnorm"))
    translator[is.na(translator)] <- 0
    
    parchi_map <- number_revalue(parchi_original %>% as.matrix(), translator$block_unnorm, translator$block)
    colnames(parchi_map) <- colnames(parchi_original)
    parchi_map[parchi_map == 0] <- NA
    
    parchi_map <- parchi_map %>% as.data.frame()
    if(ncol(parchi_map)>1){
      parchi_map %<>% 
        dplyr::arrange(!!!rlang::syms(colnames(parchi_map)))
    }
    
    max_block_number <- coords_refset %>% dplyr::pull(.data$block) %>% max()
    
    # now manage leftouts
    if(nrow(cx) > 0){
      ## missing locations = predictions: assign them to same block as their nearest-neighbor.
      column_select <- colnames(cx)[grepl("Var", colnames(cx))]
      cx_leftover <- cx[,c(column_select, "ix", "sort_mv_id")]
      max_res <- coords_refset$res %>% max()
      coords_refset_sub <- coords_refset %>% dplyr::filter(.data$res %in% c( max_res ))
      
      # CHERRY PICKING -- 
      # find same-margin nearest neighbor and assign to that block
      leftover_vars <- cx_leftover$sort_mv_id %>% unique()
      cx_leftover %<>% dplyr::mutate(block=NA)
      
      if(!cherrypick_same_margin){
        #for(vv in leftover_vars){
        # row_of_this <- which(cx_leftover$sort_mv_id == vv)
        
        target_coords_same_margin <- coords_refset_sub %>% 
          #  filter(sort_mv_id == vv) %>% 
          dplyr::select(dplyr::contains("Var"), .data$ix, .data$block)
        
        cx_leftover_same_margin <- cx_leftover #%>%
        #filter(sort_mv_id == vv)
        
        nn_of_leftover_same_margin <- FNN::get.knnx(target_coords_same_margin[,1:dd], 
                                                    cx_leftover_same_margin[,1:dd], k=1, algorithm="kd_tree")
        
        block_of_leftover_same_margin <- target_coords_same_margin[nn_of_leftover_same_margin$nn.index, "block"]
        
        cx_leftover[#row_of_this
          , "block"] <- block_of_leftover_same_margin
        #}
        
      } else {
        for(vv in leftover_vars){
          row_of_this <- which(cx_leftover$sort_mv_id == vv)
          
          target_coords_same_margin <- coords_refset_sub %>% 
            dplyr::filter(.data$sort_mv_id == vv) %>% 
            dplyr::select(dplyr::contains("Var"), .data$ix, .data$block)
          
          cx_leftover_same_margin <- cx_leftover %>%
            dplyr::filter(.data$sort_mv_id == vv)
          
          nn_of_leftover_same_margin <- FNN::get.knnx(target_coords_same_margin[,1:dd], 
                                                      cx_leftover_same_margin[,1:dd], k=1, algorithm="kd_tree")
          
          block_of_leftover_same_margin <- target_coords_same_margin[nn_of_leftover_same_margin$nn.index, "block"]
          
          cx_leftover[row_of_this
                      , "block"] <- block_of_leftover_same_margin
        }
        
      }
      
      block_of_leftover <- cx_leftover$block
      
      #target_coords <- coords_refset_sub %>% select(contains("Var"))
      #nn_of_leftover <- FNN::get.knnx(target_coords, cx_leftover, k=1, algorithm="kd_tree")
      #block_of_leftover <- coords_refset_sub[nn_of_leftover$nn.index, "block"]
      res_of_leftover <- max_res + 1
      
      blockname = sprintf("block_res%02d", res_of_leftover)
      parent_blockname = sprintf("block_res%02d", res_of_leftover-1)
      
      coords_res_leftover <- cx %>% 
        as.data.frame() %>%
        dplyr::mutate(parent_block = block_of_leftover,
               block = as.numeric(factor(block_of_leftover)) + max_block_number,
               res = res_of_leftover)
      
      coords_all <- dplyr::bind_rows(coords_refset, coords_res_leftover %>% #mutate(part=NA) %>% 
                                dplyr::select(-.data$parent_block))
      
      # keep track of overall tessellation
      suppressMessages({
        coords_allres %<>% 
          dplyr::left_join(coords_res_leftover %>% 
                             dplyr::select(-.data$res) %>% dplyr::rename(!!rlang::sym(blockname) := .data$block))
      })
      
      parchi_of_leftover <- coords_res_leftover %>% 
        dplyr::rename(!!rlang::sym(parent_blockname) := .data$parent_block, !!rlang::sym(blockname) := .data$block) %>% 
        dplyr::select(dplyr::contains("block_res")) %>% 
        unique() 
      
      suppressMessages({
        parchi_map %<>% dplyr::left_join(parchi_of_leftover)
      })
      
      res_is_ref <- c(res_is_ref, 0)
    } else {
      coords_all <- coords_refset
    }
    
    if(length(res_is_ref) == 1){
      res_is_ref <- 1
    }
    
    # now manage missing
    # if there are leftovers then look at previous level
    max_res <- coords_all$res %>% max()
    res_of_missing <- max_res + 1 
    blockname = sprintf("block_res%02d", res_of_missing)
    
    if(nrow(coords_missing) > 0){
      max_block_number <- coords_all %>% dplyr::pull(.data$block) %>% max()
      ## missing locations = predictions: assign them to same block as their nearest-neighbor.
      column_select <- colnames(cx)[grepl("Var", colnames(coords_missing))]
      cx_missing <- coords_missing[,c(column_select, "ix", "sort_mv_id")]
      coords_refset_sub <- coords_refset %>% dplyr::filter(.data$res %in% c(max(res)))
      
      #target_coords <- coords_refset_sub %>% select(contains("Var"))
      #nn_of_missing <- FNN::get.knnx(target_coords, cx_missing, k=1, algorithm="kd_tree")
      #block_of_missing <- coords_refset_sub[nn_of_missing$nn.index, "block"]
      
      ## CHERRY PICKING
      
      missing_vars <- cx_missing$sort_mv_id %>% unique()
      cx_missing %<>% dplyr::mutate(block=NA, parent_res=NA)
      
      if(!cherrypick_same_margin){
        for(vv in missing_vars){
          #row_of_this <- which(cx_missing$sort_mv_id == vv)
          
          target_coords_same_margin <- coords_refset_sub %>% 
            #filter(sort_mv_id == vv) %>% 
            #dplyr::filter(res == max(res)) %>%
            dplyr::select(dplyr::contains("Var"), .data$ix, .data$block, .data$res)
          
          cx_missing_same_margin <- cx_missing# %>%
          #filter(sort_mv_id == vv)
          
          nn_of_missing_same_margin <- FNN::get.knnx(target_coords_same_margin[,1:dd], 
                                                     cx_missing_same_margin[,1:dd], k=1, algorithm="kd_tree")
          
          block_of_missing_same_margin <- target_coords_same_margin[nn_of_missing_same_margin$nn.index, "block"]
          
          parentres_of_missing_same_margin <- target_coords_same_margin[nn_of_missing_same_margin$nn.index, "res"]
          
          cx_missing[,"block"] <- block_of_missing_same_margin
          cx_missing[,"parent_res"] <- parentres_of_missing_same_margin
          
        }
      } else {
        for(vv in missing_vars){
          row_of_this <- which(cx_missing$sort_mv_id == vv)
          
          target_coords_same_margin <- coords_refset_sub %>% 
            dplyr::filter(.data$sort_mv_id == vv) %>%
            dplyr::select(dplyr::contains("Var"), .data$ix, .data$block, .data$res)
          
          cx_missing_same_margin <- cx_missing %>%
            dplyr::filter(.data$sort_mv_id == vv)
          
          nn_of_missing_same_margin <- FNN::get.knnx(target_coords_same_margin[,1:dd], 
                                                     cx_missing_same_margin[,1:dd], k=1, algorithm="kd_tree")
          
          block_of_missing_same_margin <- target_coords_same_margin[nn_of_missing_same_margin$nn.index, "block"]
          parentres_of_missing_same_margin <- target_coords_same_margin[nn_of_missing_same_margin$nn.index, "res"]
          
          cx_missing[row_of_this, "block"] <- block_of_missing_same_margin
          cx_missing[row_of_this, "parent_res"] <- parentres_of_missing_same_margin
          
        }
      }
      
      block_of_missing <- cx_missing$block
      parentres_of_missing <- cx_missing$parent_res
      
      coords_res_miss <- coords_missing %>% 
        as.data.frame() %>%
        dplyr::mutate(parent_block = block_of_missing,
               block = as.numeric(factor(block_of_missing)) + max_block_number,
               res = res_of_missing,
               parent_res = parentres_of_missing)
      
      coords_all <- dplyr::bind_rows(coords_all, coords_res_miss %>% #mutate(part=NA) %>% 
                                dplyr::select(-.data$parent_block, -.data$parent_res))
      
      # keep track of overall tessellation
      suppressMessages({
        coords_allres %<>% 
          dplyr::left_join(coords_res_miss %>% 
                             dplyr::select(-.data$res, -.data$parent_block, -.data$parent_res) %>% 
                             dplyr::rename(!!rlang::sym(blockname) := .data$block))
      })
      
      for(jres in unique(coords_res_miss$parent_res)){
        parent_blockname = sprintf("block_res%02d", jres)
        
        parchi_of_missing <- coords_res_miss %>% dplyr::filter(.data$parent_res == jres) %>% 
          dplyr::rename(!!rlang::sym(parent_blockname) := .data$parent_block, !!rlang::sym(blockname) := .data$block) %>% 
          dplyr::select(dplyr::contains("block_res")) %>% 
          unique() 
        suppressMessages({
          parchi_map %<>% dplyr::left_join(parchi_of_missing)
        })
      }
      
      res_is_ref <- c(res_is_ref, 0)
    }
  })
  
  return(list(coords_blocking = coords_all,
              parchi_map = parchi_map %>% unique(),
              thresholds = thresholds_list,
              res_is_ref = res_is_ref))
}

