#
#  IFL: Iterative Fused LASSO Algorithm
#
#  Last update: 19/07/25
#

library(glmnet)     # LASSO and Elastic-Net 

# Supporting Function
source("supportingfunctions.R")

#
# IFL
#

# IFL(y, x, handle_outlier = true, intercept = true)
# 
# Intercept = true => There is an intercept in x, then once the data is centralized, 
#   it can be removed from the first column of X (zeros) 
# 
# Returns beta_np_1 = v_beta_Par , beta_n_p = m_beta_Par,           # Solutions without Outliers
#         betao_np_1 = v_beta_Paro , betao_n_p = m_beta_Paro,       # Solutions with Outliers
#         beta0 = beta0_hat,                                        # Estimated Intercept
#         y_hat =  y_original_hat ,                                 # Insample Estimators without Outliers
#         y_hato = y_original_hat_woutl,                            # Insample Estimators with Outliers
#         mean_x  = mean_x , mean_y  = mean_y,                      # column-wise mean of X and mean of y
#         bic_value = bic_value,                                    # BIC value model without outliers
#         bic_value_outliers = bic_value_outliers,                  # BIC value model with outliers
#         chosen_var = chosen_variables,                            # List of selected columns in X
#         chosen_var_bk = chosen_variables_break,                   # List of selected columns with estim breaks in X
#         chosen_var_outliers = chosen_variables_outliers,          # List of selected columns in X (model w/outliers)
#         chosen_var_outliers_bk = chosen_variables_outliers_break, # List of selected columns with estim breaks in X (model w/outliers)
#         RMSE_inS = RMSE_inS, RMSE_inSo = RMSE_inSo,               # In Sample RMSE (models with and without outliers)
#         ind_out = ind_out,                                        # list of outlier in y  
#         IFL_t0 = IFL_t0                                           # time to converge
# 
# Last review: 19 Jul 25
#

IFL <- function(y, x, handle_outlier = TRUE, intercept = TRUE) {

  
  # TESTE
  #handle_outlier <- TRUE
  #intercept <- TRUE
  # TESTE
  
  # Store originals
  x_original <- x
  y_original <- y
  
  # Always center the data
  mean_x <- colMeans(x_original) 
  mean_y <- mean(y_original)
  beta0 = mean_y
  y <- y_original - mean_y
  x <- scale(x_original, center = TRUE, scale = FALSE)
  
  # No need to process intercept column that equals 0!
  if (intercept) {
    x <- x[, -1]
  }
  
  ind_out <- numeric()
  
  ########################################################
  # Problem Dimensions
  n <- length(y) 
  p <- dim(x)[2]
  ########################################################
  
  # Create Matrix XX 
  XX <- NULL
  for (i in 1:p) {
    #i <- 1
    XX <- cbind(XX, diag(x[,i]))
  }
  
  # Setup Iteractive Fused LASSO 
  nbetas_In  <- c(rep(n,p))
  nbetas_In_Tot <- sum(nbetas_In)
  nbetas_Out <- c(rep(0,p))
  nbetas_Out_Tot <- sum(nbetas_Out)
  
  W <- diag(n*p)
  
  # Main Loop
  IFL_t0 <- system.time({
    Ld <- L_Matrix(1, nbetas_In)
    Ld_i <- inverse_L(nbetas_In)
    
    # Update model
    H <- XX %*% W %*% Ld_i 
    
    # Ada LASSO solution with smallest BIC
    # Intercept is managed outside glmnet, therefore intercept here is FALSE
    theta_hat_Est <- Ada_LASSO_intercept(H, y,FALSE, FALSE)
    theta_hat <- theta_hat_Est$coef
    
    # VERIFICAR !!!!!
    theta_hat_Intercept <- theta_hat_Est$parameters[1]
    theta_hat_itcp <- theta_hat_Est$itcp
    # VERIFICAR !!!!!
    
    # Gamma_d
    Gamma_d_TF <- abs(theta_hat) > 0
    Gamma_d <- matrix(as.integer(Gamma_d_TF) ,
                      nrow = nbetas_In_Tot, ncol = 1, byrow = FALSE)
    Gamma_d_Acc <- apply(Gamma_d, 2, cumsum)
    
    # Checking for 1s in free betas
    aux1 <- 0  
    nbetas_In_Acc <- cumsum(nbetas_In)
    for (i in 1:p) {
      aux2 <- aux1 + min(1,nbetas_In[i]) 
      Gamma_d[(aux1+1):aux2] <- 1
      aux1 <- nbetas_In_Acc[i]
    }
    
    # Calculating qty of diff betas in the next iteration
    nbetas_Out <- c(rep(0,p))
    nbetas_Out[1] <- sum(Gamma_d[1:nbetas_In_Acc[1]])
    for (j in 2:p) {
      aux1 <- nbetas_In_Acc[j-1] + 1
      aux2 <- nbetas_In_Acc[j]
      nbetas_Out[j] = sum(Gamma_d[aux1:aux2])
    }
    nbetas_Out_Tot <- sum(nbetas_Out)
    
    # Updating Matrix M
    if(nbetas_In_Tot != nbetas_Out_Tot){  
      g_d <- cumsum(Gamma_d) 
      M <- matrix(0, nrow = nbetas_In_Tot, ncol = nbetas_Out_Tot, byrow = TRUE)
      for (i in 1:nbetas_In_Tot){
        if(Gamma_d[i] != 0){
          M[i, g_d[i]] <- 1
        } else{
          M[i, g_d[i-1]] <- 1
        }
      }
      
      # Update W
      W <- W %*% M
      nbetas_In <- nbetas_Out
      nbetas_In_Tot <- sum(nbetas_In)
      LdP <- L_Matrix(1, nbetas_In)
      LdP_i <- inverse_L(nbetas_In)
      
      # Partial solution: solve problem on new dimensions 
      HP <- XX %*% W %*% LdP_i 
      
      if (handle_outlier){
        # Handling Outliers
        HPo <- cbind(HP, diag(1, n, n))
        
        # Ada LASSO solution with smallest BIC
        theta_hatP_Est <- Ada_LASSO_intercept_outlier(HP, y, HPo, FALSE, FALSE)
        
        # Solution without outliers
        theta_hatP <- theta_hatP_Est$coef
        theta_hatp_Intercept <- theta_hatP_Est$parameters[1]
        # BIC Value
        bic_value <- theta_hatP_Est$bic[theta_hatP_Est$bick]
        
        # Solution with outliers
        theta_hatPo <- theta_hatP_Est$coefo
        theta_hatp_Intercepto <- theta_hatP_Est$itcpo
        # BIC Value
        bic_value_outliers <- theta_hatP_Est$bico[theta_hatP_Est$bicko]
        
        y_hato <- HPo %*% theta_hatPo # With Intercept
        y_hat  <- HP  %*% theta_hatP  # Without Intercept
        
        # Number of Outliers
        ind_out <- which(tail(theta_hatPo, n) > 0)
      
      } else{
        # Ada LASSO solution with smallest BIC
        theta_hatP_Est <- Ada_LASSO_intercept(HP, y, FALSE, FALSE)
        theta_hatP <- theta_hatP_Est$coef
        theta_hatp_Intercept <- theta_hatP_Est$parameters[1]
        # BIC Value
        bic_value <- theta_hatP_Est$bic[theta_hatP_Est$bick]
        bic_value_outliers <- NULL
        
        y_hato <- rep(0, n)          # With Intercept
        y_hat  <- HP  * theta_hatP  # Without Intercept
        
        #v_beta_Par <- W %*% LdP_i %*% theta_hatP
        #m_beta_Par <- matrix(v_beta_Par, nrow = n, ncol = p, byrow = FALSE)}
        
      } # Handling outliers
    } # Updating M
  }) # Time
  
  # beta estimated without outliers
  v_beta_Par <- W %*% LdP_i %*% theta_hatP # Without the intercept
  m_beta_Par <- matrix(v_beta_Par, nrow = n, ncol = p, byrow = FALSE)
  
  # beta estimated with outliers
  if (handle_outlier){
    v_beta_Paro <- W %*% LdP_i %*% theta_hatPo[1:(length(theta_hatPo)-n)]
    # Without the intercept
    m_beta_Paro <- matrix(v_beta_Paro, nrow = n, ncol = p, byrow = FALSE)
  } else{
    v_beta_Paro <- v_beta_Par
    m_beta_Paro <- m_beta_Par
  }
  
  # Estimate the intercept
  beta0_hat <- mean(y_original) - colMeans(XX) %*% v_beta_Par 
  
  # Always consider the estimated intercept. Since data was centralized, it needs to be decentralized back.
  y_original_hat <- as.numeric(beta0_hat) + y_hat 
  
  if(handle_outlier) {
    y_original_hat_woutl  <- as.numeric(beta0_hat) + y_hato
  } else{
    y_original_hat_woutl  <- y_original_hat_woutl
  }
  
  # RMSE In Sample
  RMSE_inS  <- sqrt(1/length(y)*sum((y-y_hat)^2))
  RMSE_inSo <- sqrt(1/length(y)*sum((y-y_hato)^2))
  
  # Selected variables X unselected variables, and variables with break 
  chosen_variables <- rep(0,p)
  chosen_variables_break <- rep(0,p)
  chosen_variables_outliers <- rep(0,p)
  chosen_variables_outliers_break <-  rep(0,p)
  
  for (j in 1:p) {
    if (abs(sum(m_beta_Par[,j])) > 0) {
      chosen_variables[j] <- 1
      featbk <- FALSE
      betaref <- m_beta_Par[1,j] # first value of m_beta_Par[:,j] 
      for (k in 1:n) {
        if(m_beta_Par[k,j] != betaref) { # At least one break
          featbk <- TRUE
        }
      }
      if(featbk) {
        chosen_variables_break[j] <- 1
      }
    }
    
    if (abs(sum(m_beta_Paro[,j])) > 0) {
      chosen_variables_outliers[j] <- 1
      featbk <- FALSE
      betaref <- m_beta_Paro[1,j] # first value of m_beta_Par[:,j] 
      for (k in 1:n) {
        if(m_beta_Paro[k,j] != betaref) { # At least one break
          featbk <- TRUE
        }
      }
      if(featbk) {
        chosen_variables_outliers_break[j] <- 1
      }
    }
  }
  
  list( beta_np_1 = v_beta_Par , beta_n_p = m_beta_Par, 
        betao_np_1 = v_beta_Paro , betao_n_p = m_beta_Paro,
        beta0 = beta0_hat,
        y_hat =  y_original_hat,
        y_hato = y_original_hat_woutl,
        mean_x  = mean_x , mean_y  = mean_y,
        bic_value = bic_value,
        bic_value_outliers = bic_value_outliers,
        chosen_var = chosen_variables,
        chosen_var_bk = chosen_variables_break,
        chosen_var_outliers = chosen_variables_outliers,
        chosen_var_outliers_bk = chosen_variables_outliers_break,
        RMSE_inS = RMSE_inS, RMSE_inSo = RMSE_inSo,
        ind_out = ind_out,
        IFL_t0 = IFL_t0  )
  

}
  
  