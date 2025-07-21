#
#  Supporting Functions
#    for the Interactive algorithm
#
#  AM 05/07/24 
#   v.20/07/25
#

library(glmnet)     # LASSO and Elastic-Net 

################################################################################
# Supporting Functions
################################################################################

#
# Ldk Matrix
#

Ldk <- function(d, k) {
  L_d_k <- NULL
  if(k > d) {
    # Supporting matrices
    Z_d_kmd <- matrix(0, nrow = d, ncol=(k-d))
    Z_kmd_d <- matrix(0, nrow = (k-d), ncol=d)
    Z_dd    <- matrix(0, nrow = d, ncol=d)
    I_kmd   <- diag(k-d)
    L_d_k <- diag(k) - rbind(cbind(Z_d_kmd, Z_dd), cbind(I_kmd, Z_kmd_d))  
  } else{
    L_d_k <- diag(k)
  }
  return(L_d_k)
}

#
# L_Matrix
#

L_Matrix <- function(d, n_betas) {
  
  Ld <- NULL
  tam <- length(n_betas)
  list_Ls <- list()
  list_dims <- list()
  for (i in 1 : tam){
    #cat("Ldk: d: ",i, " k: ", n_betas[i],  "\n")
    Ld_i <- Ldk(d, n_betas[i])
    list_Ls[[length(list_Ls) + 1]] <- Ld_i
    if(!is.null(dim(Ld_i))){
      list_dims[[length(list_dims) + 1]] <- dim(Ld_i)
    } else {
      list_dims[[length(list_dims) + 1]] <- c(1,1)
    }
  }
  
  for (i in 1: tam){
    L_row <- NULL
    L_left <- NULL
    if (i > 1) {
      for (left in 1: (i-1)){
        ml <- matrix(0, nrow = list_dims[[i]][1], ncol = list_dims[[left]][2])
        L_left <- cbind(L_left, ml)
      }
    }
    L_right <- NULL
    if (i < tam) {
      for (right in (i+1):tam){
        mr <- matrix(0, nrow = list_dims[[i]][1], ncol = list_dims[[right]][2])
        L_right <- cbind(L_right, mr)
      }
    }
    L_row <- cbind(L_left, list_Ls[[i]])
    L_row <- cbind(L_row, L_right)
    Ld <- rbind(Ld, L_row)
  }
  return(Ld)
}

#
# Sparse Lkd
#

Ldk_Sp <- function(d, k) {
  L_d_k <- NULL
  if(k > d) {
    # Supporting matrices
    Z_d_kmd <- Matrix(0,     d, (k-d))
    Z_kmd_d <- Matrix(0, (k-d),     d)
    Z_dd    <- Matrix(0,     d,     d)
    I_kmd   <- Diagonal(x = 1, k-d)
    L_d_k   <- Diagonal(x=1, k) - rbind(cbind(Z_d_kmd, Z_dd), cbind(I_kmd, Z_kmd_d))  
  } else{
    L_d_k <- Diagonal(x=1, k)
  }
  return(L_d_k)
}

#
# Sparse L_Matrix
#

L_Matrix_Sp <- function(d, n_betas) {
  Ld <- NULL
  tam <- length(n_betas)
  list_Ls <- list()
  list_dims <- list()
  for (i in 1 : tam){
    #cat("Ldk: d: ",i, " k: ", n_betas[i],  "\n")
    Ld_i <- Ldk_Sp(d, n_betas[i])
    list_Ls[[length(list_Ls) + 1]] <- Ld_i
    if(!is.null(dim(Ld_i))){
      list_dims[[length(list_dims) + 1]] <- dim(Ld_i)
    } else {
      list_dims[[length(list_dims) + 1]] <- c(1,1)
    }
  }
  
  for (i in 1: tam){
    L_row <- NULL
    L_left <- NULL
    if (i > 1) {
      for (left in 1: (i-1)){
        ml <- Matrix(0, list_dims[[i]][1], list_dims[[left]][2])
        L_left <- cbind(L_left, ml)
      }
    }
    L_right <- NULL
    if (i < tam) {
      for (right in (i+1):tam){
        mr <- Matrix(0, list_dims[[i]][1], list_dims[[right]][2])
        L_right <- cbind(L_right, mr)
      }
    }
    L_row <- cbind(L_left, list_Ls[[i]])
    L_row <- cbind(L_row, L_right)
    Ld <- rbind(Ld, L_row)
  }
  return(Ld)
}

#
# Ada_LASSO
#

Ada_LASSO <- function(XX, W, n_y, d, nbetas, inverte) {  
  
  #cat("nbetas: ", nbetas ,"\n")
  #cat("d: ",           d ,"\n")
  Ld_i <- NULL
  if (inverte){
    Ld <- L_Matrix(d, nbetas_In)
    Ld_i <- solve(Ld)
  }
  inverte <- FALSE
  
  # Update model
  H <- XX %*% W %*% Ld_i 
  
  # First step: initial Ridge solution
  bic.ridge <- bic.glmnet(H, n_y, alpha=0, standardize=FALSE, intercept = FALSE)
  
  w <- 1/abs(matrix(bic.ridge$bicbeta))^1 ## Using gamma = 1
  
  #class(w)
  #for (i in 1 :length(w) ){
  #  cat(i, ": ", w[i], "\n")
  #}
  
  w[w[,1] == Inf] <- 1e98 ## Replacing values estimated as infinite with 999999999
  
  #w[1] <- 1e8
  #for (i in 2:length(nbetas)){
  #  w[(nbetas[i-1]+1)] <- 1e8
  #  }
  
  #cat("Revisado \n")
  #for (i in 1 :length(w) ){
  #  cat(i, ": ", w[i], "\n")
  #}
  
  bic.lasso1 <- bic.glmnet(H, n_y, standardize=FALSE, intercept=FALSE, penalty.factor=w)
  
  # coef1 is theta
  coef1 <- matrix(bic.lasso1$bicbeta)
  
  list(coef = coef1, inverte = inverte, Ld_i = Ld_i)
}

# Ada_LASSO Intercept without outliers

Ada_LASSO_intercept <- function(H, y, standardize = FALSE, intercept = FALSE) {
  
  # First step: initial Ridge solution
  bic.ridge <- bic.glmnet(H, y, alpha=0, standardize= standardize, intercept=intercept)
  
  w <- 1/abs(matrix(bic.ridge$bicbeta))^1 ## Using gamma = 1
  
  w[w[,1] == Inf] <- 1e4 # Replacing values estimated as infinite with 10_000
  
  # AdaLASSO solution => alpha = 1
  bic.lasso1 <- bic.glmnet(H, y, 1, standardize=FALSE, intercept=intercept, penalty.factor=w)
  
  # coef1 is theta
  coef1 <- matrix(bic.lasso1$bicbeta)
  itcp <- coef1[1, ]
  
  list(coef = coef1, parameters = bic.lasso1$parameters, 
       betas = bic.lasso1$betas, df = bic.lasso1$df,
       lambda = bic.lasso1$lambda, 
       bick = bic.lasso1$bick, bic = bic.lasso1$bic,
       biclambda = bic.lasso1$biclambda,
       itcp = itcp)
}


# Ada_LASSO Intercept with outliers

Ada_LASSO_intercept_outlier <- function(H, y, x2, standardize = FALSE, intercept = FALSE) {
  
  # TESTE
  #standardize <- FALSE
  #intercept <- FALSE
  #H <- HP
  #x2 <- HPo
  # TESTE
  
  # Handling Outliers
  # First step: initial Ridge solution, alpha = 0
  bic.ridge <- bic.glmnet(H, y, alpha=0, standardize= standardize, intercept=intercept)
  
  # Ridge regression gamma
  gamma <- 1
  
  w <- 1/abs(matrix(bic.ridge$bicbeta))^gamma ## Using gamma = 1
  w[w[,1] == Inf] <- 1e6 # Replacing values estimated as infinite with 1_000_000
  
  # Find Outlier Weights
  res <- y - H %*% bic.ridge$bicbeta
  wd <- 1 /abs(res)^gamma
  wd[wd[,1] == Inf] <- 1e6 # Replacing values estimated as infinite with 1_000_000
  
  w_extend <- rbind(w, wd)
  
  # AdaLASSO solution => alpha = 1
  bic.adalasso          <- bic.glmnet(H,  y, 1, standardize=FALSE, intercept=intercept, penalty.factor=w)
  bic.adalasso_outliers <- bic.glmnet(x2, y, 1, standardize=FALSE, intercept=intercept, penalty.factor=w_extend)
  
  # coef1 is theta
  coef1 <- matrix(bic.adalasso$bicbeta)
  itcp <- coef1[1, ]
  coef1o <- matrix(bic.adalasso_outliers$bicbeta)
  itcpo <- coef1o[1, ]
  
  list(coef = coef1, parameters = bic.adalasso$parameters, 
       betas = bic.adalasso$betas, df = bic.adalasso$df,
       lambda = bic.adalasso$lambda, 
       bick = bic.adalasso$bick, bic = bic.adalasso$bic,
       bicko = bic.adalasso_outliers$bick, bico = bic.adalasso_outliers$bic,
       biclambda = bic.adalasso$biclambda,
       itcp = itcp,
       coefo = coef1o, itcpo = itcpo )
}

#
# Inverse L Matrix
#

inverse_L <- function(nbetas_In) {

  Ldi_2 <- NULL
  tam <- length(nbetas_In)
  for (i in 1:tam){
    Block_dim <- nbetas_In[i]
    Block_i <-  matrix(0, Block_dim, Block_dim)
    Block_i[lower.tri(Block_i, diag = TRUE)] <- 1
    # Block_row
    B_row <- NULL
    B_left <- NULL
    if (i > 1) {
      for (left in 1: (i-1)){
        ml <- matrix(0, nrow = Block_dim, ncol = nbetas_In[left])
        B_left <- cbind(B_left, ml)
      }
    }
    B_right <- NULL
    if (i < tam) {
      for (right in (i+1):tam){
        mr <- matrix(0, nrow = Block_dim, ncol = nbetas_In[right])
        B_right <- cbind(B_right, mr)
      }
    }
    B_row <- cbind(B_left, Block_i)
    B_row <- cbind(B_row, B_right)
    Ldi_2 <- rbind(Ldi_2, B_row)
  }
  return(Ldi_2)
}

#
# bic.glmnet
#

bic.glmnet <- function(x, y, alpha=1, standardize=FALSE, intercept=FALSE, penalty.factor=NULL) {
  
  if (is.null(penalty.factor)) {
    lasso1 <- glmnet(x, y, standardize=standardize, intercept=intercept, alpha=alpha)
  } else {
    #lasso1 <- glmnet(x, y, , standardize=standardize, intercept=intercept, alpha=alpha, penalty.factor=penalty.factor)
    lasso1 <- glmnet(x, y, standardize=standardize, intercept=intercept, alpha=alpha, penalty.factor=penalty.factor)
  }
  k <- lasso1$dim[2]
  p <- lasso1$dim[1]
  n <- lasso1$nobs
  
  logLike <- rep(NA, times=k)
  bic     <- rep(NA, times=k)
  for (i in 1:k) {
    # usando a relação "sum((y - XB)^2) == lasso$nulldev*(1-lasso$dev.ratio[i])"
    # sigma2 = Deviance/n
    sig2 <- lasso1$nulldev*(1-lasso1$dev.ratio[i])/n
    logLike[i] <- -n/2*log(2*pi) - n/2*log(sig2) - n/2
    bic[i] <- lasso1$df[i]*log(n) -2*logLike[i]
  }
  
  # Escolhendo o valor com menor bic
  idx <- which.min(bic)
  bicbeta <- lasso1$beta[,idx]
  biclambda <- lasso1$lambda[idx]
  bick <- lasso1$df[idx]
  
  #
  # Outras formas de calcular o BIC
  #
  bicV2    <- rep(NA, times=k)
  for (i in 1:k){
    sigma2 <- 0
    sigma2 <- 1/n * sum((y - x %*% lasso1$beta[,i])^2)
    bicV2[i] <- n * log(sigma2) + log(n)* lasso1$df[i]
  } 
  
  # Escolhendo a solução que minimiza o BIC
  idx2 <- which.min(bicV2)
  bicbeta2 <- lasso1$beta[,idx2]
  biclambda2 <- lasso1$lambda[idx2]
  bick2 <- lasso1$df[idx2]
  
  # Artigo Camila Epprecht
  bicV3    <- rep(NA, times=k)
  for (i in 1:k){
    sigma2 <- 0
    sigma2 <- 1/n * sum((y - x %*% lasso1$beta[,i])^2)
    bicV3[i] <- log(sigma2) + 1/n * log(n)* lasso1$df[i]
  } 
  
  # Escolhendo a solução com o menor BIC
  idx3 <- which.min(bicV3)
  bicbeta3 <- lasso1$beta[,idx3]
  biclambda3 <- lasso1$lambda[idx3]
  bick3 <- lasso1$df[idx3]
  
  # Considering the Intercept
  parameters <- coef(lasso1, s = biclambda3)  
  
  ##############################
  
  # Original
  #list(logLike = logLike, bic = bic, betas = lasso1$beta, lambda = lasso1$lambda, 
  #     df = lasso1$df, biclambda = biclambda, bicbeta = bicbeta, bick = bick,
  #     parameters = parameters)
  
  # BIC V2
  #list(logLike = logLike, bic = bicV2, betas = lasso1$beta, lambda = lasso1$lambda, 
  #     df = lasso1$df, biclambda = biclambda2, bicbeta = bicbeta2, bick = bick2,
  #      parameters = parameters)
  
  # BIC V3
  list(logLike = logLike, bic = bicV3, betas = lasso1$beta, lambda = lasso1$lambda, 
       df = lasso1$df, biclambda = biclambda3, bicbeta = bicbeta3, bick = bick3,
       parameters = parameters)
  
  # BIC TODOS
  #list(logLike = logLike, bic = bic, betas = lasso1$beta, lambda = lasso1$lambda,
  #     df = lasso1$df, biclambda = biclambda, bicbeta = bicbeta, bick = bick,
  #     bicV2 = bicV2, biclambdaV2 = biclambda2, bicbetaV2 = bicbeta2, bickV2 = bick2,
  #     bicV3 = bicV3, biclambdaV3 = biclambda3, bicbetaV3 = bicbeta3, bickV3 = bick3  )
  
}




#
# Choose best BIC (used with genLASSO)
#

beta_BIC <- function(gSol, y, x){
  k <- length(gSol$lambda)
  bic  <- rep(NA, times=k)
  n <- length (y)
  for (i in 1:k){
    sigma2 <- 0
    sigma2 <- 1/n * sum((y - x %*% gSol$beta[,i])^2)
    #bic[i] <- n * log(sigma2) + log(n)* gSol$df[i]
    #
    # Versão Camila Epreccht
    #
    bic[i] <- log(sigma2) + 1/n * gSol$df[i] * log(n)
    
  } 
  
  # Choosing smaller value of BIC
  idx <- which.min(bic)
  bicbeta <- gSol$beta[,idx]
  biclambda <- gSol$lambda[idx]
  bick <- gSol$df[idx]
  
  list(bic = bic, bicbeta = bicbeta, biclambda = biclambda,  bick = bick, 
       lambda = gSol$lambda, df = gSol$df)
  
}
