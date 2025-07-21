#
#  Supporting functions 
#
#  AM 22/09/24
#
#  Last Update: 19/07/25
#

#
# Ldk Matrix
#
function Ldk(d::Int, k::Int) 
  #
  L_d_k = nothing
  if(k > d) 
    # Supporting matrices
    Z_d_kmd = zeros(d, k-d)
    Z_kmd_d = zeros(k-d,d)
    Z_dd    = zeros(d,d)
    I_kmd   = I(k-d)
    L_d_k   = I(k) - [Z_d_kmd Z_dd ;I_kmd Z_kmd_d]  
  else
    L_d_k  = I(k)
  end 
  return L_d_k
end

#
# L_Matrix
#
function L_Matrix(d::Int64, nbetas::Vector{Int64}) 
  Ld  = Matrix{Int}(undef, 0, 0)
  tam  = length(nbetas)
  list_Ls  = []
  list_dims = []
  for i in 1: tam
    #println("Ldk: d: ",i, " k: ", nbetas[i])
    Ld_i  = Ldk(d, nbetas[i])
    #println(Ld_i)
    push!(list_Ls, Ld_i)
    if(size(Ld_i) != (1,1))
      push!(list_dims, size(Ld_i))
    else
      push!(list_dims, (1,1))
    end
  end
  
  for i in 1: tam
    L_row =  Matrix{Int}(undef, 0, 0)
    L_left = Matrix{Int}(undef, 0, 0)
    L_right = Matrix{Int}(undef, 0, 0)
    if (i > 1) 
      for left in 1: (i-1)
        ml = zeros(list_dims[i][1], list_dims[left][2])
        if (size(L_left, 1) == 0)
          L_left = ml
        else
          L_left = hcat(L_left, ml)
        end
      end
    end
    if (i < tam) 
      for right in (i+1):tam
         mr = zeros(list_dims[i][1], list_dims[right][2]) 
         if (size(L_right, 1) == 0)  
          L_right = mr
        else size(L_right, 1) == size(mr, 1)  # Check if row dimensions match
            L_right = hcat(L_right, mr)
        end
      end
    end
    if (size(L_left, 1) == 0)
      L_row = list_Ls[i]
    else 
      L_row = hcat(L_left, list_Ls[i])
    end
    
    if (size(L_right, 1) != 0)
    #  L_row = list_Ls[i]
    #else
      L_row = hcat(L_row,  L_right)
    end
    
    if (size(Ld, 1) == 0)
      Ld    = L_row
    else
      Ld    = vcat(Ld, L_row)
    end
  end
  return Ld
end

#
# Inverse_L
#
function Inverse_L(nbetas::Vector{Int64}) 
  Ldi_2  = Matrix{Int}(undef, 0, 0)
  tam = length(nbetas)
  for i in 1:tam
    Block_dim = Int.(nbetas[i])
    Block_i = LowerTriangular( ones(Block_dim, Block_dim))
    # Block_row
    B_row = Matrix{Int}(undef, 0, 0)
    B_left = Matrix{Int}(undef, 0, 0)
    B_right = Matrix{Int}(undef, 0, 0)

    if (i > 1) 
      for left in 1: (i-1)
        ml = zeros(Block_dim, nbetas[left])
        if (size(B_left,1) == 0)
          B_left  = ml
        else
          B_left  = hcat(B_left, ml)
        end
      end
    end

    if (i < tam) 
      for right in (i+1):tam
        mr = zeros(Block_dim, nbetas[right])
        if (size(B_right, 1) == 0)
          B_right = mr
        else
          B_right = hcat(B_right, mr)
        end
      end
    end

    if (size(B_left,1) == 0)
      B_row = Block_i
    else
      B_row = hcat(B_left, Block_i)
    end

    if (size(B_right,1) != 0)
    #  B_row = B_row, B_right)
    #else
      B_row = hcat(B_row, B_right)
    end
        
    if (size(Ldi_2, 1) == 0)
      Ldi_2 = B_row
    else
      Ldi_2 = vcat(Ldi_2, B_row)
    end

  end

  return Ldi_2
end

#
# Bic_GLMNet
#
function bic_glmnet(x::Matrix{Float64}, y::Vector{Float64}, alpha::Number; standardize = false, intercept = false, penalty_factor = NaN)
	"""
  On glmnet standardize & intercept have default values of TRUE
  alpha = 1 => LASSO
  """
  if abs(alpha - 1) > 1
    alpha = 1.0
  end
  
  if (any(isnan,penalty_factor))
		lasso =  glmnet(x, y, standardize=standardize, intercept=intercept, alpha=alpha)
	else
    lasso = glmnet(x, y, standardize=standardize, intercept=intercept, alpha=alpha, penalty_factor=penalty_factor)
  end
  
  #lasso =  glmnet(H, y)
  k  = size(lasso.betas,2) #<- lasso1$dim[2]
	n  = size(x, 1) #<- lasso1$nobs
  p  = size(x, 2)  # /n<- lasso1$dim[1]
  df = 0.0

	logLike = fill(0.0, k, 1) #<- rep(NA, times=k)
	bic     = fill(0.0, k, 1)#<- rep(NA, times=k)
	
	# BIC from Wang et al. (2007): Veiga - Epprecht - Guégan
	for i in 1:k
    sigma2  = 0
	  sigma2  = 1/n * sum((y .- x * lasso.betas[:, i]).^2)
    df = count( !=(0.0), lasso.betas[:, i])
    bic[i]  = log(sigma2) + 1/n * log(n) * df
  end

	# Smallest BIC
  idx = argmin(bic)[1]
	bicbeta = lasso.betas[:,idx]
	biclambda = lasso.lambda[idx]
	bick = idx #lasso1 
  bicitcp = lasso.a0[idx]
  
	# Considering the Intercept
	parameters  = lasso.betas 
    		     
  return (logLike = logLike, bic = bic, betas = lasso.betas, df = df , lambda = biclambda,
          beta = bicbeta, bick = bick, parameters= parameters, lambdas = lasso.lambda,
          itcp =bicitcp)
	
end

#
# Adaptive LASSO wo Outliers
#
function Ada_LASSO_intercept(x::Matrix{Float64}, y::Vector{Float64}, standardize = false, intercept = false)
  
  # First step: initial Ridge solution  => alpha = 0
  bic_ridge  = bic_glmnet(x, y, 0.0, standardize=standardize)
  
  gamma = 1
  w = 1 ./abs.(bic_ridge.beta) .^gamma  ## Using gamma = 1

  w .= ifelse.(isinf.(w), 1e4, w)
  
  # Ada LASSO solution => alpha = 1
  bic_adalasso  = bic_glmnet(x, y, 1.0, standardize=false, intercept=intercept, penalty_factor=w)

  # coef1 is theta
  coef1 = bic_adalasso.beta
  coef0 = bic_adalasso.itcp  # Intercept
  
  return (coef = coef1, parameters = bic_adalasso.parameters, 
          betas = bic_adalasso.betas, df = bic_adalasso.df, 
          lambda = bic_adalasso.lambda, 
          bick = bic_adalasso.bick, bic = bic_adalasso.bic, 
          itcp = coef0)

end

#
# Adaptive LASSO with Outliers
#
function Ada_LASSO_intercept_outlier(x::Matrix{Float64}, y::Vector{Float64}, x2::Matrix{Float64}, standardize = false, intercept = false)
  #  Handles Outliers
  
  # First step: initial Ridge solution  => alpha = 0
  bic_ridge  = bic_glmnet(x, y, 0.0, standardize=standardize)
  
  # Ridge regression gamma
  gamma = 1

  w = 1 ./abs.(bic_ridge.beta) .^gamma  ## Using gamma = 1
  w .= ifelse.(isinf.(w), 1e6, w)

  # Find Outlier Weights
  res = y - x * bic_ridge.beta
  wd =  1 ./abs.(res).^gamma
  wd .= ifelse.(isinf.(wd), 1e6, wd)

  w_extend = vcat(w, wd) 
  
  # Ada LASSO solution => alpha = 1
  bic_adalasso          = bic_glmnet(x, y, 1.0, standardize=false, intercept=intercept, penalty_factor=w)
  bic_adalasso_outliers = bic_glmnet(x2, y, 1.0, standardize=false, intercept=intercept, penalty_factor=w_extend)

  # coef1 is theta
  coef1 = bic_adalasso.beta
  coef0 = bic_adalasso.itcp  # Intercepto!!
  coef1o = bic_adalasso_outliers.beta
  coef0o = bic_adalasso_outliers.itcp  # Intercepto!!
    
  return (coef = coef1, parameters = bic_adalasso.parameters, 
          betas = bic_adalasso.betas, df = bic_adalasso.df, 
          lambda = bic_adalasso.lambda, 
          bick = bic_adalasso.bick, bic = bic_adalasso.bic,
          bicko = bic_adalasso_outliers.bick, bico = bic_adalasso_outliers.bic, 
          itcp = coef0,
          coefo = coef1o, itcpo = coef0o)

end
