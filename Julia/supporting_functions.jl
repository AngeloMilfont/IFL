#
#  Supporting functions 
#
#  AM 22/09/24
#
#  Last Update: 13/02/25
#

using Plots
using LinearAlgebra
using GLMNet

#
# MSE
#
function MSE_xy(x, y::Vector{Float64})
  return (1/length(x)) * sqrt(sum((x - y).^2))
end

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
  #
  #  Não é válida para d>1
  #
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
		#println("LASSO")
		lasso =  glmnet(x, y, standardize=standardize, intercept=intercept, alpha=alpha)
	else
    #println("AdaLASSO")
		lasso = glmnet(x, y, standardize=standardize, intercept=intercept, alpha=alpha, penalty_factor=penalty_factor)
  end
  
  #lasso =  glmnet(H, y)
  k  = size(lasso.betas,2) #<- lasso1$dim[2]
	n  = size(x, 1) #<- lasso1$nobs
  p  = size(x, 2)  # /n<- lasso1$dim[1]
  df = 0.0

	logLike = fill(0.0, k, 1) #<- rep(NA, times=k)
	bic     = fill(0.0, k, 1)#<- rep(NA, times=k)
	
	# Artigo Álvaro - Camila Epprecht - Guégan
	for i in 1:k
    sigma2  = 0
	  sigma2  = 1/n * sum((y .- x * lasso.betas[:, i]).^2)
    df = count( !=(0.0), lasso.betas[:, i])
    bic[i]  = log(sigma2) + 1/n * log(n) * df
  end

	# Escolhendo o valor com menor bicV2
	idx = argmin(bic)[1]
	bicbeta = lasso.betas[:,idx]
	biclambda = lasso.lambda[idx]
	bick = idx #lasso1 
  bicitcp = lasso.a0[idx]
  # BIC plot
  p1 = plot(bic)
  #display(p1)

	# Considering the Intercept
	parameters  = lasso.betas 
  
  #println("Saídas")
  #println("Alpha  ", alpha)
  #println("n ",  n, " p ", p," k ", k)
  #println("size y ",  length(y))
  #println("size x ",  size(x))
  #println("size logLike ",  size(logLike))
  #println("size bic ",  size(bic))
  #println("size betas ",  size(lasso.betas))
  #println("df ",  df)
  #println("lambda ",  size(biclambda))
  #println("beta ",  size(bicbeta))
  #println("bic k ",  bick)
  #println("parameters ",  parameters)
  		     
  return (logLike = logLike, bic = bic, betas = lasso.betas, df = df , lambda = biclambda,
          beta = bicbeta, bick = bick, parameters= parameters, lambdas = lasso.lambda,
          itcp =bicitcp)
	
end

#
# Adaptive LASSO wo Outliers
#
function Ada_LASSO_intercept(x::Matrix{Float64}, y::Vector{Float64}; standardize = false, intercept = false, CD = false)
  #  13/04/25
  #  inclusão dois parâmetros standardize e intercept, com valores default  = false
  #  Se standardize é FALSO, então GLMNET não padroniza dados, considera que dados foram padronizados pelo usuário.
  #  Se intercept é FALSO, então GLMNET força o intercepto igual a zero!
  # 
  #  05/08/25
  #  keyword CD: false, then glmnet or true then coordinate descent
  #

  # First step: initial Ridge solution  => alpha = 0
  bic_ridge  = bic_glmnet(x, y, 0.0, standardize=standardize)  
  gamma = 1
  w = 1 ./abs.(bic_ridge.beta) .^gamma  ## Using gamma = 1
  w .= ifelse.(isinf.(w), 1e4, w)
  
  # Ada LASSO solution => alpha = 1
  if (CD)
    bic_adalasso  = bic_CD(x, y, 1.0, standardize=false, intercept=intercept, penalty_factor=w)
  else 
    bic_adalasso  = bic_glmnet(x, y, 1.0, standardize=false, intercept=intercept, penalty_factor=w)
  end

  #return (logLike = logLike, bic = bic, betas = lasso.betas, df = df , lambda = biclambda,
  #beta = bicbeta, bick = bick, parameters= parameters)

  # coef1 is theta
  coef1 = bic_adalasso.beta
  coef0 = bic_adalasso.itcp  # Intercepto!!
  
  return (coef = coef1, parameters = bic_adalasso.parameters, 
          betas = bic_adalasso.betas, df = bic_adalasso.df, 
          lambda = bic_adalasso.lambda, 
          bick = bic_adalasso.bick, bic = bic_adalasso.bic, 
          itcp = coef0)

end

#
# Adaptive LASSO with Outliers
#
function Ada_LASSO_intercept_outliers(x::Matrix{Float64}, y::Vector{Float64}, x2::Matrix{Float64};
                                       standardize = false, intercept = false, CD = false)
  #  Criado em 05/05/25
  #  Tratamento de Outliers
  #  wo Outliers alterado em 13/04/25
  #  inclusão dois parâmetros standardize e intercept, com valores default  = false
  #  Se standardize é FALSO, então GLMNET não padroniza dados, considera que dados foram padronizados pelo usuário.
  #  Se intercept é FALSO, então GLMNET força o intercepto igual a zero!
  #
  #  05/08/25
  #  keyword CD: false, then glmnet or true then coordinate descent
  #
  
  # First step: initial Ridge solution  => alpha = 0
  bic_ridge  = bic_glmnet(x, y, 0.0, standardize=standardize)
  
  # gamma for Ridge regressions
  gamma = 1

  w = 1 ./abs.(bic_ridge.beta) .^gamma  ## Using gamma = 1
  w .= ifelse.(isinf.(w), 1e8, w)

  # Find Outliers Weights
  res = y - x * bic_ridge.beta
  wd =  1 ./abs.(res).^gamma
  wd .= ifelse.(isinf.(wd), 1e8, wd)

  w_extend = vcat(w, wd) 
  
  # Ada LASSO solution => alpha = 1
  if (CD)
    bic_adalasso          = bic_CD(x, y, 1.0, standardize=false, intercept=intercept, penalty_factor=w)
    bic_adalasso_outliers = bic_CD(x2, y, 1.0, standardize=false, intercept=intercept, penalty_factor=w_extend)
  else
    bic_adalasso          = bic_glmnet(x, y, 1.0, standardize=false, intercept=intercept, penalty_factor=w)
    bic_adalasso_outliers = bic_glmnet(x2, y, 1.0, standardize=false, intercept=intercept, penalty_factor=w_extend)
  end

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

#
# Function to read all subdirectories in a directory and store them in a list
#
function read_subdirectories(dir_path::String)
  all_items = readdir(dir_path)  # Get all items (files and subdirectories) in the directory
  subdirectories = []  # Initialize an empty list to store subdirectories
  
  for item in all_items
      full_path = joinpath(dir_path, item)  # Get the full path of each item
      if isdir(full_path)  # Check if the item is a directory
          push!(subdirectories, item)  # Store the subdirectory path in the list
      end
  end

  return subdirectories
end

#
# feature_plot
#
function feature_plot(p::Int64,q::Int64, ndd::Number, betas::Matrix{Float64}, beta_Est::Matrix{Float64},
     d::Int64, lmin::Int64, lmax::Int64)
     
     len = size(betas,1)
     nomeBetas = Any[]
     for i in 1:len
        nome  = "beta" * string(i)
        push!(nomeBetas, nome)
    end

    # create beta series
    beta_series = []
    for i in 1:p
        beta_serie = []
        for k in 1:len
            for j in 1: Int(ndd)
                push!(beta_serie, betas[k,i])
            end
        end
        push!(beta_series, beta_serie)
    end
    
    # Create Sub plots
    ngraf = length(beta_series) # qde graficos: 1 por componente p
    plot_list = []
    #scat_list = []
    #conc_list = []
    for i in 1:ngraf
        ymin, ymax = 0,9* minimum(beta_Est[:,i]), 1,1* maximum(beta_Est[:,i])
        pplot = plot(beta_series[i], title= "d = "* string(d)*". Feat: "*string(i),
                label = "original") #,ylims=(ymin, ymax))
        plot!(beta_Est[:,i], seriestype="scatter", label = "estimated") #,ylims=(ymin, ymax))
        push!(plot_list, pplot)
        #push!(scat_list, scat)
        #push!(conc_list, plot(pplot, scat, layout = (1, 1)))  # Side-by-side display (1 row, 2 columns)
    end

    #for p in conc_list
    #    display(p)
    #end
    
    for p in plot_list
        display(p)
    end


end

#
# Confusion Matrix
#
function confusion_matrix(beta::Vector{Float64}, beta_hat::Vector{Float64}, p::Int64, q::Int64, tol::Float64)

  TP, FP, TN, FN  = 0,0,0,0
  
  n = Int(length(beta)/p)
  for m in 1:n
    for i in 1:p
      pos = (m-1) *p + i
      if (i <= q)  # this one must be different than zero
        if(abs(beta[pos]-beta_hat[pos]) <= tol) 
          TP += 1 # Acertou relevante
        else 
          FP += 1 # Errou relevante     # era FN - alterado em 27/10/24
        end
      else  # this one must be zero
        if(beta_hat[pos] <= tol) 
          TN += 1 # acertou Irrelevante
        else 
          FN += 1  # errou Irrelevante  # era FP - alterado em 27/10/24
        end
      end
    end  
  end
  
  #println("                         Matriz de Confusão", "\n",
  #    "                           % Estimado como", "\n", 
  #    "                       Relevante (P)   Irrelevante (N)", "\n",
  #    "      Relevante (P)        ", round(TP/(n*p),digits = 2), "       ", round(FN/(n*p),digits = 2) , "\n",
  #    "Real\n",
  #    "      Irrelevante (N)      ", round(FP/(n*p),digits = 2), "       ", round(TN/(n*p),digits = 2) , "\n",
  #    "check ", (n*p) - TP - FN - TN - FP, "\n",
  #    "% Acuracia ", round((TP+TN)/float(n*p),digits = 3),"  Erro de classificação ", 
  #                  round((FN+FP)/(n*p),digits = 3), "\n\n" )
  
  return [TP, TN, FP, FN] 
end

#
# Descripive Statistics of the parameters estimate
#
function descriptive_stats(beta::Vector{Float64}, beta_hat::Vector{Float64}, p::Int64, q::Int64)
  #
  #  Epprecht_Veiga_Guégan (pdf.16)
  #
  # Vector of Bias (observations 1,2,...,p)
  #   for each candidate variable (p) vs. the true parameter for all n observations 
  # Vector of MSE (observations 1,2,...,p)
  #   for each candidate variable (p) vs. the true parameter for all n observations 
  #
  # returns sum of components of each vector px1
  
  n = Int(length(beta)/p)
  #beta = matrix(beta, nrow = n*p, ncol = 1)
  #beta_hat <- unlist(beta_hat) #matrix(beta_hat, nrow = n*p, ncol = 1)
  
  vBias = zeros(1, p)
  vMSE  = zeros(1, p)
  
  for i in 1:p
    Bias, MSE  = 0,0
    for j in 1:n
      pos = (i-1)*n + j
      Bias = Bias + abs(beta_hat[pos]-beta[pos])
      MSE  = MSE  + (beta_hat[pos]-beta[pos])^2  
    end
    
    vBias[i] = 1/(n*p) * Bias # 1/np to allow comparison among models with different qtys of candidate variables.
    vMSE[i]  = 1/(n*p) * MSE  # 1/np to allow comparison among models with different qtys of candidate variables.
  end
  
  Bias_R = sum(vBias)
  MSE_R  = sum(vMSE)
  
  #println("Descriptive Statistics\n",
  #    " of each candidate variable (p) vs. the true parameter for every observation n.\n",
  #    "  Bias : ",Bias_R, "\n",
  #    "  MSE  : ",MSE_R, "\n\n\n")
      
  return [Bias_R, MSE_R] 
end

###############################################################################
# Coordinate Descent
###############################################################################

# --- Soft-threshold function ---
function soft_threshold(rho, λ)
  if abs(rho) <= λ
      return 0.0
  elseif rho > λ
      return rho - λ
  else
      return rho + λ
  end
end

# --- AdaLASSO coordinate descent for fixed λ (standardized X) ---
function Alasso_λ(ys, Xs, λ, β, w; max_iter=10_000, tol=1e-4)
  """
  For Standardized X and centralized y 
  """
  p = size(Xs)[2]   
  m = 0
  X_col_norm2 = sum(Xs.^2, dims=1) |> vec
  
  global beta_path= []

  for iter in 1:max_iter
      #iter = 1 
      β_old = copy(β)
      push!(beta_path, β_old)
      for j in 1:p
          #println(m) 
          #j = 6 
          m += 1 
          j_idx = (m - 1) % p + 1
          # Optimality condition for OLS: x'(y-xβ) = x'u = x'r_j = 0
          # AdaLASSO
          r_j = ys .- Xs * β .+ Xs[:, j_idx] * β[j_idx]
          G_j = dot(Xs[:, j_idx], r_j)
          β[j_idx] = soft_threshold(G_j , λ * w[j_idx])/ X_col_norm2[j_idx]
      end
      if norm(β - β_old, 2) < tol
          break
      end
  end

  return β, beta_path
end

# --- AdaLASSO coordinate descent ---
function AdaLASSO_CD(y, X, w ; max_sols=20) 
  n,p = size(X) 
  β_list = []
  βu_list = []
  itcp_list = [] 
  β0 = zeros(p)
  
  # Standardize X and center y
  X_mean = mean(X, dims=1)
  X_std = std(X, dims=1)
  Xs = (X .- X_mean) ./ X_std
  y_mean = mean(y)
  ys = y .- y_mean

  # λ path: from large to small (log-spaced)
  λ_max = maximum(abs.(Xs' * (ys .- mean(ys)))) / n
  λ_path = exp10.(range(log10(λ_max), stop=log10(1/(n*p)), length=max_sols))
  
  β_old = copy(β0)
  for λ in λ_path
      #λ = λ_path[4]
      # function expects standardized X and y
      β_new, beta_path = Alasso_λ(ys, Xs, λ, β_old, w)
      push!(β_list, β_new)
      β_old = copy(β_new)
      # Unstandardize and SAVE coefficients
      β_unscaled = β_new ./ vec(X_std)
      intercept = y_mean - sum((X_mean .* β_unscaled))
      push!(βu_list, β_unscaled)
      push!(itcp_list, intercept)
  end
  
  β_matrix     = hcat(β_list...)
  βu_matrix    = hcat(βu_list...)
  itcp_vector  = hcat(itcp_list...)
     
  bic = fill(0.0, max_sols, 1)
     
  # Artigo Álvaro - Camila Epprecht - Guégan
  for i in 1:max_sols
     sigma2  = 0
     sigma2  = 1/n * sum((y .- X * β_matrix[:, i]).^2)
     global df = count( !=(0.0),  β_matrix[:, i]) 
     bic[i]  = log(sigma2) + 1/n * log(n) * df
  end

  # Escolhendo o valor com menor bicV2
  idx = argmin(bic)[1]
  bicbeta = β_matrix[:,idx] # without Intercept
  biclambda = λ_path[idx]
  bick = idx 
  
  return (βs_sol = bicbeta,
      βu_sol = βu_matrix[:,idx],
      itcp_sol = itcp_vector[idx], 
      λsol = biclambda, 
      BIC = bic[idx], 
      df = df,
      βs_sols = β_matrix,
      βu_sols = βu_matrix,
      itcp_sols = itcp_list,
      λ_pth = λ_path, 
      bics = bic,
      bick = bick
  )
  
end

# --- LASSO coordinate descent for fixed λ (standardized X) ---
function lasso_λ(ys, Xs, λ, β; max_iter=10_000, tol=1e-6)
  """
  For Standardized X and centralized y 
  """
  p = size(Xs)[2]   
  m = 0
  X_col_norm2 = sum(Xs.^2, dims=1) |> vec
  
  global beta_path= []

  for iter in 1:max_iter
      #iter = 1 
      β_old = copy(β)
      push!(beta_path, β_old)
      for j in 1:p
          #println(m) 
          #j = 1 
          m += 1 
          j_idx = (m - 1) % p + 1
          # Optimality condition for OLS: x'(y-xβ) = x'u = x'r_j = 0
          r_j = ys .- Xs * β .+ Xs[:, j_idx] * β[j_idx]
          G_j = dot(Xs[:, j_idx], r_j)
          check = β[j_idx]
          β[j_idx] = soft_threshold(G_j / X_col_norm2[j_idx], λ)
          if (check == 0) && (β[j_idx] != 0)
              println("Component ", j_idx, " leaves zero.")
          end
      end
      if norm(β - β_old, 2) < tol
          break
      end
  end

  return β, beta_path
end

# --- LASSO coordinate descent ---
function LASSO_CD(y, X; max_sols=20)
      
  n,p = size(X) 
  β_list = []
  βu_list = []
  itcp_list = [] 
  β0 = zeros(p)
  
  # Standardize X and center y
  X_mean = mean(X, dims=1)
  X_std = std(X, dims=1)
  Xs = (X .- X_mean) ./ X_std
  y_mean = mean(y)
  ys = y .- y_mean

  # λ path: from large to small (log-spaced)
  λ_max = maximum(abs.(Xs' * (ys .- mean(ys)))) / n
  λ_path0 = exp10.(range(log10(2 * λ_max), stop=log10(1/(n*p)), length=max_sols))
  # Create New path based on λ critical for every component
  λcrits = []
  for j in 1:p
      λ0 = 0
      for i in 1:n
          λ0 = λ0 + abs(-ys[i]*Xs[i,j])/2 # W in the next version 
      end
      #println("Component ", j, " critic value is ", round(log10(λ0), digits = 3))
      push!(λcrits, log10(λ0))
  end
  λ_path = sort(λcrits, rev = true) # Descending
  pushfirst!(λ_path, 2*λcrits[1])
  
  # Returns
  λ_path = λ_path0
  max_sols = length(λ_path)
  
  β_old = copy(β0)
  for λ in λ_path
      #λ = λ_path[2]
      println(" lambda path ", log10(λ))
      # function expects standardized X and y
      β_new, beta_path = lasso_λ(ys, Xs, λ, β_old)
      println("β new ", β_new)
      push!(β_list, β_new)
      β_old = copy(β_new)
      # Unstandardize and SAVE coefficients
      β_unscaled = β_new ./ vec(X_std)
      intercept = y_mean - sum((X_mean .* β_unscaled))
      push!(βu_list, β_unscaled)
      push!(itcp_list, intercept)
  end
  
  β_matrix     = hcat(β_list...)
  βu_matrix    = hcat(βu_list...)
  itcp_vector  = hcat(itcp_list...)
     
  bic = fill(0.0, max_sols, 1)
     
  # Artigo Álvaro - Camila Epprecht - Guégan
  for i in 1:max_sols
     sigma2  = 0
     sigma2  = 1/n * sum((y .- X * β_matrix[:, i]).^2)
     global df = count( !=(0.0),  β_matrix[:, i]) 
     bic[i]  = log(sigma2) + 1/n * log(n) * df
  end

  # Escolhendo o valor com menor bicV2
  idx = argmin(bic)[1]
  bicbeta = β_matrix[:,idx] # without Intercept
  biclambda = λ_path[idx]
  bick = idx 
  
  return (βs_sol = bicbeta,
      βu_sol = βu_matrix[:,idx],
      itcp_sol = itcp_vector[idx], 
      λsol = biclambda, 
      BIC = bic[idx], 
      df = df,
      βs_sols = β_matrix,
      βu_sols = βu_matrix,
      itcp_sols = itcp_list,
      λ_pth = λ_path, 
      bics = bic,
      bick = bick
  )
  
end

#
# BIC_GLMNet_CD AM
#
function bic_CD(x::Matrix{Float64}, y::Vector{Float64}, alpha::Number; standardize = false, intercept = false, penalty_factor = NaN)
	"""
  bic_glmnet as Coordinate Descent
  """
  if abs(alpha - 1) > 1
    alpha = 1.0
  end
  
  if (any(isnan,penalty_factor))
		#println("CD LASSO")
		#lasso =  glmnet(x, y, standardize=standardize, intercept=intercept, alpha=alpha)
    solution = LASSO_CD(y, x)
	else
    #println("CD AdaLASSO")
		#lasso = glmnet(x, y, standardize=standardize, intercept=intercept, alpha=alpha, penalty_factor=penalty_factor)
    solution = AdaLASSO_CD(y, x, penalty_factor)
  end
    
	# Considering the Intercept
	parameters = solution.βs_sols # lasso.betas 
  
  return (logLike = nothing, bic = solution.bics, betas = solution.βu_sols, df = solution.df , lambda = solution.λsol,
          beta = solution.βu_sol, bick = solution.bick, parameters= parameters, lambdas = solution.λ_pth,
          itcp =solution.itcp_sol)
	
end
