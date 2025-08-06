#
#  IFL: Iterative Fused LASSO Algorithm
#
#  last update: AM 17/07/25
#

using LinearAlgebra
using GLMNet

# Supporting Function
include("supporting_functions.jl")

#
# IFL
#
"""
IFL(y, x; handle_outliers = true, intercept = true, CD = false)

Intercept = true => There is an intercept in x, then once the data is centralized, 
  it can be removed from the first column of X (zeros) 

CD = false # uses glmnet
CD = true  # uses Coordinate Descent
  
Returns β_np_1 = v_beta_Par , β_n_p = m_beta_Par,                 # Solutions without Outliers
        βo_np_1 = v_beta_Paro , βo_n_p = m_beta_Paro,             # Solutions with Outliers
        β0 = β0_hat,                                              # Estimated Intercept
        y_hat =  y_original_hat ,                                 # Insample Estimators without Outliers
        y_hato = y_original_hat_woutl,                            # Insample Estimators with Outliers
        mean_x  = mean_x , mean_y  = mean_y,                      # column-wise mean of X and mean of y
        bic_value = bic_value,                                    # BIC value model without outliers
        bic_value_outliers = bic_value_outliers,                  # BIC value model with outliers
        chosen_var = chosen_variables,                            # List of selected columns in X
        chosen_var_bk = chosen_variables_break,                   # List of selected columns with estim breaks in X
        chosen_var_outliers = chosen_variables_outliers,          # List of selected columns in X (model w/outliers)
        chosen_var_outliers_bk = chosen_variables_outliers_break, # List of selected columns with estim breaks in X (model w/outliers)
        RMSE_inS = RMSE_inS, RMSE_inSo = RMSE_inSo,               # In Sample RMSE (models with and without outliers)
        ind_out = ind_out,                                        # list of outlier in y  
        IFL_t0 = IFL_t0                                           # time to converge

Last review: 19 Jul 25

"""
function IFL(y, x; handle_outliers = true, intercept = true,CD = false)
  """
  CD = false # uses glmnet
  CD = true  # uses Coordinate Descent
  """
  
  # Store originals
  x_original = x
  y_original = y

  # Always center the data
  mean_x = mean(x_original, dims=1)
  mean_y = mean(y_original)
  β0 = mean_y
  y = y_original .- mean_y
  x = x_original .- mean_x

  # No need to process intercept column that equals 0!
  if(intercept) 
      x = x[:,2:end]  
  end

  ind_out=[]
  
  ########################################################
  # Problem Dimensions
  n = length(y) 
  p = size(x)[2]
  ########################################################
  
  global XX = Matrix{Float64}(undef, 0, 0)  # Alterei de Int para Float64!!! 27/05/25
  for i in 1:p
      if (size(XX, 1) == 0)
          XX = Diagonal(x[:, i])  # Initialize with the first diagonal matrix
      else
          XX = hcat(XX, Diagonal(x[:, i]))  # Horizontally concatenate diagonal matrices
      end
  end
  
  # Setup Iteractive Fused LASSO 
  nbetas_In  = repeat([n], p)
  nbetas_In_Tot =sum(nbetas_In)
  nbetas_Out = repeat([0], p)
  nbetas_Out_Tot = sum(nbetas_Out)

  W = I(n*p)

  # Main loop
  IFL_t0 = @elapsed begin 
    Ld = L_Matrix(1, Int.(nbetas_In))
    Ld_i = Inverse_L(Int.(nbetas_In))
    
    # Update model
    H = XX * W * Ld_i
    # Ada LASSO solution with smallest BIC
    # Intercept is managed outside glmnet, therefore intercept here is always FALSE
    theta_hat_Est = Ada_LASSO_intercept(H, y, standardize = false, intercept = intercept, CD = CD)
    theta_hat = theta_hat_Est.coef
    
    # Gamma_d
    Gamma_d = abs.(theta_hat) .> 0
    #Gamma_d_Acc = cumsum(Gamma_d) 
      
    # Checking for 1s in free betas
    aux1 = 0
    nbetas_In_Acc = cumsum(nbetas_In)
    for i in 1:p
      aux2 = aux1 + min(1,nbetas_In[i]) 
      Gamma_d[(aux1+1):aux2] .= 1
      aux1 = nbetas_In_Acc[i]
    end
      
    # Calculating qty of diff betas in the next iteration
    nbetas_Out = zeros(p)
    nbetas_Out[1] = sum(Gamma_d[1:nbetas_In_Acc[1]])
    for j in 2:p
      aux1 = nbetas_In_Acc[j-1] + 1
      aux2 = nbetas_In_Acc[j]
      nbetas_Out[j] = sum(Gamma_d[aux1:aux2])
    end
    nbetas_Out_Tot = sum(nbetas_Out)
      
    # Updating Matrix M
    if(nbetas_In_Tot != nbetas_Out_Tot)  
      g_d = cumsum(Gamma_d) 
      M = zeros(Int(nbetas_In_Tot), Int(nbetas_Out_Tot))
      for i in 1:nbetas_In_Tot
        if(Gamma_d[i] != 0)
          M[i, g_d[i]] = 1
        else
          M[i, g_d[i-1]] = 1
        end
      end
              
      # Update W
      W  =  W * M
      nbetas_In = nbetas_Out
      nbetas_In_Tot = sum(nbetas_In)
      LdP = L_Matrix(1, Int.(nbetas_In))
      global LdP_i = Inverse_L(Int.(nbetas_In))
      # Partial solution: solve problem on new dimensions 
      global HP = XX * W * LdP_i 
        
      if (handle_outliers) 
        # Handlig Outliers
        global HPo = hcat(HP, Matrix{Float64}(I, Int(n), Int(n)))
            
        # Ada LASSO solution with smallest BIC
        theta_hatP_Est = Ada_LASSO_intercept_outliers(HP, y, HPo, 
                            standardize = false, intercept = intercept, CD = CD)
          
        # Solution without outliers
        global theta_hatP = theta_hatP_Est.coef
        global theta_hatP_itcp = theta_hatP_Est.itcp
        # BIC value
        global bic_value = theta_hatP_Est.bic[theta_hatP_Est.bick] 

        # Solution with Outliers
        global theta_hatPo = theta_hatP_Est.coefo
        global theta_hatPo_itcpo = theta_hatP_Est.itcpo
        # BIC value
        global bic_value_outliers = theta_hatP_Est.bico[theta_hatP_Est.bicko] 
          
        global y_hato = HPo * theta_hatPo           # With Intercept 
        global y_hat = HP * theta_hatP              # Without Intercept
          
        # No. of Outliers
        ind_out = findall(pts -> pts > 0, theta_hatPo[(end-n+1):end])  
                  
      else
        theta_hatP_Est = Ada_LASSO_intercept(HP, y, standardize = false, intercept = intercept, CD = CD)
         
        global theta_hatP = theta_hatP_Est.coef
        global theta_hatP_itcp = theta_hatP_Est.itcp
        # BIC value
        global bic_value = theta_hatP_Est.bic[theta_hatP_Est.bick] 
        global bic_value_outliers = nothing 
        
        global y_hat = HP * theta_hatP # Without Intercept
        global y_hato = zeros(n) 
          
      end # Handle Outliers
    end # Updating M
  end  # Time

  # β estimated without outliers
  v_beta_Par = W * LdP_i * theta_hatP  # Without Intercept
  m_beta_Par = reshape(v_beta_Par, n, p) 

  # β estimated with outliers
  if(handle_outliers)                 
    v_beta_Paro = W * LdP_i * theta_hatPo[1:end-n] # dim(theta_hatPo) is (np + n) times 1
    # Without Intercept
    m_beta_Paro = reshape(v_beta_Paro, n, p) 
  else # without outliers => always returns a value for v_beta and m_beta 
    v_beta_Paro = v_beta_Par
    m_beta_Paro = m_beta_Par 
  end
      
  # Estimate Intercept
  β0_hat = mean(y_original) .- mean(XX, dims = 1) * v_beta_Par 
  
  # Always consider the estimated intercept. Since data was centralized, it needs to be decentralized back.
  y_original_hat = β0_hat .+ y_hat
  
  if(handle_outliers)
    y_original_hat_woutl  = β0_hat .+ y_hato
  else
    y_original_hat_woutl =  y_original_hat
  end
  
  # RMSE In Sample
  RMSE_inS  = sqrt(1/length(y)*sum((y.-y_hat).^2))
  RMSE_inSo = sqrt(1/length(y)*sum((y.-y_hato).^2))

  # Selected variables X unselected variables, and variables with break 
  chosen_variables = zeros(p)
  chosen_variables_break =  zeros(p)
  chosen_variables_outliers = zeros(p)
  chosen_variables_outliers_break =  zeros(p)
  
  for j in 1:p
    if (abs.(sum(m_beta_Par[:,j])) > 0)
      chosen_variables[j] = 1
      featbk = false
      βref = m_beta_Par[1,j] # first value of m_beta_Par[:,j] 
      for k in 1:n
        if(m_beta_Par[k,j] != βref) # At least one break
            featbk = true
        end
      end
      if(featbk)
        chosen_variables_break[j] = 1
      end
    end
    if (abs.(sum(m_beta_Paro[:,j])) > 0)
      chosen_variables_outliers[j] = 1
      featbk = false
      βref = m_beta_Paro[1,j] # first value of m_beta_Par[:,j] 
      for k in 1:n
        if(m_beta_Paro[k,j] != βref) # At least one break
            featbk = true
        end
      end
      if(featbk)
        chosen_variables_outliers_break[j] = 1
      end
    end
  end

  return (β_np_1 = v_beta_Par , β_n_p = m_beta_Par,
          βo_np_1 = v_beta_Paro , βo_n_p = m_beta_Paro,
          β0 = β0_hat, 
          y_hat =  y_original_hat ,  y_hato = y_original_hat_woutl,
          mean_x  = mean_x , mean_y  = mean_y, 
          bic_value = bic_value, bic_value_outliers = bic_value_outliers,            
          chosen_var = chosen_variables,
          chosen_var_bk = chosen_variables_break,
          chosen_var_outliers = chosen_variables_outliers,
          chosen_var_outliers_bk = chosen_variables_outliers_break,
          RMSE_inS = RMSE_inS, RMSE_inSo = RMSE_inSo,
          ind_out = ind_out, IFL_t0 = IFL_t0)

end
