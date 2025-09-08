#
#  IFL: Iterative Fused LASSO Algorithm
#
#  last update: AM 12/08/25
#

using LinearAlgebra
using GLMNet

# Supporting Function
include("supporting_functions.jl")

#
# IFL
#
"""
IFL(y, x; handle_outliers = true, force_intercept = false, CD = false)

force_intercept = true => for simulated data only. The data was created with a first column of x = 1s.
  Then, once the data is centralized, it becomes a column of zeros, and  it can be removed 
  from X. Thus ncols(X) := ncols(x) - 1  

Data is centralized inside IFL, so the intercept is calculated outside the solution engine.

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

Last review: 08 Aug 25

"""
function IFL(y, x; handle_outliers = true, force_intercept = false, CD = false, verbose = false)
  """
  """
  
  # Store originals
  x_original = x
  y_original = y

  # Always center the data
  mean_x = mean(x_original, dims=1)
  mean_y = mean(y_original)
  # Removes the intercept
  y = y_original .- mean_y 
  x = x_original .- mean_x

  # If assume_intercept is set to true, 
  #   then there is no need to process its column that equals 0!
  if(force_intercept) 
      x = x[:,2:end]  # matrix dimension is reduced: p := p-1
  end

  ind_out=[]
  
  ####################################################################
  # Problem Dimensions
  n = length(y) 
  p = size(x)[2] # When Intercept is assumed p has already turned to p-1
  ####################################################################
    
  # Setup Iteractive Fused LASSO 
  nbetas_In  = repeat([n], p)
  nbetas_In_Tot =sum(nbetas_In)
  nbetas_Out = repeat([0], p)
  nbetas_Out_Tot = sum(nbetas_Out)

  global XX = Matrix{Float64}(undef, 0, 0)  # Alterei de Int para Float64!!! 27/05/25
  for i in 1:p
      if (size(XX, 1) == 0)
          XX = Diagonal(x[:, i])  # Initialize with the first diagonal matrix
      else
          XX = hcat(XX, Diagonal(x[:, i]))  # Horizontally concatenate diagonal matrices
      end
  end

  # Main loop
  IFL_t0 = @elapsed begin 
    
    time_H = @elapsed begin
      Hs = build_H(x)
    end
    if(verbose)
      println("Step 1, Build H: ", time_H)
    end 
    
    H = Matrix(Hs)         

    # Ada LASSO solution with smallest BIC
    # Intercept is being managed outside glmnet, therefore intercept here is always FALSE
    time_AdaS1 = @elapsed begin
        theta_hat_Est = Ada_LASSO_intercept(H, y, standardize = false, intercept = false, CD = CD)
    end 
    if(verbose)
      println("Step 1, Ada Step1: ", time_AdaS1)
    end 
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

      time_M = @elapsed begin   
        for i in 1:nbetas_In_Tot
          if(Gamma_d[i] != 0)
            M[i, g_d[i]] = 1
          else
            M[i, g_d[i-1]] = 1
          end
        end
      end
      if(verbose)
        println("Step 2, matrix M: ", time_M)
      end

      # Update W = M
      nbetas_In = nbetas_Out
      nbetas_In_Tot = sum(nbetas_In)
      
      global LdP_i = Inverse_L(Int.(nbetas_In))
      # Partial solution: solve problem on new dimensions 
      global HP = XX * M * LdP_i 
        
      if (handle_outliers) 
        # Handlig Outliers
        global HPo = hcat(HP, Matrix{Float64}(I, Int(n), Int(n)))
            
        # Ada LASSO solution with smallest BIC
        # Intercept is being managed outside glmnet, therefore intercept here is always FALSE
        time_AdaS3wO = @elapsed begin   
           theta_hatP_Est = Ada_LASSO_intercept_outliers(HP, y, HPo, 
                            standardize = false, intercept = false, CD = CD)
        end
        if(verbose)
          println("Step 3, AdaLASSO with Outliers: ", time_AdaS3wO)
        end
        
        # Solution without outliers
        global theta_hatP = theta_hatP_Est.coef
        global theta_hatP_itcp = theta_hatP_Est.itcp
        # BIC value
        global bic_value = theta_hatP_Est.bic[theta_hatP_Est.bick] 

        # Solution with Outliers
        global theta_hatPo = theta_hatP_Est.coefo
        #global theta_hatPo_itcpo = theta_hatP_Est.itcpo
        # BIC value
        global bic_value_outliers = theta_hatP_Est.bico[theta_hatP_Est.bicko] 
          
        global y_hato = HPo * theta_hatPo           # Without Intercept 
        global y_hat  = HP * theta_hatP              # Without Intercept
          
        # No. of Outliers
        ind_out = findall(pts -> pts > 0, theta_hatPo[(end-n+1):end])  
                  
      else
        time_AdaS3woO = @elapsed begin   
          theta_hatP_Est = Ada_LASSO_intercept(HP, y, standardize = false, intercept = false, CD = CD)
        end
        if(verbose)
          println("Step 3, AdaLASSO without Outliers: ", time_AdaS3woO)
        end
        
        global theta_hatP = theta_hatP_Est.coef
        #global theta_hatP_itcp = theta_hatP_Est.itcp
        # BIC value
        global bic_value = theta_hatP_Est.bic[theta_hatP_Est.bick] 
        global bic_value_outliers = nothing 
        
        global y_hat = HP * theta_hatP # Without Intercept
        global y_hato = zeros(n) 
          
      end # Handle Outliers
    end # Updating M
  end  # Time

  # β estimated without outliers
  v_beta_Par = M * LdP_i * theta_hatP     # Without Intercept
  m_beta_Par = reshape(v_beta_Par, n, p) 

  # β estimated with outliers
  if(handle_outliers)                 
    v_beta_Paro = M * LdP_i * theta_hatPo[1:end-n] # dim(theta_hatPo) is (np + n) times 1
    # Without Intercept
    m_beta_Paro = reshape(v_beta_Paro, n, p) 
  else # without outliers => always returns a value for v_beta and m_beta 
    v_beta_Paro = v_beta_Par
    m_beta_Paro = m_beta_Par 
  end
    
  # Estimate Intercept
  if(force_intercept)
    β0_hat = mean_y .- mean_x   * vcat(0, m_beta_Par[end,:]) # valid only for the last intercept.
    β0_hato = mean_y .- mean_x  * vcat(0, m_beta_Paro[end,:]) # valid only for the last intercept.
  else
    β0_hat = mean_y .- mean_x * m_beta_Par[end,:] # valid only for the last intercept.
    β0_hato = mean_y .- mean_x * m_beta_Paro[end,:] # valid only for the last intercept.
  end
    
  # Since data was centralized, it needs to be decentralized back.
  y_original_hat = y_hat .+ mean_y
  
  if(handle_outliers)
    y_original_hat_woutl  = y_hato .+ mean_y
  else
    y_original_hat_woutl =  y_original_hat
  end

  # Last Observation Check
  #println("Last observation (Centered) is                                :", y[end])
  #println()
  #println("Last observation (Centered) estimation WO Out         is      :", y_hat[end])
  #println("Last observation (Centered) estimation WO Out recalculated is :", x[end,:]' * m_beta_Par[end,:])
  ## It needs to take into consideration all dummy variables with Xout = HPo[end,:] #* theta_hatPo
  #println("Last observation (Centered) estimation Wt Out         is      :", y_hato[end])
  #println("Last observation (Centered) estimation Wt Out recalculated is :", HPo[end,:]' * theta_hatPo)
  #println()
  ## Check 
  #println("Last observation (rescaled) is                                :", y_original[end])
  #println("Last observation (rescaled) estimation WO Out         is      :", y_original_hat[end])
  #println("Last observation (rescaled) estimation WO Out recalculated is :", β0_hat[1] + x_original[end,:]' * m_beta_Par[end,:])
  #println("Last observation (rescaled) estimation Wt Out         is      :", y_original_hat_woutl[end])
  #println("Last observation (rescaled) estimation Wt Out recalculated is :", β0_hato[1] + x_original[end,:]' * m_beta_Paro[end,:])
  ## Last Observation Check
    
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

  # Incorporate the intercept in the solution, if it is assumed.
  if (force_intercept)
    m_beta_Par = hcat(fill(mean_x[1], n),m_beta_Par) 
    v_beta_Par = vec(m_beta_Par)
    m_beta_Paro = hcat(fill(mean_x[1], n),m_beta_Paro) 
    v_beta_Paro = vec(m_beta_Paro)
  end
  
  return (β_np_1 = v_beta_Par , β_n_p = m_beta_Par,
          βo_np_1 = v_beta_Paro , βo_n_p = m_beta_Paro,
          β0 = β0_hat, β0o = β0_hato, 
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
