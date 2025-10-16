#
#  IFL: Iterative Fused LASSO Algorithm
#
#  last update: AM 16/10/25
#

using LinearAlgebra
using GLMNet

# Supporting Function
include("supporting_functions.jl")

#
# IFL
#
"""
IFL(y, x; handle_outliers = true, intercept = false, CD = false, verbose = false)

Since IFL considers a dynamic model, data should be standardized OUTSIDE IFL.

CD = false # uses glmnet
CD = true  # uses Coordinate Descent
  
Returns β_np_1 = v_beta_Par , β_n_p = m_beta_Par,                 # Solutions without Outliers
        βo_np_1 = v_beta_Paro , βo_n_p = m_beta_Paro,             # Solutions with Outliers
        β0 = β0_hat, β0_hato = β0_hato,                           # Estimated Intercept
        y_hat =  y_original_hat ,                                 # Insample Estimators without Outliers
        y_hato = y_original_hat_woutl,                            # Insample Estimators with Outliers
        bic_value = bic_value,                                    # BIC value model without outliers
        bic_value_outliers = bic_value_outliers,                  # BIC value model with outliers
        chosen_var = chosen_variables,                            # List of selected columns in X
        chosen_var_bk = chosen_variables_break,                   # List of selected columns with estim breaks in X
        chosen_var_outliers = chosen_variables_outliers,          # List of selected columns in X (model w/outliers)
        chosen_var_outliers_bk = chosen_variables_outliers_break, # List of selected columns with estim breaks in X (model w/outliers)
        RMSE_inS = RMSE_inS, RMSE_inSo = RMSE_inSo,               # In Sample RMSE (models with and without outliers)
        ind_out = ind_out,                                        # list of outlier in y  
        IFL_t0 = IFL_t0                                           # time to converge

Last review: 16 Oct 25

"""
function IFL(y, x; handle_outliers = true, intercept = false, CD = false, verbose = false)
  """
  """ 
  
  # Store originals
  x_original = x
  y_original = y

  #
  mean_x = mean(x_original, dims=1)
  mean_y = mean(y_original)

  xc = x .- mean_x
  yc = y.- mean_y
    
  # Create a DLRM Intercept 
  if (intercept) 
    xc = hcat(ones(size(xc)[1]), xc)
  end
    
  ind_out=[]

  ####################################################################
  # Problem Dimensions
  n = length(y) 
  p = size(xc)[2] # When Intercept is assumed p has already turned to p-1
  ####################################################################
  
  # Setup Iteractive Fused LASSO 
  nbetas_In  = repeat([n], p)
  nbetas_In_Tot =sum(nbetas_In)
  nbetas_Out = repeat([0], p)
  nbetas_Out_Tot = sum(nbetas_Out)

  global XX = Matrix{Float64}(undef, 0, 0)  # Alterei de Int para Float64!!! 27/05/25
  for i in 1:p
      if (size(XX, 1) == 0)
          XX = Diagonal(xc[:, i])  # Initialize with the first diagonal matrix
      else
          XX = hcat(XX, Diagonal(xc[:, i]))  # Horizontally concatenate diagonal matrices
      end
  end

  # Main loop
  IFL_t0 = @elapsed begin 
    
    time_H = @elapsed begin
      Hs = build_H(xc)
    end
    if(verbose)
      println("Step 1, Build H: ", time_H)
    end 
    
    H = Matrix(Hs)         

    # Ada LASSO solution with smallest BIC
    # Intercept is being managed outside glmnet. 
    # Since its a Dynamic Linear Regression, the intercept is time variant.
    # GLMNet consider the intercept only for obs #1, therefore intercept here is always FALSE.
    time_AdaS1 = @elapsed begin
      theta_hat_Est = Ada_LASSO_intercept(H, yc, standardize = false, intercept = false, CD = CD) 
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
          theta_hatP_Est = Ada_LASSO_intercept_outliers(HP, yc, HPo, standardize = false, intercept = false, CD = CD) 
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
        global y_hat  = HP * theta_hatP             # Without Intercept
          
        # No. of Outliers
        ind_out = findall(pts -> pts != 0, theta_hatPo[(end-n+1):end])  
                  
      else
        time_AdaS3woO = @elapsed begin
          theta_hatP_Est = Ada_LASSO_intercept(HP, yc, standardize = false, intercept = false, CD = CD) 
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
  v_beta_Par = M * LdP_i * theta_hatP     
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
  #if (intercept)
  #  mean_x = hcat(1, mean_x) 
  #end
  #β0_hat  = mean_y .- mean_x  * m_beta_Par[end,:]  # valid only for the last intercept.
  #β0_hato = mean_y .- mean_x  * m_beta_Paro[end,:] # valid only for the last intercept. 
  if(intercept) # Estimated Intercept is the first colum of B = m_beta_Par
    β0_hat = m_beta_Par[:,1]  
    β0_hato = m_beta_Paro[:,1]  
    m_beta_Par = m_beta_Par[:,2:end]
    m_beta_Paro = m_beta_Paro[:,2:end]
    p = p-1
    v_beta_Par = reshape(m_beta_Par, n*p, 1)
    v_beta_Paro = reshape(m_beta_Paro, n*p, 1)
  else
    β0_hat  = mean_y .- mean_x  * m_beta_Par[end,:]  # valid only for the last intercept.
    β0_hato = mean_y .- mean_x  * m_beta_Paro[end,:] # valid only for the last intercept. 
  end
      
  # Since data was centralized, it needs to be decentralized back.
  ydc_hat = y_hat .+ mean_y
  if(handle_outliers)
    ydc_hat_wOutl  = y_hato .+ mean_y
  end
    
  # RMSE In Sample
  RMSE_inS  = RMSE(y,ydc_hat) 
  RMSE_inSo = RMSE(y, ydc_hat_wOutl) 
  
  # Selected variables X unselected variables, and variables with break 
  chosen_variables = zeros(p)
  chosen_variables_break =  zeros(p)
  chosen_variables_outliers = zeros(p)
  chosen_variables_outliers_break =  zeros(p)
  
  # p = number of original variables (does not consider intercept) 
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
          β0_hat = β0_hat, β0_hato = β0_hato,
          y_hat =  ydc_hat ,  y_hato = ydc_hat_wOutl,
          bic_value = bic_value, bic_value_outliers = bic_value_outliers,            
          chosen_var = chosen_variables,
          chosen_var_bk = chosen_variables_break,
          chosen_var_outliers = chosen_variables_outliers,
          chosen_var_outliers_bk = chosen_variables_outliers_break,
          RMSE_inS = RMSE_inS, RMSE_inSo = RMSE_inSo,
          ind_out = ind_out, IFL_t0 = IFL_t0)

end
