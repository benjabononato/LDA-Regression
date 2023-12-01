starts <- 50 #Número de inicios para selección del mejor modelo
var_betas <- data.frame(array(0, dim = c(starts, 2)))
iterations_star <- 100

for (qq in 1:starts) {
  huge2b <- read.csv("LDA_ejemplo.csv") #Carga de datos
  docs <- strsplit(huge2b$text, split=" " , perl=T)
  
  ## PARAMETERS
  K <- 16 #Motivaciones
  alpha <- 0.1 
  eta <- 0.001 
  iterations <- 20000
  
  #loglikelihood
  loglike_LDA <- array(0,iterations)
  loglike_RL <- array(0,iterations)
  loglike_Gamma <- array(0, c(iterations, K))
  
  #Alpha
  metsd_alpha <- rep(0.01,K)
  folder <- max(5,round(iterations/80))
  mixpalpha <- array(0,K)
  mixpalpha2 <- array(0,c(iterations,K))
  cmixpalpha <- array(0,c(iterations,K))
  alphag <- array(0,c(iterations,K))
  
  #Perplexity
  totperplexity_LDA <- array(0,iterations) 
  totperplexity_RL <- array(0,iterations) 

  #División de los datos y preparación de los documentos
  vocab <- unique(unlist(docs))
  for(i in 1:length(docs)) docs[[i]] <- match(docs[[i]], vocab)
  docsall <- docs
  huge2b$id_sample <- seq(1:nrow(huge2b)) 
  datos_seleccionados <- huge2b %>% group_by(household_key) %>%slice_head(n = 21) %>% ungroup()
  sample <- datos_seleccionados$id_sample

  #Se agrega alpha por cada boleta
  cola_inicial <- ncol(huge2b)
  cola_final <- ncol(huge2b) + 1

  huge2b <- cbind(huge2b, array(alpha, dim = c(nrow(huge2b), K)))

  docs <- docsall[sample]
  docs2 <- docsall[-sample]
  
  hugeb_train <- data.frame(huge2b[sample,  ])
  name_num <- 1:nrow(hugeb_train)
  rownames(hugeb_train) <- name_num
  hugeb_test <- huge2b[-sample, ]

  
  nsim <- 100 #Número de simulación que tendrá la asignación de motivaciones para el conjunto de test
  
  #Regresión y priors
  y <- (huge2b$nextpurchase[sample])
  y2 <- (huge2b$nextpurchase[-sample])
  
  a0 <- 3/2
  b0 <- a0*1
  lambda0 <- diag(K+1)*1/100
  beta <- t(as.matrix(rep(1.1,K+1)))
  mu0 <- 0*beta
  
  y2hat <- array(0, dim=c(length(y2),iterations)) 
  y1hat <- array(0, dim=c(length(y),iterations)) 
  
  #Regresión Alpha
  predictores_alpha <- c(15, 19, 24, 26, 29, 36, 37, 43, 45, 46, 47, 49, 52) #Covariables
  gamma_demo <- array(0, c(length(predictores_alpha)+1, K))
  lambda0_demo <- diag(ncol(hugeb_train[, 1:(length(predictores_alpha)+1)]))*1/100
  mu0_demo <- matrix(data = 0, nrow = 1, ncol(hugeb_train[, 1:(length(predictores_alpha)+1)]))
  
  ##Asignación de tópicos a cada palabra, creación de matriz Wt y Dt
  wt <- matrix(0, K, length(vocab)) 
  ta <- sapply(docs, function(x) rep(0, length(x))) 
  for(d in 1:length(docs)){ 
    for(w in 1:length(docs[[d]])){ 
      ta[[d]][w] <- sample(1:K, 1) 
      ti <- ta[[d]][w] 
      wi <- docs[[d]][w] 
      wt[ti,wi] <- wt[ti,wi]+1      
    }
  }
  
  dt <- matrix(0, length(docs), K)
  for(d in 1:length(docs)){
    for(t in 1:K){ 
      dt[d,t] <- sum(ta[[d]]==t)    
    }
  }
  aux <- rbind(rep(1,length(docs)),t(dt))
  yhat <- beta%*%aux
  sigmares <- 1
  res <- y-yhat
  
  phi <- array(0, dim=c(K,length(vocab),iterations))
  theta <- array(0, dim=c(length(docs),K,iterations))
  betag <- array(0,c(iterations,K+1))
  sigmaresg <- array(0,c(iterations,1))
  sigmaresg_demo <- array(0,c(iterations,1))
  
  alphag_household <- array(0, dim=c(iterations, K, length(unique(hugeb_train$household_key))))
  gamma_reg <- array(0, dim=c((length(predictores_alpha)+1),K,iterations))
  rownames(gamma_demo)[2:nrow(gamma_demo)] <- colnames(hugeb_train[, predictores_alpha])
  
  dimnames(phi)[[2]] <- vocab
  
  for(i in 1:iterations_star){ 
    
    loglike_LDA[i] <- 0
    loglike_RL[i] <- 0
    phi[,,i] <- (wt + eta) / (rowSums(wt+eta))
    for(d in 1:length(docs)){ 
      alpha2 <- unname(unlist(hugeb_train[d, cola_final:ncol(hugeb_train)]))
      for(w in 1:length(docs[[d]])){  
        
        t0 <- ta[[d]][w] 
        wid <- docs[[d]][w] 
        
        dt[d,t0] <- dt[d,t0]-1 
        wt[t0,wid] <- wt[t0,wid]-1 
        denom_a <- sum(dt[d,]) + sum(alpha2) 
        denom_b <- rowSums(wt) + length(vocab) * eta 
        
        #REGRESIÓN
        aux1 <- c(1,dt[d,])
        y1hat[d,i] <- beta%*%aux1
        res1 <- y[d]-y1hat[d,i]
        res1_error <- rep(res1, K) - beta[2:(K+1)]
        #Densidad de residuos
        py <- dnorm(res1_error, mean = 0, sd = sigmares, log = TRUE)
        py <-py-max(py)
        py[py< -700] <-  -700
        py <- exp(py)
        
        p_z <- (wt[,wid] + eta) / denom_b * (dt[d,] + alpha2) / denom_a
        
        p_z <- p_z * py #Se combina lda con la regresión
        
        t1 <- sample(1:K, 1, prob=p_z/sum(p_z)) 
        ta[[d]][w] <- t1 
        dt[d,t1] <- dt[d,t1]+1 
        aux1 <- c(1,dt[d,])
        y1hat[d,i] <- beta%*%aux1
        wt[t1,wid] <- wt[t1,wid]+1 
        loglike_LDA[i] <- loglike_LDA[i] + log(phi[[t1,wid,i]])
        loglike_RL[i] <- loglike_RL[i] + dnorm(res1, mean = 0, sd = sigmares, log = TRUE)
      }
      theta[d,,i] <- (dt[d,]+alpha2) / sum(dt[d,]+alpha2)
    }
    
    #REGRESIÓN DE ALPHA
    propuesta_alpha <- array(0, c(nrow(hugeb_train), K))
    demo_x <- as.matrix(hugeb_train[, predictores_alpha])
    demo_x <- cbind(rep(1,length(docs)),demo_x)
    
    for(p1 in 1:K){
      demo_y <- as.matrix(hugeb_train[, cola_inicial+p1])
      lambdani_demo <- solve(t(demo_x)%*%demo_x+lambda0_demo) 
      mun_demo <- lambdani_demo%*%(lambda0_demo%*%t(mu0_demo)+t(demo_x)%*%demo_y)
      an <- a0+length(docs)/2
      bn_demo <- b0+1/2*(t(demo_y)%*%demo_y+mu0_demo%*%lambda0_demo%*%t(mu0_demo)-t(mun_demo)%*%solve(lambdani_demo)%*%mun_demo) 
      sigmares_demo <- sqrt(1/rgamma(1,an,bn_demo))
      gamma_demo <- MASS::mvrnorm(1, mu = mun_demo, Sigma = lambdani_demo*sigmares_demo*sigmares_demo)
      propuesta_alpha[, p1] <- t(t(gamma_demo)%*%t(demo_x))
      
      #Loglikelihood gamma
      for (p2 in 1:length(propuesta_alpha[, p1])) {
        res_gamma <- demo_y[p2] - propuesta_alpha[p2, p1]
        loglike_Gamma[i, p1] <- loglike_Gamma[i, p1] + dnorm(res_gamma, mean = 0, sd = sigmares_demo, log = TRUE) 
      }
    }
    
    propuesta_alpha <- data.frame(unique(cbind(propuesta_alpha, hugeb_train$household_key)))
    colnames(propuesta_alpha)[ncol(propuesta_alpha)] <- 'household_key'
    propuesta_alpha <- replace(propuesta_alpha, propuesta_alpha < 0, 0.01)
    sigmaresg_demo[i, ] <- sigmares_demo
    gamma_reg[, , i] <- gamma_demo
    
    #VALIDACIÓN DE ALPHA
    for(q1 in 1:nrow(propuesta_alpha)){
      alpha2 <- unname(unlist(propuesta_alpha[q1, 1:K]))
      household_key_mh <- propuesta_alpha[q1, ncol(propuesta_alpha)][[1]]
      hugeb_index_household <- rownames(hugeb_train[hugeb_train$household_key ==household_key_mh, ])
      dt_mh <- dt[as.integer(hugeb_index_household), ]
      for (k in 1:K){
        alphanew <- alpha2
        alphanew[k] <- alpha2[k] + rnorm(1,0,metsd_alpha[k])
        alphanew[k] <- replace(alphanew[k], alphanew[k] < 0, 0.01)
        
        llalphanew <- 0
        llalpha <- 0
        for (r in 1:nrow(dt_mh)){
          temp <- lgamma(sum(alphanew))-lgamma(sum(alphanew+dt_mh[r,]))+lgamma(alphanew[k]+dt_mh[r,k])-lgamma(alphanew[k])
          llalphanew <- temp + llalphanew
          temp2 <- lgamma(sum(alpha2))-lgamma(sum(alpha2+dt_mh[r,]))+lgamma(alpha2[k]+dt_mh[r,k])-lgamma(alpha2[k])
          llalpha <- temp2 + llalpha
        }
        
        testalpha <- llalphanew-llalpha
        
        accept <- log(runif(1,0,1))<testalpha 
        alpha2[k] <- alpha2[k]+accept*(alphanew[k]-alpha2[k])
        mixpalpha[k] <- mixpalpha[k] + accept*1 
        mixpalpha2[i,k] <- accept*1 
      }
      propuesta_alpha[q1, 1:K] <- alpha2
      alphag_household[i, , q1] <- unname(unlist(propuesta_alpha[q1, 1:K]))
    }
    propuesta_alpha <- replace(propuesta_alpha, propuesta_alpha < 0, 0.01)
    hugeb_train[, cola_final:ncol(hugeb_train)] <- merge(hugeb_train, propuesta_alpha, by = "household_key")[(ncol(hugeb_train)+1):(ncol(hugeb_train)+K)]
    
    #AJUSTE SALTO ALPHA
    if (i<iterations/2&i>(folder-1)&(i/folder)%%1==0){
      ratio2 <- colSums(mixpalpha2[(i-folder+1):i,])/folder
      for (k in 1:K){
        if (ratio2[k]>.5){
          metsd_alpha[k] <- metsd_alpha[k]*1.05
          #metsd_alpha2[i/folder] <- metsd_alpha
        }
        else if (ratio2[k]<.35)
        {     metsd_alpha[k] <- metsd_alpha[k]*0.95}
        #metsd_alpha2[i/folder] <- metsd_alpha
      }
    }
    
    dt3 <- cbind(rep(1,length(docs)),dt)
    lambdani <- solve(t(dt3)%*%dt3+lambda0) 
    mun <- lambdani%*%(lambda0%*%t(mu0)+t(dt3)%*%y)
    beta <- MASS::mvrnorm(1, mu = mun, Sigma = lambdani*sigmares*sigmares)
    
    an <- a0+length(docs)/2
    bn <- b0+1/2*(t(y)%*%y+mu0%*%lambda0%*%t(mu0)-t(mun)%*%solve(lambdani)%*%mun) 
    sigmares <- sqrt(1/rgamma(1,an,bn))
    yhat <- beta%*%t(dt3) 
    
    betag[i,] <- beta
    sigmaresg[i,] <- sigmares
    phi[,,i] <- (wt + eta) / (rowSums(wt+eta))
    
    ####PERPLEXITY###
    hugeb_test[, cola_final:ncol(hugeb_train)] <- merge(hugeb_test, propuesta_alpha, by = 'household_key')[(ncol(hugeb_train)+1):(ncol(hugeb_train)+K)]
    totperplexity_LDA[i] <- 0
    totperplexity_RL[i] <- 0
    for(d in 1:length(docs2)){ 
      alpha2 <- unname(unlist(hugeb_test[d, cola_final:ncol(hugeb_train)]))
      thetapp <- alpha2/sum(alpha2)
      samp <- array(0,dim=c(length(docs2[[d]]),nsim))
      for(w in 1:length(docs2[[d]])){ 
        wid_p <- docs2[[d]][w]
        
        prob_p <- thetapp*phi[,wid_p,i]
        totperplexity_LDA[i] <- totperplexity_LDA[i] + log(sum(prob_p))
        prob_p <- prob_p/sum(prob_p)
        samp[w,] <- sample(1:K, nsim, replace = TRUE, prob=prob_p)
      }
      dt2 <- matrix(0, nsim, K)
      for(j in 1:nsim){ 
        for(t in 1:K){ 
          dt2[j,t] <- sum(samp[,j]==t)
        }
      }
      dt4 <- cbind(rep(1,nsim),dt2)
      y2hat[d,i] <- mean(rowSums(dt4%*%as.matrix(betag[i,])))
      samp2 <- mean(dnorm(y2[d] - rowSums(dt4%*%as.matrix(betag[i,])),mean = 0, sd = sigmaresg[i,]))
      totperplexity_RL[i] <- totperplexity_RL[i] + log(samp2)
    }
  }
  
  nombre_star <- paste("Comienzo", qq, ".RData")
  var_betas[qq, 1] <- var(beta[2:(K+1)])
  var_betas[qq, 2] <- nombre_star
  save.image(file = nombre_star)
}


var_betas <- var_betas[order(var_betas$X1), ] 
cargar <- var_betas[1, 2] #Se selecciona la opción con menor varianza en los parámetros beta
load(cargar)
inicio <- i + 1

for(i in inicio:iterations){ #Se continua desde la última iteración de la pre selección
  
  loglike_LDA[i] <- 0
  loglike_RL[i] <- 0
  phi[,,i] <- (wt + eta) / (rowSums(wt+eta))
  for(d in 1:length(docs)){ 
    alpha2 <- unname(unlist(hugeb_train[d, cola_final:ncol(hugeb_train)]))
    for(w in 1:length(docs[[d]])){ 
      
      t0 <- ta[[d]][w] 
      wid <- docs[[d]][w] 
      
      dt[d,t0] <- dt[d,t0]-1 
      wt[t0,wid] <- wt[t0,wid]-1 
      denom_a <- sum(dt[d,]) + sum(alpha2) 
      denom_b <- rowSums(wt) + length(vocab) * eta 
      
      #REGRESIÓN
      aux1 <- c(1,dt[d,])
      y1hat[d,i] <- beta%*%aux1
      res1 <- y[d]-y1hat[d,i]
      res1_error <- rep(res1, K) - beta[2:(K+1)]
      #Densidad de residuos
      py <- dnorm(res1_error, mean = 0, sd = sigmares, log = TRUE)
      py <-py-max(py)
      py[py< -700] <-  -700
      py <- exp(py)
      #######################
      p_z <- (wt[,wid] + eta) / denom_b * (dt[d,] + alpha2) / denom_a
      
      p_z <- p_z * py
      
      t1 <- sample(1:K, 1, prob=p_z/sum(p_z))
      ta[[d]][w] <- t1 
      dt[d,t1] <- dt[d,t1]+1 
      aux1 <- c(1,dt[d,])
      y1hat[d,i] <- beta%*%aux1
      wt[t1,wid] <- wt[t1,wid]+1 
      loglike_LDA[i] <- loglike_LDA[i] + log(phi[[t1,wid,i]])
      loglike_RL[i] <- loglike_RL[i] + dnorm(res1, mean = 0, sd = sigmares, log = TRUE)
    }
    theta[d,,i] <- (dt[d,]+alpha2) / sum(dt[d,]+alpha2)
  }
  
  #REGRESIÓN DE ALPHA
  propuesta_alpha <- array(0, c(nrow(hugeb_train), K))
  demo_x <- as.matrix(hugeb_train[, predictores_alpha])
  demo_x <- cbind(rep(1,length(docs)),demo_x)
  
  for(p1 in 1:K){
    demo_y <- as.matrix(hugeb_train[, cola_inicial+p1])
    lambdani_demo <- solve(t(demo_x)%*%demo_x+lambda0_demo) 
    mun_demo <- lambdani_demo%*%(lambda0_demo%*%t(mu0_demo)+t(demo_x)%*%demo_y)
    an <- a0+length(docs)/2
    bn_demo <- b0+1/2*(t(demo_y)%*%demo_y+mu0_demo%*%lambda0_demo%*%t(mu0_demo)-t(mun_demo)%*%solve(lambdani_demo)%*%mun_demo) 
    sigmares_demo <- sqrt(1/rgamma(1,an,bn_demo))
    gamma_demo <- MASS::mvrnorm(1, mu = mun_demo, Sigma = lambdani_demo*sigmares_demo*sigmares_demo)
    propuesta_alpha[, p1] <- t(t(gamma_demo)%*%t(demo_x))
    
    #Loglikelihood gamma
    for (p2 in 1:length(propuesta_alpha[, p1])) {
      res_gamma <- demo_y[p2] - propuesta_alpha[p2, p1]
      loglike_Gamma[i, p1] <- loglike_Gamma[i, p1] + dnorm(res_gamma, mean = 0, sd = sigmares_demo, log = TRUE) 
    }
  }
  
  propuesta_alpha <- data.frame(unique(cbind(propuesta_alpha, hugeb_train$household_key)))
  colnames(propuesta_alpha)[ncol(propuesta_alpha)] <- 'household_key'
  propuesta_alpha <- replace(propuesta_alpha, propuesta_alpha < 0, 0.01)
  sigmaresg_demo[i, ] <- sigmares_demo
  gamma_reg[, , i] <- gamma_demo
  
  #VALIDACIÓN DE ALPHA
  for(q1 in 1:nrow(propuesta_alpha)){
    alpha2 <- unname(unlist(propuesta_alpha[q1, 1:K]))
    household_key_mh <- propuesta_alpha[q1, ncol(propuesta_alpha)][[1]]
    hugeb_index_household <- rownames(hugeb_train[hugeb_train$household_key ==household_key_mh, ])
    dt_mh <- dt[as.integer(hugeb_index_household), ]
    for (k in 1:K){
      alphanew <- alpha2
      alphanew[k] <- alpha2[k] + rnorm(1,0,metsd_alpha[k])
      alphanew[k] <- replace(alphanew[k], alphanew[k] < 0, 0.01)
      
      llalphanew <- 0
      llalpha <- 0
      for (r in 1:nrow(dt_mh)){
        temp <- lgamma(sum(alphanew))-lgamma(sum(alphanew+dt_mh[r,]))+lgamma(alphanew[k]+dt_mh[r,k])-lgamma(alphanew[k])
        llalphanew <- temp + llalphanew
        temp2 <- lgamma(sum(alpha2))-lgamma(sum(alpha2+dt_mh[r,]))+lgamma(alpha2[k]+dt_mh[r,k])-lgamma(alpha2[k])
        llalpha <- temp2 + llalpha
      }
      
      testalpha <- llalphanew-llalpha
      
      accept <- log(runif(1,0,1))<testalpha 
      alpha2[k] <- alpha2[k]+accept*(alphanew[k]-alpha2[k])
      mixpalpha[k] <- mixpalpha[k] + accept*1 
      mixpalpha2[i,k] <- accept*1 
    }
    propuesta_alpha[q1, 1:K] <- alpha2
    alphag_household[i, , q1] <- unname(unlist(propuesta_alpha[q1, 1:K]))
  }
  propuesta_alpha <- replace(propuesta_alpha, propuesta_alpha < 0, 0.01)
  hugeb_train[, cola_final:ncol(hugeb_train)] <- merge(hugeb_train, propuesta_alpha, by = "household_key")[(ncol(hugeb_train)+1):(ncol(hugeb_train)+K)]
  
  #AJUSTE SALTO ALPHA
  if (i<iterations/2&i>(folder-1)&(i/folder)%%1==0){
    ratio2 <- colSums(mixpalpha2[(i-folder+1):i,])/folder
    for (k in 1:K){
      if (ratio2[k]>.5){
        metsd_alpha[k] <- metsd_alpha[k]*1.05
        #metsd_alpha2[i/folder] <- metsd_alpha
      }
      else if (ratio2[k]<.35)
      {     metsd_alpha[k] <- metsd_alpha[k]*0.95}
      #metsd_alpha2[i/folder] <- metsd_alpha
    }
  }
  
  dt3 <- cbind(rep(1,length(docs)),dt)
  lambdani <- solve(t(dt3)%*%dt3+lambda0) 
  mun <- lambdani%*%(lambda0%*%t(mu0)+t(dt3)%*%y)
  beta <- MASS::mvrnorm(1, mu = mun, Sigma = lambdani*sigmares*sigmares)
  
  an <- a0+length(docs)/2
  bn <- b0+1/2*(t(y)%*%y+mu0%*%lambda0%*%t(mu0)-t(mun)%*%solve(lambdani)%*%mun) 
  sigmares <- sqrt(1/rgamma(1,an,bn))
  yhat <- beta%*%t(dt3) 
  
  betag[i,] <- beta
  sigmaresg[i,] <- sigmares
  phi[,,i] <- (wt + eta) / (rowSums(wt+eta))
  
  ####PERPLEXITY###
  hugeb_test[, cola_final:ncol(hugeb_train)] <- merge(hugeb_test, propuesta_alpha, by = 'household_key')[(ncol(hugeb_train)+1):(ncol(hugeb_train)+K)]
  totperplexity_LDA[i] <- 0
  totperplexity_RL[i] <- 0
  for(d in 1:length(docs2)){
    alpha2 <- unname(unlist(hugeb_test[d, cola_final:ncol(hugeb_train)]))
    thetapp <- alpha2/sum(alpha2)
    samp <- array(0,dim=c(length(docs2[[d]]),nsim))
    for(w in 1:length(docs2[[d]])){ 
      wid_p <- docs2[[d]][w]
      
      prob_p <- thetapp*phi[,wid_p,i]
      totperplexity_LDA[i] <- totperplexity_LDA[i] + log(sum(prob_p))
      prob_p <- prob_p/sum(prob_p)
      samp[w,] <- sample(1:K, nsim, replace = TRUE, prob=prob_p)
    }
    dt2 <- matrix(0, nsim, K)
    for(j in 1:nsim){ 
      for(t in 1:K){ 
        dt2[j,t] <- sum(samp[,j]==t)
      }
    }
    dt4 <- cbind(rep(1,nsim),dt2)
    y2hat[d,i] <- mean(rowSums(dt4%*%as.matrix(betag[i,])))
    samp2 <- mean(dnorm(y2[d] - rowSums(dt4%*%as.matrix(betag[i,])),mean = 0, sd = sigmaresg[i,]))
    totperplexity_RL[i] <- totperplexity_RL[i] + log(samp2)
  }
}
