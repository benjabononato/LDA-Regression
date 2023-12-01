starts <- 50
var_betas <- data.frame(array(0, dim = c(starts, 2)))
iterations_star <- 100

for (qq in 1:starts) {
  huge2b <- read.csv("LDA_ejemplo.csv")
  docs <- strsplit(huge2b$text, split=" " , perl=T)
  K <- 16 
  alpha <- 0.1 
  eta <- 0.001 
  iterations <- 20000
  
  #loglikelihood
  loglike_LDA <- array(0,iterations)
  loglike_RL <- array(0,iterations)
  
  #alpha
  alpha2 <- rep(alpha,K)
  metsd_alpha <- rep(2,K)
  folder <- max(5,round(iterations/80))
  mixpalpha <- array(0,K)
  mixpalpha2 <- array(0,c(iterations,K))
  cmixpalpha <- array(0,c(iterations,K))
  alphag <- array(0,c(iterations,K))
  
  #Perplexity
  vocab <- unique(unlist(docs))
  for(i in 1:length(docs)) docs[[i]] <- match(docs[[i]], vocab)
  docsall <- docs
  
  #División de los datos
  huge2b$id_sample <- seq(1:nrow(huge2b)) 
  datos_seleccionados <- huge2b %>% group_by(household_key) %>%slice_head(n = 21) %>% ungroup()
  sample <- datos_seleccionados$id_sample
  docs <- docsall[sample]
  docs2 <- docsall[-sample]
  
  #Guardamos las variables jerárquicas
  nombres <- c(15, 19, 24, 26, 29, 36, 37, 43, 45, 46, 47, 49, 52)
  jer_train <- huge2b[sample, nombres]
  jer_test <- huge2b[-sample, nombres]

  nsim <- 100
  totperplexity_LDA <- array(0,iterations) 
  totperplexity_RL <- array(0,iterations) 
  
  #Regresión
  y <- huge2b$nextpurchase[sample]
  y2 <- huge2b$nextpurchase[-sample]
  
  a0 <- 3/2
  b0 <- a0*1
  lambda0 <- diag(K + 1 + length(nombres))*1/100
  beta <- t(as.matrix(rep(1.1,K + 1 + length(nombres))))
  mu0 <- 0*beta
  
  y2hat <- array(0, dim=c(length(y2),iterations)) 
  y1hat <- array(0, dim=c(length(y),iterations)) 
  
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
  aux <- rbind(rep(1,length(docs)),t(dt), t(jer_train))
  yhat <- beta%*%aux
  sigmares <- 1
  res <- y-yhat
  
  phi <- array(0, dim=c(K,length(vocab),iterations))
  theta <- array(0, dim=c(length(docs),K,iterations))
  alphag <- array(0,c(iterations,K))
  colnames(alphag) <- colnames(alphag, do.NULL = FALSE, prefix = "alpha")
  betag <- array(0,c(iterations,K + 1 + length(jer_train)))
  sigmaresg <- array(0,c(iterations,1))
  
  dimnames(phi)[[2]] <- vocab
  
  for(i in 1:iterations_star){ 
    
    loglike_LDA[i] <- 0
    loglike_RL[i] <- 0
    phi[,,i] <- (wt + eta) / (rowSums(wt+eta))
    for(d in 1:length(docs)){
      for(w in 1:length(docs[[d]])){ 
        
        t0 <- ta[[d]][w] 
        wid <- docs[[d]][w] 
        
        dt[d,t0] <- dt[d,t0]-1 
        wt[t0,wid] <- wt[t0,wid]-1 
        denom_a <- sum(dt[d,]) + sum(alpha2) 
        denom_b <- rowSums(wt) + length(vocab) * eta 
        
        #######REGRESIÓN#######
        aux1 <- c(1,dt[d,], unname(unlist(jer_train[d, ])))
        y1hat[d,i] <- beta%*%aux1
        res1 <- rep(y[d]-y1hat[d,i],K)-beta[2:(K+1)]
        
        #Densidad de residuos
        py <- dnorm(res1[1:K], mean = 0, sd = sigmares, log = TRUE)
        py <-py-max(py)
        py[py< -700] <-  -700
        py <- exp(py)
        
        p_z <- (wt[,wid] + eta) / denom_b * (dt[d,] + alpha2) / denom_a  
        p_z <- p_z*py                                              
        
        t1 <- sample(1:K, 1, prob=p_z/sum(p_z)) 
        ta[[d]][w] <- t1 
        dt[d,t1] <- dt[d,t1]+1 
        wt[t1,wid] <- wt[t1,wid]+1 
        aux1 <- c(1,dt[d,], unname(unlist(jer_train[d, ])))
        y1hat[d,i] <- beta%*%aux1
        loglike_LDA[i] <- loglike_LDA[i] + log(phi[[t1,wid,i]])
        loglike_RL[i] <- loglike_RL[i] + dnorm(res1[t1], mean = 0, sd = sigmares, log = TRUE)
      }
    }
    
    
    #ACTUALIZACIÓN DE ALPHA
    for (k in 1:K){
      alphanew <- alpha2
      alphanew[k] <- exp(log(alpha2[k]) + rnorm(1,0,metsd_alpha[k]))
      
      #test de alpha
      llalphanew <- 0
      llalpha <- 0
      for (r in 1:length(docs)){
        temp <- lgamma(sum(alphanew))-lgamma(sum(alphanew+dt[r,]))+lgamma(alphanew[k]+dt[r,k])-lgamma(alphanew[k])
        llalphanew <- temp + llalphanew
        temp2 <- lgamma(sum(alpha2))-lgamma(sum(alpha2+dt[r,]))+lgamma(alpha2[k]+dt[r,k])-lgamma(alpha2[k])
        llalpha <- temp2 + llalpha
      }
      
      testalpha <- llalphanew-llalpha
      
      accept <- log(runif(1,0,1))<testalpha 
      alpha2[k] <- alpha2[k]+accept*(alphanew[k]-alpha2[k])
      mixpalpha[k] <- mixpalpha[k] + accept*1 
      mixpalpha2[i,k] <- accept*1 
    }
    
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
    
    dt3 <- as.matrix(cbind(rep(1,length(docs)), dt, jer_train))
    lambdani <- solve(t(dt3)%*%dt3+lambda0) 
    mun <- lambdani%*%(lambda0%*%t(mu0)+t(dt3)%*%y)
    beta <- MASS::mvrnorm(1, mu = mun, Sigma = lambdani*sigmares*sigmares)
    
    an <- a0+length(docs)/2
    bn <- b0+1/2*(t(y)%*%y+mu0%*%lambda0%*%t(mu0)-t(mun)%*%solve(lambdani)%*%mun) 
    sigmares <- sqrt(1/rgamma(1,an,bn))
    yhat <- beta%*%t(dt3) 
    
    betag[i,] <- beta
    sigmaresg[i,] <- sigmares
    alphag[i,] <- alpha2 
    theta[,,i] <- (dt+alpha2) / rowSums(dt+alpha2) 
    phi[,,i] <- (wt + eta) / (rowSums(wt+eta))
    
    ####PERPLEXITY###
    thetapp <- alpha2/sum(alpha2)
    totperplexity_LDA[i] <- 0
    totperplexity_RL[i] <- 0
    for(d in 1:length(docs2)){ 
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
      dt4 <- cbind(rep(1,nsim), dt2)
      for (jt in 1:ncol(jer_test)) {
        dt4 <- cbind(dt4, jer_test[d, jt]) 
      }
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
cargar <- var_betas[1, 2]
load(cargar)
inicio <- i + 1

for(i in inicio:iterations){
  
  loglike_LDA[i] <- 0
  loglike_RL[i] <- 0
  phi[,,i] <- (wt + eta) / (rowSums(wt+eta))
  for(d in 1:length(docs)){
    for(w in 1:length(docs[[d]])){ 
      
      t0 <- ta[[d]][w] 
      wid <- docs[[d]][w] 
      
      dt[d,t0] <- dt[d,t0]-1 
      wt[t0,wid] <- wt[t0,wid]-1 
      denom_a <- sum(dt[d,]) + sum(alpha2) 
      denom_b <- rowSums(wt) + length(vocab) * eta 
      
      #######REGRESIÓN#######
      aux1 <- c(1,dt[d,], unname(unlist(jer_train[d, ])))
      y1hat[d,i] <- beta%*%aux1
      res1 <- rep(y[d]-y1hat[d,i],K)-beta[2:(K+1)]
      
      #Densidad de residuos
      py <- dnorm(res1[1:K], mean = 0, sd = sigmares, log = TRUE)
      py <-py-max(py)
      py[py< -700] <-  -700
      py <- exp(py)
      
      p_z <- (wt[,wid] + eta) / denom_b * (dt[d,] + alpha2) / denom_a  
      p_z <- p_z*py                                              
      
      t1 <- sample(1:K, 1, prob=p_z/sum(p_z)) 
      ta[[d]][w] <- t1 
      dt[d,t1] <- dt[d,t1]+1 
      wt[t1,wid] <- wt[t1,wid]+1 
      aux1 <- c(1,dt[d,], unname(unlist(jer_train[d, ])))
      y1hat[d,i] <- beta%*%aux1
      loglike_LDA[i] <- loglike_LDA[i] + log(phi[[t1,wid,i]])
      loglike_RL[i] <- loglike_RL[i] + dnorm(res1[t1], mean = 0, sd = sigmares, log = TRUE)
    }
  }
  
  
  #ACTUALIZACIÓN DE ALPHA
  for (k in 1:K){
    alphanew <- alpha2
    alphanew[k] <- exp(log(alpha2[k]) + rnorm(1,0,metsd_alpha[k]))
    
    #test de alpha
    llalphanew <- 0
    llalpha <- 0
    for (r in 1:length(docs)){
      temp <- lgamma(sum(alphanew))-lgamma(sum(alphanew+dt[r,]))+lgamma(alphanew[k]+dt[r,k])-lgamma(alphanew[k])
      llalphanew <- temp + llalphanew
      temp2 <- lgamma(sum(alpha2))-lgamma(sum(alpha2+dt[r,]))+lgamma(alpha2[k]+dt[r,k])-lgamma(alpha2[k])
      llalpha <- temp2 + llalpha
    }
    
    testalpha <- llalphanew-llalpha
    
    accept <- log(runif(1,0,1))<testalpha 
    alpha2[k] <- alpha2[k]+accept*(alphanew[k]-alpha2[k])
    mixpalpha[k] <- mixpalpha[k] + accept*1 
    mixpalpha2[i,k] <- accept*1 
  }
  
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
  
  dt3 <- as.matrix(cbind(rep(1,length(docs)), dt, jer_train))
  lambdani <- solve(t(dt3)%*%dt3+lambda0) 
  mun <- lambdani%*%(lambda0%*%t(mu0)+t(dt3)%*%y)
  beta <- MASS::mvrnorm(1, mu = mun, Sigma = lambdani*sigmares*sigmares)
  
  an <- a0+length(docs)/2
  bn <- b0+1/2*(t(y)%*%y+mu0%*%lambda0%*%t(mu0)-t(mun)%*%solve(lambdani)%*%mun) 
  sigmares <- sqrt(1/rgamma(1,an,bn))
  yhat <- beta%*%t(dt3) 
  
  betag[i,] <- beta
  sigmaresg[i,] <- sigmares
  alphag[i,] <- alpha2 
  theta[,,i] <- (dt+alpha2) / rowSums(dt+alpha2) 
  phi[,,i] <- (wt + eta) / (rowSums(wt+eta))
  
  ####PERPLEXITY###
  thetapp <- alpha2/sum(alpha2)
  totperplexity_LDA[i] <- 0
  totperplexity_RL[i] <- 0
  for(d in 1:length(docs2)){ 
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
    dt4 <- cbind(rep(1,nsim), dt2)
    for (jt in 1:ncol(jer_test)) {
      dt4 <- cbind(dt4, jer_test[d, jt]) 
    }
    y2hat[d,i] <- mean(rowSums(dt4%*%as.matrix(betag[i,])))
    samp2 <- mean(dnorm(y2[d] - rowSums(dt4%*%as.matrix(betag[i,])),mean = 0, sd = sigmaresg[i,]))
    totperplexity_RL[i] <- totperplexity_RL[i] + log(samp2)
  }
}

