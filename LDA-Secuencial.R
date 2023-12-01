huge2b <- read.csv("LDA_ejemplo.csv")

docs <- strsplit(huge2b$text, split=" " , perl=T)

## PARAMETERS
K <- 16 
alpha <- 0.1 
eta <- 0.001 
iterations <- 20000 

#llikelihood
loglike <- array(0,iterations)

#alpha#
alpha2 <- rep(alpha,K)
metsd_alpha <- rep(2,K)
folder <- max(5,round(iterations/80))
mixpalpha <- array(0,K)
mixpalpha2 <- array(0,c(iterations,K))
alphag <- array(0,c(iterations,K))

#Perplexity
vocab <- unique(unlist(docs))
for(i in 1:length(docs)) docs[[i]] <- match(docs[[i]], vocab)
docsall <- docs

huge2b$id_sample <- seq(1:nrow(huge2b)) 
datos_seleccionados <- huge2b %>% group_by(household_key) %>%slice_head(n = 21) %>% ungroup()
sample <- datos_seleccionados$id_sample
docs <- docsall[sample]
docs2 <- docsall[-sample]

nsim <- 100
totperplexity <- array(0,iterations)

#División de los datos
y <- huge2b$nextpurchase[sample]
y2 <- huge2b$nextpurchase[-sample]
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

phi <- array(0, dim=c(K,length(vocab),iterations))
theta <- array(0, dim=c(length(docs),K,iterations))
alphag <- array(0,c(iterations,K)) 
dimnames(phi)[[2]] <- vocab

for(i in 1:iterations){ 
  loglike[i] <- 0

  phi[,,i] <- (wt + eta) / (rowSums(wt+eta))
  for(d in 1:length(docs)){
    for(w in 1:length(docs[[d]])){ 
      
      t0 <- ta[[d]][w] 
      wid <- docs[[d]][w] 
      
      dt[d,t0] <- dt[d,t0]-1 
      wt[t0,wid] <- wt[t0,wid]-1 
      denom_a <- sum(dt[d,]) + sum(alpha2) 
      denom_b <- rowSums(wt) + length(vocab) * eta 
      p_z <- (wt[,wid] + eta) / denom_b * (dt[d,] + alpha2) / denom_a 
      t1 <- sample(1:K, 1, prob=p_z/sum(p_z)) 
      ta[[d]][w] <- t1 
      dt[d,t1] <- dt[d,t1]+1 
      wt[t1,wid] <- wt[t1,wid]+1 
      loglike[i] <- loglike[i] + log(phi[t1,wid,i])
    }
  }
  
  train_data <- data.frame(dt)
  train_data$y <- y
  modelo <- lm(y ~ ., data = train_data)
  y1hat[, i] <- predict(modelo, newdata = train_data[, -length(train_data)])
  
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
  
  
  alphag[i,] <- alpha2
  
  theta[,,i] <- (dt+alpha2) / rowSums(dt+alpha2) 
  phi[,,i] <- (wt + eta) / (rowSums(wt+eta)) 
  
  ####PERPLEXITY###
  thetapp <- alpha2/sum(alpha2) 
  totperplexity[i] <- 0
  for(d in 1:length(docs2)){ 
    samp <- array(0,dim=c(length(docs2[[d]]),nsim))
    for(w in 1:length(docs2[[d]])){ 
      wid_p <- docs2[[d]][w]
      
      prob_p <- thetapp*phi[,wid_p,i]
      totperplexity[i] <- totperplexity[i] + log(sum(prob_p))
      prob_p <- prob_p/sum(prob_p)
      samp[w,] <- sample(1:K, nsim, replace = TRUE, prob=prob_p)
    }
    dt2 <- matrix(0, nsim, K)
    for(j in 1:nsim){ 
      for(t in 1:K){
        dt2[j,t] <- sum(samp[,j]==t) 
      }
    }
    
    dt4 <- data.frame(dt2)
    y2hat[d,i] <- mean(predict(modelo, newdata = dt4))
  }
}
