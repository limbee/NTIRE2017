#th main.lua -batchSize 32 -nResBlock 60 -save ex1 2>&1 | tee -a ../experiment/ex1.log
#th main.lua -batchSize 32 -nResBlock 30 -nFeat 128 -nThreads 7 -save ex2 2>&1 | tee -a ../experiement/ex2.log
th main.lua -load ex2 -nThreads 4 2>&1 | tee -a ../experiment/ex2.log
