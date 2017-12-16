#########################overall claim#######################################
# Combined-quantitative-models
The Narrow channel break out and shape V strategies combined in an integration. 
This public repository does not contain anything on back testing and simulation, for I cannot put the result and all optimizations publicly!
So if you use this strategy to your investment, you should take all the responsibilities!

Also the API setting files and data I used are not included in this repository.

This repository contains:

1. strategy files
2. run scripts       x
3. strategy summary      x
4. flow chart
5. presentation        x
6. setting files        x

It is not the final one, especially for strategy summary.


##########################strategy file####################################

NarC_ShpV_v4.py: only Narrow channel strategy，temporary test file

NarC_ShpV_v5.py: only ShapeV strategy， temporary test file
NarC_ShpV_v6.py: both Narrow channel and ShapeV strategy, clean file with debugging code removed. 

self.narChaBreak.run(time, last, high, low, vol,ThrReady) Replacing ThrReady with false will switch off narrow channel strategy
self.shapeV.run(time, last, high, low, vol, ThrReady) Replacing ThrReady with false will switch off ShapeV strategy

###########################updated#######################################
