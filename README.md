# To Run
pip install tensorflow numpy
python driver.py

# Introduction
I wanted to try something I hadn't before with this model and from a course in college and some messing around I had previously done a little reinforcement learning. From that and watching OpenAI crush professional Dota2 players with there deep q reinforcement learning models I wanted to take the time to implement that here and see if we could get something working.

## Files
- agent.py defines the deep q agent and handles training of the agent and storing "matches" in the replay buffer for better training
- battleship_grid.py defines, as you may have guessed, the gamespace and handles the logic for placing ships, checking hits and misses, sinking ships and win state
- driver.py runs the simulation and is responsible for stepping forward in the gamestate and handling rewards and punishments for the agent.


## Results

- For a V1 pirate defense system this model as it stands will sink maybe one or two ships before getting locked in some minima somewhere and trying to fire into the ocean, though there are a few trials where the model won but the randomness was still high. I have a few thoughts as to why that might be the case, one my grid system is a bit adhoc compared to the implementations I saw when I was trying to learn more about implementing both battleship and a model to play it. 
- Additionally in the cases that have seemed to learn to play well for battleship the grid placements were not totally random the placements were created from placements of live human players and analysis of those placements showed that we have a tendency to place ships in specific areas of the grid, contrasting with my solution where the grid placement is totally "random". 
- Third I'm a little uncertain on my reward mechanism and how to properly balance the rewards vs punishments and finally it looks like that the more successful implementations wind up using visual grids with CNNs for the action selection. 
- If I was going to continue this I would reimplement this with grids made from better human sourced data, visual grids and a simple CNN for the model architecture and I would figure out a better action selection mechanism. Right now the "solution" is the result is a number 1-100 that determines a space on the grid to select. 
## Approaches I decided to leave in
- Replay training which randomly samples previous moves and trains based on that so we aren't always training from the same match in consecutive steps.
- Target model training - normally the model trains against its own prediction which can cause issues if that model is outright incorrect the target model acts as a secondary model that produces the results the model that is playing trains against. The target model has its weights manually set based on combination of the playing model's weights and some discount factor
- Using the tracking grid as the state the model trains on and updates - the model shouldn't know the fully observable state
- Simple densely connected network with mean_squared_error - The softmax version performed worse from what I saw
## Additional Things I Tried
- Swapping the model for a softmax activated output layer and switching to categorical_crossentropy - I was hoping that given the 100 actions function sort of like a multiclass classification I could improve the correctness
- Swapping the model for a CNN based model with SumPooling - SumPooling would allow the model to target areas with the highest clustering of hits based on the size of the sliding window I chose - the issue is neither Tensorflow or Keras implement SumPooling so I would have to implement that myself.
- Swapping the action selection to choose the row first and return certain rewards based on how close that row was to a previous row and then select the column - the model effectively just made random choices.
