from battleship_grid import BattleshipGrid
from agent import Agent
import datetime
from string import ascii_lowercase

def step_forward(grid, tracking_grid, action):
  row, col = grid.actions[action]
  done = False
  
  hit = grid.fire(row, col)
  reward = 0.0

  if hit:
    tracking_grid.place_hit(row, col)
    reward = 10.0
    print(f"Hit! Decision: {(row, col)}")
  else:
    lookup = tracking_grid._letter_coord_lookup(row)
    if tracking_grid._grid[lookup][col] == 0.5 or tracking_grid._grid[lookup][col] == 1.0:
      return tracking_grid, -10.0, done
    tracking_grid.place_miss(row, col)

  if grid.check_win():
    print("Won!")
    done = True
  
  return tracking_grid, reward, done

def main():
  agent = Agent()
  trials = 1000
  time = 500
  batch_size = 128

  for trial in range(trials):
    tracking_grid = BattleshipGrid()
    p1_grid = BattleshipGrid()
    p1_grid.place_ships()
    won = False
    cur_state = tracking_grid._grid
    for step in range(time):
      action = agent.act(cur_state)
      new_state, reward, done = step_forward(p1_grid, tracking_grid, action)
      reward = reward if not done else -10.0
      new_state = new_state._grid
      agent.remember(cur_state, action, reward, new_state, done)

      if len(agent.memory) >= batch_size:
        agent.replay(batch_size)      

      agent.target_train()

      cur_state = new_state

      if done:
        won = True
        break
      if step % 50 == 0:
        print(f"Step: {step}")
    if won:
      print(f"Trial: {trial}, Time: {step}, Epsilon: {agent.epsilon}")
      won = False
      agent.save_model(f"model-win-{datetime.datetime.now()}")
    else: 
      print(f"Trial: {trial}, Time: {step}, Epsilon: {agent.epsilon}")
      print(f"Failed trial: {trial}")
      print(p1_grid._placed_ships)
    
    if trial % 100 == 0:
      print(f"Trial: {trial}, Time: {step}, Epsilon: {agent.epsilon}")
      agent.save_model(f"model-{datetime.datetime.now()}")


if __name__ == "__main__":
  main()