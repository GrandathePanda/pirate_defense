import numpy as np
from string import ascii_lowercase
from itertools import product
from random import choice

class BattleshipGrid():
  def __init__(self, tracking=False):
    self._grid = np.zeros((10,10))
    self._alphabet_rev_dict = { ascii_lowercase[i]: i for i in range(0, 10) }
    self._ship_sizes = {
      "aircraft_carrier": 5,
      "battleship": 4,
      "cruiser": 3,
      "destroyer": 2,
      "submarine": 1
    }
    self.previous_moves = set()
    self._directions = ["left", "right", "up", "down"]
    self._placed_ships = {}
    self.tracking = False
    self.actions = [ (letter, j) for j in range(0, 10) for letter in ascii_lowercase[0:10] ]

  def _letter_coord_lookup(self, coord):
    return self._alphabet_rev_dict[coord]

  def place_ship(self, ship, direction, row, col):
    row = self._letter_coord_lookup(row)

    if ship not in self._ship_sizes.keys():
      raise ValueError("Ship must be one of: " + "".join(self._ship_sizes.keys()))

    if direction not in self._directions:
      raise ValueError("Direction must be one of: " + "".join(self._directions))

    ship_size = self._ship_sizes[ship]
    coords = None

    if direction == "left":
      if col - ship_size < 0:
        raise ValueError(f"Cannot place {ship} left, this would exceed the bounds of the grid.")
      end_row = row
      end_col = col-ship_size
      coords = self._possible_coords(row, end_row, end_col, col)

      if self.check_overlap(coords):
        raise ValueError("Ship would overlap with another existing ship.")

      for i in range(end_col, col):
        self._grid[row][i] = 1

    if direction == "right":
      if col + ship_size > 10:
        raise ValueError(f"Cannot place {ship} right, this would exceed the bounds of the grid.")
      end_row = row
      end_col = col+ship_size
      coords = self._possible_coords(row, end_row, col, end_col)

      if self.check_overlap(coords):
        raise ValueError("Ship would overlap with another existing ship.")
      for i in range(col, end_col):
        self._grid[row][i] = 1

    if direction == "up":
      if row + ship_size > 10:
        raise ValueError(f"Cannot place {ship} up, this would exceed the bounds of the grid.")

      end_row = row+ship_size
      end_col = col
      coords = self._possible_coords(row, end_row, col, end_col)

      if self.check_overlap(coords):
        raise ValueError("Ship would overlap with another existing ship.")
      for i in range(row, end_row):
        self._grid[i][col] = 1

    if direction == "down":
      if row - ship_size < 0:
        raise ValueError(f"Cannot place {ship} down, this would exceed the bounds of the grid.")

      end_row = row-ship_size
      end_col = col
      coords = self._possible_coords(end_row, row, col, end_col)

      if self.check_overlap(coords):
        raise ValueError("Ship would overlap with another existing ship.")
      for i in range(end_row, row):
        self._grid[i][col] = 1

    self._placed_ships[ship] = {
      "hits": 0,
      "coords": list(set(list(coords))),
      "sunk": False
    }

  def place_hit(self, row, col):
    row = self._letter_coord_lookup(row)
    self._grid[row][col] = 1

  def place_miss(self, row, col):
    row = self._letter_coord_lookup(row)
    self._grid[row][col] = 0.5

  def fire(self, row, col):
    row = self._letter_coord_lookup(row)
    if (row, col) in self.previous_moves:
      return False
    self.previous_moves.add((row,col))
    if self._grid[row][col] == 1:
      for ship, data in self._placed_ships.items():
        if (row, col) in data["coords"]:
          data["coords"].remove((row, col))
          self._placed_ships[ship]["coords"] = data["coords"]
          self._placed_ships[ship]["hits"] += 1
          if self._ship_sizes[ship] <= self._placed_ships[ship]["hits"]:
            self._placed_ships[ship]["sunk"] = True
          return True

    return False

  def check_win(self):
    return all(map(lambda tup: tup[1]["sunk"], self._placed_ships.items()))

  def check_overlap(self, coords):
    for coord in coords:
      for _, data in self._placed_ships.items():
        if coord in data["coords"]:
          return True

    return False

  def check_sunk(self, ship):
    if self._placed_ships[ship]["hits"] >= self._ship_sizes[ship]:
      return True
    
    return False

  def place_ships(self):
    for ship in self._ship_sizes.keys():
      while True:
        direction = choice(self._directions)
        row = choice(range(0, 10))
        col = choice(range(0, 10))
        try:
          self.place_ship(ship, direction, ascii_lowercase[row], col)
          break
        except ValueError:
          continue

  def _possible_coords(self, row, end_row, col, end_col):
    list1 = list(range(row, end_row))
    list2 = list(range(col, end_col))

    if row == end_row:
      list1 = [row, row]
    if col == end_col:
      list2 = [col, col]

    return list(product(list1, list2))

