local classic = require 'classic'
local log = require 'include.log'

local DeepSea, super = classic.class('DeepSea', Env)

-- Constructor
function DeepSea:_init(opts)

  opts = opts or {}

  local row1  = {  1,  1,  1,  1,  1,  1,    1,     1,     1,     1, 1}
  local row2  = {0.5,  1,  1,  1,  1,  1,    1,     1,     1,     1, 1}
  local row3  = { -1, 28,  1,  1,  1,  1,    1,     1,     1,     1, 1}
  local row4  = { -1, -1, 52,  1,  1,  1,    1,     1,     1,     1, 1}
  local row5  = { -1, -1, -1, 73, 82, 90,    1,     1,     1,     1, 1}
  local row6  = { -1, -1, -1, -1, -1, -1,    1,     1,     1,     1, 1}
  local row7  = { -1, -1, -1, -1, -1, -1,    1,     1,     1,     1, 1}
  local row8  = { -1, -1, -1, -1, -1, -1,115.5, 120.4,     1,     1, 1}
  local row9  = { -1, -1, -1, -1, -1, -1,   -1,    -1,     1,     1, 1}
  local row10 = { -1, -1, -1, -1, -1, -1,   -1,    -1, 134.6,     1, 1}
  local row11 = { -1, -1, -1, -1, -1, -1,   -1,    -1,    -1, 143.5, 1}
  self.map = {row1, row2, row3, row4, row5, row6, row7, row8, row9, row10, row11}

end

-- 2 states returned, of type 'real', of dimensionality 1, with ranges 1-11
function DeepSea:getStateSpec()
  return {
    {'int', 1, {1, 11}}, -- row
    {'int', 1, {1, 11}}  -- col
  }
end

-- 1 action required, of type 'int', of dimensionality 1, between 1 and 4  (up, down, left, right)
function DeepSea:getActionSpec()
  return {'int', 1, {1, 4}}
end

-- Min and max reward per objective
function DeepSea:getRewardSpec()
  return {0, 1}, {-1, -1}
end

-- Reset the positions
function DeepSea:start()
  self.row = 1
  self.col = 1

  return {self.row, self.col}
end

-- uses the submarine: 1 up, 2 down, 3 left, 4 right
function DeepSea:step(action)

  if action == 1 and self.row > 1 and self.map[self.row - 1][self.col] ~= -1 then
    self.row = self.row - 1
  elseif action == 2 and self.row < 11 and self.map[self.row + 1][self.col] ~= -1 then
    self.row = self.row + 1
  elseif action == 3 and self.col > 1 and self.map[self.row][self.col - 1] ~= -1 then
    self.col = self.col - 1
  elseif action == 4 and self.col < 11 and self.map[self.row][self.col + 1] ~= -1 then
    self.col = self.col + 1
  end

  -- Calculate reward
  local rewards = {0, -1/143.5}

  -- Calculate termination
  local terminal = false

  if self.map[self.row][self.col] ~= -1 and self.map[self.row][self.col] ~= 1 then
    rewards[1] = self.map[self.row][self.col] / 143.5
    terminal = true
  end

  return rewards, torch.Tensor({self.row, self.col}), terminal
end

return DeepSea
