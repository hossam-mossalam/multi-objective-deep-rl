local classic = require 'classic'
local log = require 'include.log'

local ImageDeepSea, super = classic.class('ImageDeepSea', Env)

-- Constructor
function ImageDeepSea:_init(opts)

  opts = opts or {}

  -- original map
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

  -- initial image
  local im = torch.Tensor(3, 11, 11)
  for i = 1, 11 do
    for j = 1, 11 do
      if self.map[i][j] == -1 then      --walls are black
        im[1][i][j] = 0
        im[2][i][j] = 0
        im[3][i][j] = 0
      elseif self.map[i][j] == 1 then   --empty cells blue
        im[1][i][j] = 0
        im[2][i][j] = 0
        im[3][i][j] = 1
      else                              --goals are red
        im[1][i][j] = 1
        im[2][i][j] = 0
        im[3][i][j] = 0
      end
    end
  end
  self.im = im

end

-- returning the initial state image
function ImageDeepSea:getStateSpec()
  return #self.im
end

-- 1 action required, of type 'int', of dimensionality 1, between 1 and 4  (up, down, left, right)
function ImageDeepSea:getActionSpec()
  return {'int', 1, {1, 4}}
end

-- Min and max reward per objective
function ImageDeepSea:getRewardSpec()
  return {0, 1}, {-1, -1}
end

-- Reset the positions
function ImageDeepSea:start()
  self.row = 1
  self.col = 1

  local image = self.im:clone()
  image[2][1][1] = 1

  return image
end

-- uses the submarine: 1 up, 2 down, 3 left, 4 right
function ImageDeepSea:step(action)

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
    rewards[1] = self.map[self.row][self.col] / 143
    terminal = true
  end

  local image = self.im:clone()
  image[2][self.row][self.col] = 1

  return rewards, image, terminal
end

return ImageDeepSea
