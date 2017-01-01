--[[

DOL(-R)
by Hossam Mossalam

]]--

local log = require 'include.log'
torch.setdefaulttensortype('torch.FloatTensor')

-- inserts w in the priority queue according to its improvement value
-- w must be in the format (w1, improvement)
function enqueueWeight(queue, w)
  table.insert(queue, w)
  for i = #queue, 2, -1 do
    if queue[i][2] > queue[i - 1][2] then
      local temp = queue[i - 1]
      queue[i - 1] = queue[i]
      queue[i] = temp
    else
      break
    end
  end
end

-- returns the weight with max improvement stored in queue
-- weight is of the format: (w1, improvement)
function dequeueWeight(queue)
  return table.remove(queue, 1)
end

-- used to check if a value vector V is included in the table arr.
function isIncluded(arr, V)
  for i = 1, #arr do
    if arr[i][1] == V[1] and arr[i][2] == V[2] then
      return true
    end
  end
  return false
end

-- similarity measure between different weights
function weightDistance(w1, w2)
  return math.sqrt((w1[1] - w2[1])^2 + (w1[2] - w2[2])^2)
end

-- returns intersection point of 4 points in the 2-d coordinate systsm
function intersection(p1, p2, p3, p4)
  local denominator = ((p1[1] - p2[1]) * (p3[2] - p4[2]))
                      - ((p1[2] - p2[2]) * (p3[1] - p4[1]))
  local numeratorX = (((p1[1] * p2[2]) - (p1[2] * p2[1])) * (p3[1] - p4[1]))
                      - ((p1[1] - p2[1]) * ((p3[1] * p4[2]) - (p3[2] * p4[1])))
  local numeratorY = (((p1[1] * p2[2]) - (p1[2] * p2[1])) * (p3[2] - p4[2]))
                      - ((p1[2] - p2[2]) * ((p3[1] * p4[2]) - (p3[2] * p4[1])))
  return numeratorX/denominator, numeratorY/denominator
end

-- calculates the new corner weights in the convex upper sub-surface
function newCornerWeights(V_PI, W_DEL, S, VToPoints)
  local p1 = VToPoints[V_PI][1]
  local p2 = VToPoints[V_PI][2]
  local y_axis_intersect = VToPoints[V_PI][3]
  local CornerWeights = {}
  for i = 1, #S do
    local Vi = S[i]
    local p3 = VToPoints[Vi][1]
    local p4 = VToPoints[Vi][2]
    local current_y_axis_intersect = VToPoints[Vi][3]
    local cornerW, Y = intersection(p1, p2, p3, p4)
    -- checks if the new corner weight is in the range of the two lines
    -- Redundant sets of the points back
    if not (cornerW > p2[1] or cornerW < p1[1] or cornerW > p4[1] or cornerW < p3[1]) then
      if y_axis_intersect > current_y_axis_intersect then
        p3[1] = cornerW
        p3[2] = Y
        p2[1] = cornerW
        p2[2] = Y
        VToPoints[V_PI][2] = p2
        VToPoints[Vi][1] = p3
      else
        p4[1] = cornerW
        p4[2] = Y
        p1[1] = cornerW
        p1[2] = Y
        VToPoints[V_PI][1] = p1
        VToPoints[Vi][2] = p4
      end
      table.insert(CornerWeights, {cornerW, nil})
    end
  end
  return CornerWeights
end

-- estimates the improvement of a corner weight
function estimateImprovement(cornerWeight, S, VToPoints)
  local startingVector = nil
  local endingVector = nil
  for i = 1, #S do
    local Vi = VToPoints[S[i]]
    if Vi[1][1] == cornerWeight then
      startingVector = Vi
    elseif Vi[2][1] == cornerWeight then
      endingVector = Vi
    end
  end
  local firstPoint = endingVector[1]
  local cornerPoint = startingVector[1]
  local lastPoint = startingVector[2]
  --changed height calculation to be with the point corner-weigt and 100
  local _, height = intersection(firstPoint, lastPoint, {cornerWeight, 100}, cornerPoint)
  -- removed the division to handle the case when the value vector is negative
  return (height - cornerPoint[2]) --/ cornerPoint[2]
end

-- checks if a weight has w1 between the range of s and e
-- and in that case it deletes it
function removeObseleteWeights(Q, s, e)
  for i = #Q, 1, -1 do
    if (Q[i][1] < e and Q[i][1] > s) and Q[i][2] < math.huge then
      table.remove(Q, i)
    end
  end
end

-- finds the most similar weight and returns the model of that weight
-- the input weight is of the format {w0, w1}
function mostSimilarModel(weight, W, Models)
  if #Models == 0 then
    return nil
  end
  local mostSimilar = nil
  local distance = math.huge
  for key, value in pairs(Models) do
    local w_i1 = key[1]
    local w_i0 = 1 - w_i1
    w_i = { w_i0, w_i1}
    if weightDistance(weight, w_i) < distance then
      mostSimilar = value
    end
  end
  return mostSimilar:clone()
end

-- Removes value vectors made obselete by the new V_PI.
function removeObseleteValueVectors(V_PI, S, W, VToPoints)
  local obseleteValueVectors = {}
  for i = #S, 1, -1 do
    local V_i = S[i]
    local range_vi = VToPoints[V_i]
    local s = range_vi[1][1]
    local e = range_vi[2][1]
    local w1_s = s
    local w0_s = 1 - w1_s
    local w1_e = e
    local w0_e = 1 - w1_e
    if w0_s * V_PI[1] + w1_s * V_PI[2] > w0_s * V_i[1] + w1_s * V_i[2] and
        w0_e * V_PI[1] + w1_e * V_PI[2] > w0_e * V_i[1] + w1_e * V_i[2] then
      table.insert(obseleteValueVectors, V_i)
      table.remove(S, i)
    end
  end
  return obseleteValueVectors
end

function hasImprovement(w, V_PI, S, VToPoints)
  if #S == 0 or w[2] == math.huge then
    return true
  end
  local w1 = w[1]
  local currentHeight = nil
  for i = 1, #S do
    local Vi = VToPoints[S[i]]
    if Vi[1][1] == w1 then
      currentHeight = Vi[1][2]
      break
    end
  end
  local x, y = intersection({w1, 0}, {w1, currentHeight}, {0, V_PI[1]}, {1, V_PI[2]})
  if y > currentHeight then
    return true
  else
    return false
  end
end

function estimateError(S, W)
  local total_err = 0
  local current_error = 0
  for _, w in pairs(W) do
    local w1 = w[1][1]
    local w0 = 1 - w1
    current_error = 0
    local sv = S[1]
    for _, v in pairs(S) do
      if v[1] * w0 + v[2] * w1 > sv[1] * w0 + sv[2] * w1 then
        sv = v
      end
    end
    local optimal_v = solveSODP2({w0, w1}, nil, nil)
    total_err = total_err + math.abs((optimal_v[1] * w0 + optimal_v[2] * w1) - (sv[1] * w0 + sv[2] * w1))
  end
  return total_err
end

-- returns model and V_PI
-- V_PI is the value vector obtained by solving the MOMDP at w and
-- current_model is the model at the end of learning
function solveSODP(weight, W, Models, reuse_flag)
  local initialModel = nil
  if reuse_flag then
    local initialModel = mostSimilarModel(weight, W, Models)
  end
  --local V_PI, model = (require 'Imagedqn'){weight, initialModel}
  local V_PI, model = (require 'dqn'){weight, initialModel}
  return V_PI, model
end

-- DOL/DOL-R
-- weights are in the form {w1, improvement}
-- S contains discovered value vectors
-- W contains the weights that were used to get value vectors
-- Q contains the weights that need to be checked
-- VToPoints contains the mapping from a vector to the range it is optimal at
-- and each point is {x, y}
local S = {}
local W = {}
local Models = {}
local rejectedWeights = {}
local VToPoints = {}

-- hyper-parameters

-- used to control the minimum-improvement value for a weight to be used by DOL/DOL-R.
local minimum_improvement = 0

-- used to control the peek weights used by DOL/DOL-R.
local maxWeight = 1

-- setting initial weights (w1, improvement)
local Q = {{1 - maxWeight, math.huge}, {maxWeight, math.huge}}


-- solution specs
local reuse_flag = false

while #Q ~= 0 do
  local w = dequeueWeight(Q)
  local V_PI, current_model = solveSODP({1 - w[1], w[1]}, W, Models, reuse_flag)

  -- 2 points on the line to represent the
  local p1, p2 = {0, V_PI[1]}, {1, V_PI[2]}
  table.insert(W, {w, V_PI})
  if not isIncluded(S, V_PI) and hasImprovement(w, V_PI, S, VToPoints) then
      Models[w] = current_model  -- should I store the model for any w even if it is not the one accepted
      -- Stores the range for which V_PI is optimal in the current CCS,
      -- and it's intercept with the y-axis
      VToPoints[V_PI] = {p1, p2, V_PI[1]}
      local W_DEL = {w}
      -- Removes value vectors made obselete
      local obseleteValueVectors = removeObseleteValueVectors(V_PI, S, W, VToPoints)
      local W_V_PI = newCornerWeights(V_PI, W_DEL, S, VToPoints)
      local range = VToPoints[V_PI]
      table.insert(S, V_PI)
      removeObseleteWeights(Q, VToPoints[V_PI][1][1], VToPoints[V_PI][2][1])
      for i = 1, #W_V_PI do
          local cornerWeight = W_V_PI[i]
          cornerWeight[2] = estimateImprovement(cornerWeight[1], S, VToPoints)
          if cornerWeight[2] > minimum_improvement then
              enqueueWeight(Q, cornerWeight)
          else
              table.insert(rejectedWeights, cornerWeight)
          end
      end
  end
end
