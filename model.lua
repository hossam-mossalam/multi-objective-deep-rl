require 'modules.rmsprop'
require 'nn'
require 'nngraph'
require 'optim'

local kwargs = require 'include.kwargs'
local log = require 'include.log'

return function(opt)
  local opt = kwargs(opt, {
    { 'a_size', type = 'int-pos' },
    { 's_size', type = 'int-pos' },
    { 'r_size', type = 'int-pos' },
    { 'dtype', type = 'string', default = 'torch.FloatTensor' },
    { 'lr', type = 'number', default = 1e-3 }
  })

  local exp = {}

  function exp.optim(iter)
    local optimfunc = optim.adam
    local optimconfig = {learningRate = opt.lr}
    return optimfunc, optimconfig
  end

  function exp.test(opt, env, model, action_size, reward_size)

    -- Run for N steps
    local s_t, s_t1, a_t, r_t
    local terminal = false

    -- Initial state
    s_t = torch.Tensor(env:start()):type(opt.dtype)

    local step = 1
    local r = 0
    if reward_size > 1 then
      r = torch.zeros(reward_size)
    end
    while step <= opt.nsteps and not terminal do

      -- get argmax_u Q from DQN
      local q = model:forward(s_t):clone()

      -- Pick an action
      if reward_size == 1 then
        local max_q, max_a = torch.max(q, 2)
        a_t = max_a:squeeze()
      else
        q:resize(action_size, reward_size)
        q = q:transpose(1,2)
        local max_q, max_a = opt.maximum(q)
        a_t = max_a:squeeze()
      end

      --compute reward for current state-action pair
      r_t, s_t1, terminal = env:step(a_t)
      s_t1 = torch.Tensor(s_t1):type(opt.dtype)

      if reward_size > 1 then
        r = r + torch.Tensor(r_t)
      else
        r = r + r_t
      end

      -- next state
      s_t = s_t1:clone()

      step = step + 1
    end

    if terminal == false then
      step = math.huge
    end

    return r, step
  end

  local function create_model(opt)
    local opt = kwargs(opt, {
      { 'a_size', type = 'int-pos' },
      { 's_size', type = 'int-pos' },
      { 'r_size', type = 'int-pos' },
      { 'dtype', type = 'string', default = 'torch.FloatTensor' }
    })

    local model = nn.Sequential()
    model:add(nn.View(-1, opt.s_size))
    model:add(nn.Linear(opt.s_size, opt.s_size * 20))
    model:add(nn.ReLU(true))
    model:add(nn.Linear(opt.s_size * 20, opt.a_size * opt.r_size))
    return model:type(opt.dtype)
  end

  -- Create model
  exp.model = create_model {
    a_size = opt.a_size,
    s_size = opt.s_size,
    r_size = opt.r_size,
    dtype = opt.dtype
  }

  return exp
end
