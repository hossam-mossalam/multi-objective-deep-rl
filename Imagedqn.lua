--[[

DQN
by: Yannis Assael
modified by: Hossam Mossalam

]] --

return function (ols_parameters)
  local weight = ols_parameters[1]
  local initialModel = ols_parameters[2]

  -- Configuration
  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('MultiObjective Image DQN')
  cmd:text()
  cmd:text('Options')

  -- general options:
  cmd:option('-seed', 5, 'initial random seed')
  cmd:option('-threads', 1, 'number of threads')

  -- model
  cmd:option('-replay_memory', 1e+5, 'experience replay memory')
  cmd:option('-gamma', 0.97, 'discount factor')
  cmd:option('-eps_start', 1, 'start ¿-greedy policy')
  cmd:option('-eps_end', 0.05, 'final ¿-greedy policy')
  cmd:option('-learn_start', 1, 'start learning episode')
  cmd:option('-eps_endt', 3000, 'final ¿-greedy policy episode')

  -- training
  cmd:option('-bs', 32, 'batch size')
  cmd:option('-nepisodes', 6000, 'number of episodes')
  cmd:option('-nsteps', 1000, 'number of steps')
  cmd:option('-target_step', 100, 'target network updates')

  -- print options
  cmd:option('-step', 100, 'print every episodes')
  cmd:option('-step_test', 10, 'print every episodes')

  cmd:text()

  local opt = cmd:parse(arg)
  opt.bs = math.min(opt.bs, opt.replay_memory)

  -- Requirements
  require 'nn'
  require 'optim'
  local kwargs = require 'include.kwargs'
  local log = require 'include.log'

  -- Set float as default type
  --torch.manualSeed(opt.seed)
  torch.random()
  torch.setnumthreads(opt.threads)
  torch.setdefaulttensortype('torch.FloatTensor')

  -- Cuda initialisation
  if opt.cuda then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(1)
    opt.dtype = 'torch.CudaTensor'
    print(cutorch.getDeviceProperties(1))
  else
    opt.dtype = 'torch.FloatTensor'
  end

  -- Converting the q  vector into a matrix in which the columns
  -- indicate the vector of rewards per action
  function opt.getQVectors(q, action_size, reward_size)
    q:resize(action_size, reward_size)
    q = q:transpose(1, 2)
    return q
  end

  -- scalarization weight
  local weight_size = #weight
  weight = torch.Tensor(weight):resize(weight_size, 1)
  local weight_bs = weight:view(1, 1, weight_size):expand(opt.bs, 1, weight_size)

  -- outputs best action
  -- q taken as input consists of a matrix in which column vectors
  -- represent the rewards per action
  function opt.maximum(q)
    return torch.max(torch.mm(weight:t(), q)[1], 1)
  end

  -- Initialise game
  local env = require 'ImageDeepSea'()
  local test_env = require 'ImageDeepSea'()

  local a_space = env:getActionSpec()[3]
  local r_space = { env:getRewardSpec() }
  local s_space = env:getStateSpec()
  local action_size = a_space[2] - a_space[1] + 1
  if type(r_space[2]) == "number" then
    r_space = { r_space }
  end
  local reward_size = #r_space

  -- Initialise model
  local exp = (require 'Imagemodel') {
    a_size = action_size,
    s_size = #s_space,
    r_size = reward_size,
    dtype = opt.dtype
  }

  local model = exp.model
  local model_target = exp.model:clone()
  if initialModel ~= nil then
    model = initialModel:clone()
    -- randomly initializing last layer
    model.modules[#model.modules]:reset()
    model_target = model:clone()
  end

  local params, gradParams = model:getParameters()
  local params_target, _ = model_target:getParameters()

  -- Optimisation function
  local optim_func, optim_config = exp.optim()
  local optim_state = {}

  -- Initialise aux vectors
  local td_err = torch.Tensor(opt.bs, action_size * reward_size):type(opt.dtype)

  local train_r_episode = torch.zeros(opt.nsteps, reward_size)
  local train_q_episode = torch.zeros(opt.nsteps)

  local train_r = 0
  local train_r_avg = 0

  local train_q = 0
  local train_q_avg = 0

  local test_r = 0
  local test_r_avg = 0
  local test_r_all = torch.zeros(opt.nepisodes, reward_size)

  local test_step = 0
  local test_step_avg = 0
  local test_step_all = torch.zeros(opt.nepisodes):type(opt.dtype)

  local train_episodes_reached_goal = 0
  local test_episodes_reached_goal = 0

  local step_count = 0
  local replay = {}

  -- Episode and training values storage
  local train = {
    s_t = torch.Tensor(opt.bs, s_space[1], s_space[2], s_space[3]):type(opt.dtype),
    s_t1 = torch.Tensor(opt.bs, s_space[1], s_space[2], s_space[3]):type(opt.dtype),
    r_t = torch.Tensor(opt.bs, reward_size):type(opt.dtype),
    a_t = torch.Tensor(opt.bs):type(opt.dtype),
    terminal = torch.Tensor(opt.bs):type(opt.dtype)
  }

  -- start time
  local beginning_time = torch.tic()

  for e = 1, opt.nepisodes do

    -- epsilon-greedy annealing
    opt.eps = (opt.eps_end +
        math.max(0, (opt.eps_start - opt.eps_end) * (opt.eps_endt -
        math.max(0, e - opt.learn_start)) / opt.eps_endt))

    -- Initial state
    local episode = {}
    episode.s_t = torch.Tensor(env:start())
    episode.terminal = false

    -- Initialise clock
    local time = sys.clock()

    -- Run for N steps
    local step = 1
    while step <= opt.nsteps and not episode.terminal do

      -- Compute Q values
      local q = model:forward(episode.s_t:type(opt.dtype)):clone()

      -- Pick an action (epsilon-greedy)
      if torch.uniform() < opt.eps then
        episode.a_t = torch.random(action_size)
      else
        -- converting the q into a matrix in which the columns
        -- indicate the vector of rewards of the actions
        q = opt.getQVectors(q, action_size, reward_size)
        local max_q, max_a = opt.maximum(q)
        episode.a_t = max_a:squeeze()
      end

      --compute reward for current state-action pair
      episode.r_t, episode.s_t1, episode.terminal = env:step(episode.a_t)

      if episode.terminal then
        train_episodes_reached_goal = train_episodes_reached_goal + 1
      end

      -- Store rewards
      train_r_episode[step] = torch.Tensor(episode.r_t)

      -- Store current step
      local r_id = (step_count % opt.replay_memory) + 1
      replay[r_id] = {
        r_t = episode.r_t,
        a_t = episode.a_t,
        s_t = episode.s_t,
        s_t1 = episode.s_t1,
        terminal = episode.terminal and 1 or 0
      }

      -- Fetch from experiences
      local q_next, q_next_max
      if #replay >= opt.bs then

        for b = 1, opt.bs do
          local exp_id = torch.random(#replay)
          train.r_t[b] = torch.Tensor(replay[exp_id].r_t)
          train.a_t[b] = replay[exp_id].a_t
          train.s_t[b] = replay[exp_id].s_t
          train.s_t1[b] = replay[exp_id].s_t1
          train.terminal[b] = replay[exp_id].terminal
        end

        -- Compute Q
        q = model:forward(train.s_t):clone()

        -- Use target network to predict q_max
        q_next = model_target:forward(train.s_t1):clone()
        q_next_max = torch.Tensor(opt.bs, reward_size)
        for i = 1, opt.bs do
          local current_q_next = q_next[i]
          current_q_next = opt.getQVectors(
                                            current_q_next,
                                            action_size,
                                            reward_size
                                          )
          local _, max_a_next = opt.maximum(current_q_next)
          max_a_next = max_a_next:squeeze()
          q_next_max[i] = current_q_next[{{},{max_a_next}}]
        end

        for b = 1, opt.bs do
          if train.terminal[b] == 1 then
            q_next[b] = 0
            q_next_max[b] = 0
          end
        end

        -- Q learnt value
        td_err:zero()
        for b = 1, opt.bs do
          local a_t_idx = {
                            (train.a_t[b] - 1) * reward_size + 1,
                            train.a_t[b] * reward_size
                          }
          td_err[{ { b }, a_t_idx }] = torch.Tensor(train.r_t[b])
                                        +  q_next_max[b] * opt.gamma
                                        - q[{ { b }, a_t_idx }]
        end


        -- Backward pass
        local feval = function(x)

          -- Reset parameters
          gradParams:zero()

          -- Backprop
          train_q_episode[step] = td_err:clone():pow(2):mean() * 0.5
          model:backward(train.s_t, -td_err)

          -- Clip Gradients
          --gradParams:clamp(-5, 5)

          return 0, gradParams
        end

        optim_func(feval, params, optim_config, optim_state)


        if step_count % opt.target_step == 0 then
          params_target:copy(params)
        end
      end

      -- next state
      episode.s_t = episode.s_t1:clone()
      step = step + 1

      -- Total steps
      step_count = step_count + 1
    end

    -- Compute statistics
    train_q = train_q_episode:narrow(1, 1, step - 1):mean()
    train_r = train_r_episode:narrow(1, 1, step - 1):sum(1)
    test_r, test_step = exp.test(opt, test_env, model, action_size, reward_size)

    if test_step < math.huge then
        test_episodes_reached_goal = test_episodes_reached_goal + 1
    end

    -- Compute moving averages
    if e == 1 then
        train_q_avg = train_q
        train_r_avg = train_r
        test_r_avg = test_r
        test_step_avg = test_step
    else
        train_q_avg = train_q_avg * 0.99 + train_q * 0.01
        train_r_avg = train_r_avg * 0.99 + train_r * 0.01
        test_r_avg  = test_r_avg  * 0.99 + test_r  * 0.01
        test_step_avg = test_step_avg * 0.99 + test_step * 0.01
    end
    test_r_all[e] = test_r
    if test_step ~= math.huge then
        test_step_all[e] = test_step
    end

    -- Print statistics
    if e == 1 or e % opt.step == 0 then

      local temp = test_r_all:narrow(1,1,e):mean(1)
      log.infof('e=%d, train_q=%.3f, train_q_avg=%.3f, last_train_r=%.3f;%.3f, ' ..
          'train_r_avg=%.3f;%.3f, test_r=%.3f,%.3f, test_r_avg=%.3f;%.3f, ' ..
          't/e=%.2f sec, t=%d min., test_step_avg=%.3f, ' ..
          'train_episodes_reached_goal=%d, test_episodes_reached_goal=%d ' ..
          'current_weight = < %.4f, %.4f >', e, train_q, train_q_avg, train_r[1][1], train_r[1][2],
          train_r_avg[1][1], train_r_avg[1][2], test_r[1], test_r[2], temp[1][1], temp[1][2],
          sys.clock() - time, torch.toc(beginning_time) / 60,
          test_step_all:narrow(1,1,e):mean(), train_episodes_reached_goal, test_episodes_reached_goal,
          weight[1]:squeeze(), weight[2]:squeeze())

    end
  end

  local initial_state_q = model:forward(torch.Tensor(env:start()):type(opt.dtype)):clone()
  initial_state_q = opt.getQVectors(initial_state_q, action_size, reward_size)
  local max_q, max_a = opt.maximum(initial_state_q)
  max_a = max_a:squeeze()
  local temp = model:clone()
  return initial_state_q:t()[max_a], model
end
