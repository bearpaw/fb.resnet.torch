--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'paths'
require 'optim'
require 'nn'
local DataLoader = require 'dataloader'
local models = require 'models/init'
local Trainer = require 'train'
local opts = require 'opts'
local checkpoints = require 'checkpoints'

-- we don't  change this to the 'correct' type (e.g. HalfTensor), because math
-- isn't supported on that type.  Type conversion later will handle having
-- the correct type.
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)

-- Count operations
local opCounter = require 'utils.torch-opCounter.src.profiler'

local inputRes = 224
if opt.dataset == 'cifar10' or opt.dataset == 'cifar100' then
   inputRes = 32
end

local sampleInput = torch.randn(1, 3, inputRes, inputRes):cuda()
local total, layer_ops = count_ops(model, sampleInput)
print(string.format('    Total: %.2f GFLOPS', total/10^9))

-- Count parameters
params, gradParams = model:getParameters()
print(('    Parameters: %.2fM'):format(params:size(1)/1000000))