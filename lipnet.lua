--[[

LipNet: End-to-End Sentence-level Lipreading. arXiv preprint arXiv:1611.01599 (2016).

Copyright (C) 2017 Yannis M. Assael, Brendan Shillingford, Shimon Whiteson, Nando de Freitas

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

]]--


--
-- Dependencies
--

require 'io'
require 'sys'
require 'nn'
require 'nngraph'
require 'optim'
require 'hdf5'
require 'paths'

require 'cutorch'
require 'cunn'
require 'cudnn'
cudnn.benchmark = true
cudnn.fastest = true

require 'warp_ctc'

require 'nnx'

local log = require 'util.log'
log.level = "debug"

require 'modules.CTCCriterionFull'


--
-- Configuration
--

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Options')
cmd:option('-seed', 123, 'initial random seed')
cmd:option('-datapath', 'data', 'video data path')
cmd:option('-bs', 50, 'batch size')
cmd:option('-rnn_size', 256, 'rnn size')
cmd:option('-test_overlapped', 0, 'test overlapped speakers')
cmd:option('-normalise', 1, 'normalise images')
cmd:option('-dropout', 0.5, '')
cmd:option('-data_augmentation_temporal', 0.05, '')
cmd:option('-gradient_noise', 0, '')
cmd:option('-curriculum', 0.925, '')
cmd:option('-mode_img', 'mouth', 'mouth')
cmd:option('-threads', 4, 'number of torch built-in threads')
cmd:option('-exp', 'exp.0001', 'lua file that returns model, experiment name, optimization settings, ...')
cmd:option('-exp_i', 1, 'if exp returns more than one model, uses this index')
cmd:option('-ignore_checkpoint', 1, 'do not continue from checkpoint')
cmd:option('-print_every', 1, 'iterations between printing')
cmd:option('-test_every', 1, 'iterations between testing')
cmd:option('-checkpoint_every', 1, 'iterations between saving checkpoints')
cmd:option('-checkpoint', '', '')
cmd:option('-use_optnet', 0, 'use OptNet')
cmd:option('-num_threads', 10, 'dataloading threads')
cmd:option('-debug', 0, '')
cmd:text()
local opt = cmd:parse(arg)
assert(opt.exp ~= '', 'exp lua file required')
for k, v in pairs(opt) do
    log.infof('opt: %s=%s', k, tostring(v))
end

--
-- Initialisation
--
torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')

--
-- CUDA
--
math.randomseed(opt.seed)
torch.manualSeed(opt.seed)
cutorch.manualSeed(opt.seed)

-- set up experiment results directory:
local expname = opt.exp .. '__' .. opt.exp_i
local expdir = paths.concat('results', expname)
local function expfile(fn) -- returns full pathname to file in experimen's subdir
    return paths.concat(expdir, fn)
end

log.outfile = expfile('log_' .. os.date('%Y-%m-%d_%H-%M-%S') .. '.txt')

if paths.dirp(expdir) then
    log.warn('directory for experiment already exists.')
else
    paths.mkdir(expdir)
end
log.infof('experiment name incl index = %s', expname)
log.infof('experiment dir = %s', expdir)

--
-- Aux functions
--

function math.finite(x)
    if x == nil or x >= math.huge or x <= -math.huge or x ~= x then
        return false
    else
        return true
    end
end


function test(model, exp, n)
    n = math.min(n, opt.bs)
    -- get a sample
    local input = exp.data_loader:forward(nil, true)
    -- Get predictions
    local y_preds = model.pred:forward(input[1]):clone()
    -- Softmax it
    y_preds:exp()
    y_preds:cdiv(y_preds:sum(3):expandAs(y_preds))

    -- First is black magic
    for b = 1, n do
        local len = input[3][b]
        local y = input[2][b]
        local _, y_pred = torch.max(y_preds:narrow(2, b, 1):squeeze(), 2)
        y_pred = y_pred:add(-1):squeeze()

        local y_str = {}
        for j = 1, #y do
            y_str[j] = exp.data_loader.vocab[y[j]]
        end

        local y_pred_str = {}
        for j = 1, len do
            if y_pred[j] > 0 then
                if j > 1 then
                    if y_pred[j - 1] ~= y_pred[j] then
                        table.insert(y_pred_str, exp.data_loader.vocab[y_pred[j]])
                    end
                else
                    table.insert(y_pred_str, exp.data_loader.vocab[y_pred[j]])
                end
            end
        end

        if opt.mode_sub == 3 then
            log.infof('test seq %d: %s; pred seq: %s; pred seq sum: %d',
                b, table.concat(y_str, ""), table.concat(y_pred_str, ""), y_pred:gt(0):sum())
        else
            log.infof('test seq %d: %s; pred seq: %s; pred seq sum: %d',
                b, table.concat(y_str, " "), table.concat(y_pred_str, " "), y_pred:gt(0):sum())
        end
    end
end

-- 
-- Construct model:
--
local exp = require(opt.exp)(opt)
local model = exp.model -- MUST already be cudified
local crit = nn.CTCCriterionFull():cuda() -- from nnx
local softmax = nn.SoftMax():cuda()
local grad_noise
if opt.gradient_noise == 1 then
    grad_noise = model.grads.new():resizeAs(model.grads)
end

-- Load Model
if opt.checkpoint ~= '' then
    log.info('Loading Model')
    local checkpoint = torch.load(opt.checkpoint)
    exp.model = checkpoint.model
    model = exp.model
    print(checkpoint.opt)
end

-- OptNet
if opt.use_optnet == 1 then
    local sampleInput = torch.zeros(opt.bs, 3, 75, 50, 100):cuda()
    optnet.optimizeMemory(model.pred, sampleInput, { inplace = false, mode = 'training' })
end

-- Training loop:
local stats = {
    losses = torch.FloatTensor(exp.nIter):zero(),
    losses_test = torch.FloatTensor(exp.nIter):zero(),
    loss_ewma
}

local optimState = {}
for ep = 1, exp.nIter do

    if exp.preTrainCallback then exp.preTrainCallback(model) end
    local optimFunc, optimConfig = exp.optim(ep)

    local idx_shuffle = torch.randperm(#exp.data_loader.dataset)

    -- Train
    local bs_count = 0
    model.pred:training()
    for it = 1, #exp.data_loader.dataset, opt.bs do
        xlua.progress(it, #exp.data_loader.dataset)

        -- batch indexes
        local idx = {}
        for i = 0, opt.bs - 1 do
            if it + i <= #exp.data_loader.dataset then
                table.insert(idx, idx_shuffle[it + i])
            end
        end

        -- zero params
        model.grads:zero()

        -- load data
        local x, y, lengths = unpack(exp.data_loader:forward(idx))

        local logits = model.pred:forward(x)
        local loss_all = torch.Tensor(crit:forward(logits, y, lengths))
        local loss = loss_all:mean()

        -- skip this iteration if the loss is nan
        if math.finite(loss) and loss >= -1000000 and loss <= 1000000 then -- nan == nan is false, math.huge
            bs_count = bs_count + #idx

            local dlogits = crit:backward(logits, y)

            -- Curriculum weighting
            if opt.curriculum > 0 then
                for b = 1, #idx do
                    local id = idx[b]
                    if exp.data_loader.dataset[id].mode == 1 then
                        local ratio = opt.curriculum ^ (ep - 1)
                        dlogits[{ {}, { b } }]:mul(ratio)
                    end
                end
            end

            model.pred:backward(x, dlogits)

            -- model.grads:clamp(-10, 10)
            -- model.grads:div(opt.bs)

            optimFunc(function() return loss, model.grads end,
                model.params, optimConfig, optimState)

            -- print(loss, model.grads:norm())

            stats.losses[ep] = stats.losses[ep] + loss * (#idx)
        else
            if type(x) == 'table' then
                print('x', x[1]:min(), x[1]:max())
            else
                print('x', x:min(), x:max())
            end
            print('logits', softmax:forward(logits):min(), softmax:forward(logits):max(), logits:size(1), logits:size(2))
            print('y', #y)
            print('lengths', lengths:min(), lengths:max())
            print(loss)
            log.debug('skipping iteration with nan loss')
        end
    end
    stats.losses[ep] = stats.losses[ep] / bs_count

    -- Initialise moving average loss
    if ep == 1 then
        stats.loss_ewma = stats.losses[ep]
    else
        stats.loss_ewma = stats.loss_ewma * 0.95 + stats.losses[ep] * 0.05
    end

    -- Test
    local bs_count = 0
    model.pred:evaluate()
    for it = 1, #exp.data_loader.dataset_val, opt.bs do

        -- batch indexes
        local idx = {}
        for i = 0, opt.bs - 1 do
            if it + i <= #exp.data_loader.dataset_val then
                table.insert(idx, it + i)
            end
        end

        -- load data
        local x, y, lengths = unpack(exp.data_loader:forward(idx, true))

        local logits = model.pred:forward(x)
        local loss_all = torch.Tensor(crit:forward(logits, y, lengths))
        local loss = loss_all:mean()
        -- print(loss)
        -- skip this iteration if the loss is nan
        if math.finite(loss) and loss >= -100000 and loss <= 100000 then
            bs_count = bs_count + (#idx)
            stats.losses_test[ep] = stats.losses_test[ep] + loss * (#idx)
        else
            if type(x) == 'table' then
                print('x', x[2]:min(), x[2]:max())
            else
                print('x', x:min(), x:max())
            end
            print('logits', softmax:forward(logits):min(), softmax:forward(logits):max(), logits:size(1), logits:size(2))
            print('y', #y)
            print('lengths', lengths:min(), lengths:max())
            print(loss)
            log.debug('skipping iteration with nan test loss')
        end
    end
    stats.losses_test[ep] = stats.losses_test[ep] / bs_count

    -- Print training statistics:
    if ep % opt.print_every == 0 then
        log.info(string.format('iter=%d,loss=%.5f,avg=%.5f,loss_test=%.5f,loss_test_best=%.5f,paramnorm=%.5f,gradnorm=%.5f',
            ep,
            stats.losses[ep],
            stats.loss_ewma,
            stats.losses_test[ep],
            stats.losses_test[{ { 1, ep } }]:min(),
            model.params:norm(),
            model.grads:norm()))
    end

    -- Run test function:
    if (ep < 3 or ep % opt.test_every == 0) and #exp.data_loader.dataset_val > 0 then
        model.pred:evaluate()
        test(model, exp, 3)
    end

    -- if ep % opt.checkpoint_every == 0 or ep == 1 or ep == exp.nIter then
    if ep == 1 or (ep > 1 and stats.losses_test[ep] < stats.losses_test[{ { 1, ep - 1 } }]:min()) then
        -- Save checkpoint:

        for _, obj in pairs(model) do if obj.clearState then obj:clearState() end end

        local checkpoint = {
            model = model,
            optimState = optimState,
            stats = stats,
            iter = ep,
            opt = opt
        }

        torch.save(expfile(string.format('checkpoint_e%08d_loss%.5f.t7', ep, stats.losses_test[ep])), checkpoint)
        log.info('saved checkpoint-----')

        -- OptNet
        if opt.use_optnet == 1 then
            local sampleInput = torch.zeros(opt.bs, 3, 75, 50, 100):cuda()
            optnet.optimizeMemory(model.pred, sampleInput, { inplace = false, mode = 'training' })
        end
    end

    -- collect garbage
    collectgarbage()
end -- for it=startIter,exp.nIter

