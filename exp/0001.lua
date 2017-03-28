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

package.path = package.path .. ";../modules/?.lua"
require 'nn'
require 'cunn'
require 'nngraph'
require 'optim'
require 'modules.TemporalJitter'
require 'modules.TimeBatchWrapper'
local log = require 'util.log'
local nninit = require 'nninit'

return function(opt)

    local exp = {}

    function exp.optim(iter)
        local optimfunc = optim.adam
        local lr = 1e-4
        local optimconfig = {
            learningRate = lr,
        }
        return optimfunc, optimconfig
    end

    --
    -- Load data
    --
    exp.data_loader = require('dataloader')(opt)
    exp.data_loader:load_data()
    --
    -- Define Network
    --
    local input_img_size = exp.data_loader.opt.size
    local vocab_size = exp.data_loader.opt.vocab_size
    local maxT = exp.data_loader.opt.max_timesteps

    -- data_source, mouth
    -- image_dims, 1x50x100, size of a single input frame, <chans>x<width>x<height>)
    -- image_mean, 128, RGB mean value of image to subtract for ALL channels)
    -- image_std, 128, RGB mean value of image to divide (ALL channels))
    local chan, height, width = unpack(input_img_size)
    assert(chan == 1 or chan == 3, '1 or 3 channels only, in input_img_size')
    assert(width > 0 and height > 0, 'width and height must be specified in input_img_size')

    local model = {}
    model.pred = nn.Sequential()
        :add(cudnn.VolumetricConvolution(3, 32, 3, 5, 5, 1, 2, 2, 1, 2, 2)
            :init('weight', nninit.kaiming, { gain = 'relu' })
            :init('bias', nninit.constant, 0))
        :add(cudnn.ReLU(true))
        :add(cudnn.VolumetricMaxPooling(1, 2, 2, 1, 2, 2))
        :add(nn.VolumetricDropout(opt.dropout))
        :add(cudnn.VolumetricConvolution(32, 64, 3, 5, 5, 1, 1, 1, 1, 2, 2)
            :init('weight', nninit.kaiming, { gain = 'relu' })
            :init('bias', nninit.constant, 0))
        :add(cudnn.ReLU(true))
        :add(cudnn.VolumetricMaxPooling(1, 2, 2, 1, 2, 2))
        :add(nn.VolumetricDropout(opt.dropout))
        :add(cudnn.VolumetricConvolution(64, 96, 3, 3, 3, 1, 1, 1, 1, 1, 1)
            :init('weight', nninit.kaiming, { gain = 'relu' })
            :init('bias', nninit.constant, 0))
        :add(cudnn.ReLU(true))
        :add(cudnn.VolumetricMaxPooling(1, 2, 2, 1, 2, 2))
        :add(nn.VolumetricDropout(opt.dropout))
        :add(nn.Transpose({ 1, 2 }, { 1, 3 }))
        :add(nn.TimeBatchWrapper {mod = nn.Sequential()
            :add(nn.View(-1, 96 * 3 * 6))
            })

    -- requires cuDNN v5:
    local rnn = cudnn.BGRU(96 * 3 * 6, opt.rnn_size, 1)
    local weights = rnn:weights()
    local biases = rnn:biases()
    for layer = 1, #weights do
        for layerId = 1, #weights[layer] do

            local weightTensor = weights[layer][layerId]
            if weightTensor:size(1) ~= opt.rnn_size ^ 2 then
                local stdv = math.sqrt(2 / (96 * 3 * 6 + opt.rnn_size))
                weightTensor:uniform(-math.sqrt(3) * stdv, math.sqrt(3) * stdv)
            else
                local fanIn = opt.rnn_size
                local fanOut = opt.rnn_size

                -- Construct random matrix
                local randMat = torch.Tensor(fanOut, fanIn):normal(0, 1)
                local U, __, V = torch.svd(randMat, 'S')

                -- Pick out orthogonal matrix
                local W
                if fanOut > fanIn then
                    W = U
                else
                    W = V:narrow(1, 1, fanOut)
                end
                -- Resize
                W:resize(weightTensor:size())

                weightTensor:copy(W)
            end

            local biasTensor = biases[layer][layerId]
            if layerId == 2 or layerId == 3 then
                -- update, new memory gates
                biasTensor:fill(0)
            else
                -- reset gate
                biasTensor:fill(0)
            end
        end
    end
    model.pred:add(rnn)
    model.pred:add(nn.Dropout(opt.dropout))

    -- requires cuDNN v5:
    local rnn = cudnn.BGRU(opt.rnn_size * 2, opt.rnn_size, 1)
    local weights = rnn:weights()
    local biases = rnn:biases()
    for layer = 1, #weights do
        for layerId = 1, #weights[layer] do

            local weightTensor = weights[layer][layerId]
            if weightTensor:size(1) ~= opt.rnn_size ^ 2 then
                local stdv = math.sqrt(2 / (opt.rnn_size * 2 + opt.rnn_size))
                weightTensor:uniform(-math.sqrt(3) * stdv, math.sqrt(3) * stdv)
            else
                local fanIn = opt.rnn_size
                local fanOut = opt.rnn_size

                -- Construct random matrix
                local randMat = torch.Tensor(fanOut, fanIn):normal(0, 1)
                local U, __, V = torch.svd(randMat, 'S')

                -- Pick out orthogonal matrix
                local W
                if fanOut > fanIn then
                    W = U
                else
                    W = V:narrow(1, 1, fanOut)
                end
                -- Resize
                W:resize(weightTensor:size())

                weightTensor:copy(W)
            end

            local biasTensor = biases[layer][layerId]
            if layerId == 2 or layerId == 3 then
                -- update, new memory gates
                biasTensor:fill(0)
            elseif layerId == 1 then
                -- reset gate
                biasTensor:fill(0)
            end
        end
    end
    model.pred:add(rnn)
    model.pred:add(nn.Dropout(opt.dropout))

    model.pred:add(nn.TimeBatchWrapper {
        mod = nn.Sequential()
        :add(nn.Linear(opt.rnn_size * 2, vocab_size + 1)
            :init('weight', nninit.kaiming, { gain = 'sigmoid' })
            :init('bias', nninit.constant, 0))
    })

    -- convert to cuda
    model.pred = model.pred:cuda()
    -- cudnn.convert(model.pred, cudnn)


    model.params, model.grads = model.pred:getParameters()
    log.infof('number of parameters in full model: %d', model.params:nElement())

    exp.model = model
    exp.nIter = 10000
    return exp
end


 