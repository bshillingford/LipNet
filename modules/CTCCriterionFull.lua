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

local CTCCriterionFull, parent = torch.class('nn.CTCCriterionFull', 'nn.Criterion')

function CTCCriterionFull:__init(batchFirst)
    require 'warp_ctc'
    parent.__init(self)
    self.acts = torch.Tensor()
    self.batchFirst = batchFirst or false
end

function CTCCriterionFull:forward(input, target, sizes)
    return self:updateOutput(input, target, sizes)
end

function CTCCriterionFull:updateOutput(input, target, sizes)
    assert(sizes,
        "You must pass the size of each sequence in the batch as a tensor")
    local acts = self.acts
    acts:resizeAs(input):copy(input)
    if input:dim() == 3 then
        if self.batchFirst then
            acts = acts:transpose(1, 2)
            acts = self:makeContiguous(acts)
        end
        acts:view(acts, acts:size(1) * acts:size(2), -1)
    end
    assert(acts:nDimension() == 2)
    self.sizes = torch.totable(sizes)
    self.gradInput = acts.new():resizeAs(acts):zero()
    if input:type() == 'torch.CudaTensor' then
        self.output = gpu_ctc(acts, self.gradInput, target, self.sizes)
    else
        acts = acts:float()
        self.gradInput = self.gradInput:float()
        self.output = cpu_ctc(acts, self.gradInput, target, self.sizes)
    end
    return self.output
end

function CTCCriterionFull:updateGradInput(input, target)
    if input:dim() == 2 then -- (seqLen * batchSize) x outputDim
    return self.gradInput
    end
    if self.batchFirst then -- batchSize x seqLen x outputDim
    self.gradInput = inverseInterleave(self.gradInput, input:size(1))
    else -- seqLen x batchSize x outputDim
    self.gradInput:view(self.gradInput, input:size(1), input:size(2), -1)
    end
    return self.gradInput
end

function CTCCriterionFull:makeContiguous(input)
    if not input:isContiguous() then
        self._input = self._input or input.new()
        self._input:typeAs(input):resizeAs(input):copy(input)
        input = self._input
    end
    return input
end

function inverseInterleave(tensor, batchSize)
    local sizes = torch.LongStorage(3)
    sizes[1] = batchSize
    sizes[2] = tensor:size(1) / batchSize
    sizes[3] = tensor:size(2)
    local result = tensor.new():resize(sizes):zero()
    local counter = 1
    for i = 1, sizes[2] do
        for j = 1, sizes[1] do
            result[j][i]:copy(tensor[counter])
            counter = counter + 1
        end
    end
    return result
end
