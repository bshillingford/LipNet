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

--[[
Merges timesteps and batchsize together so the given module 
can process all timesteps in parallel, time/batch dims are given
into the ctor, and default to 1 and 2, respectively. The same
dimension indices are used on the input and output, although other
dimensions can change.

Dimensions can be negative: -1 is last, -2 second-last, etc.

Module can contain parameters, but input/output must be plain tensors, 
no tables.

NOTE: obviously for some modules, this has potential effects on their correctness,
e.g. for batchnorm. But for almost all other layers this is fine.
]]

local kwargs = require 'util.kwargs'
local math = require 'math'

local TimeBatchWrapper, parent = torch.class('nn.TimeBatchWrapper', 'nn.Module')
function TimeBatchWrapper:__init(opt)
  local opt = kwargs(opt, {
    {'mod', type=nn.Module},
    {'time_dim', type='integer', default=1},
    {'batch_dim', type='integer', default=2},
  })
  assert(math.abs(opt.time_dim - opt.batch_dim) == 1, 'time and batch dims must be adjacent')
  self.time_dim = opt.time_dim
  self.batch_dim = opt.batch_dim
  self.mod = opt.mod

  -- sort the two dims:
  self._dim1 = math.min(opt.time_dim, opt.batch_dim)
  self._dim2 = math.max(opt.time_dim, opt.batch_dim)
end

function TimeBatchWrapper:parameters(...)
  return self.mod:parameters(...)
end

function TimeBatchWrapper:type(...)
  self.mod:type(...)
  self.output = self.mod.output
  self.gradInput = self.mod.gradInput
  return self
end

function TimeBatchWrapper:clearState()
  self.mod:clearState()
  parent.clearState(self)
  return self
end

function TimeBatchWrapper:updateOutput(input)
  -- reshape to giant time batch:
  local dim1 = self._dim1
  if dim1<0 then dim1 = dim1 + input:dim() + 1 end
  local dim2 = self._dim2
  if dim2<0 then dim2 = dim2 + input:dim() + 1 end
  assert(dim1+1 == dim2, 'internal error: batch/time dims arent adjacent; was checked in ctor already')

  -- merge: multiply 1st dim size by 2nd dim size, delete 2nd dim
  local newsize = input:size():totable()
  newsize[dim1] = newsize[dim1] * newsize[dim2]
  table.remove(newsize, dim2)
  self._mergedsize = newsize
  local newinput = input:view(unpack(newsize))
  -- forward pass on shaped:
  local out = self.mod:updateOutput(newinput)
  
  -- split the dim again: (note: fails loudly if time/batch are different on output than input)
  local outsize = out:size():totable()
  outsize[dim1] = input:size(dim2) -- dim2, since it'll get shifted
  table.insert(outsize, dim1, input:size(dim1)) -- dim1
  self._outsplitsize = outsize

  self.output = out:view(unpack(outsize))
  return self.output
end

function TimeBatchWrapper:updateGradInput(input, gradOutput)
  self.gradInput = self.mod:updateGradInput(
      input:view(unpack(self._mergedsize)), 
      gradOutput:viewAs(self.mod.output)
    ):viewAs(input)
  return self.gradInput
end

function TimeBatchWrapper:accGradParameters(input, gradOutput, scale)
  self.mod:accGradParameters(
      input:view(unpack(self._mergedsize)),
      gradOutput:viewAs(self.mod.output),
      scale)
end


