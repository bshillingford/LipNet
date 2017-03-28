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

local util = {
    model={},
    subs={},
}


-- Call these before/after serialization:
function util.model.tofloat(model, tensorcache)
    tensorcache = tensorcache or {}
    cudnn.convert(model.convnet, nn) -- NOTE: requires newest version of cudnn
    return nn.utils.recursiveType(model, 'torch.FloatTensor', tensorcache)
end
function util.model.tocuda(model, tensorcache)
    tensorcache = tensorcache or {}
    cudnn.convert(model.convnet, cudnn)
    return nn.utils.recursiveType(model, 'torch.CudaTensor', tensorcache)
end

-- Given strtime (e.g. 00:01:13,345) and fps (e.g. 29.97 frames/sec),
-- Returns frame index (ONE-INDEXED), which may be fractional. Call ceil/floor as needed.
function util.subs.time2frame(fps, strtime)
    local h, m, s, ms = strtime:split(':')
    local secs = (tonumber(h)*60 + tonumber(m))*60 + tonumber(s) + tonumber(ms)/1000
    return secs/fps
end

return util
