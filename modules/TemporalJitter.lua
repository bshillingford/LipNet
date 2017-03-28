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

local TemporalJitter, parent = torch.class('nn.TemporalJitter', 'nn.Module')

function TemporalJitter:__init(p, l)
    parent.__init(self)
    assert(type(p) == 'number', 'input is not scalar!')
    self.p = p or 0.05
    self.compute_len = l or false

    self.train = true
end

function TemporalJitter:training()
    self.train = true
end

function TemporalJitter:evaluate()
    self.train = false
end

function TemporalJitter:updateOutput(input)
    assert(input:dim() == 5 or input:dim() == 4, 'boom')

    if self.train then

        local input_dim4 = false
        if input:dim() == 4 then
            input_dim4 = true
            input = input:view(1,input:size(1),input:size(2),input:size(3),input:size(4))
        end
        
        local len = input:size(3)

        self.output:resizeAs(input):copy(input)

        for b = 1, input:size(1) do

            local prob_del = torch.Tensor(len):bernoulli(self.p)
            local prob_dup = prob_del:index(1 ,torch.linspace(len,1,len):long())

            local output_count = 1
            for t = 1, len do
                if prob_del[t] == 0 then
                    self.output[{{b},{},{output_count},{}}] = input[{{b},{},{t},{}}]
                    output_count = output_count + 1
                end
                if prob_dup[t] == 1 and output_count > 1 then
                    self.output[{{b},{},{output_count},{}}] = self.output[{{b},{},{output_count-1},{}}]
                    output_count = output_count + 1
                end
            end
        end

        if input_dim4 then
            return self.output[1]
        end
    else
        self.output:resizeAs(input)
        self.output:copy(input)
    end
    return self.output
end

function TemporalJitter:updateGradInput(input, gradOutput)
    self.gradInput:resizeAs(gradOutput)
    self.gradInput:copy(gradOutput)
    return self.gradInput
end