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

require 'image'
require 'csvigo'
require 'pprint'
require 'nn'
require 'modules.TemporalJitter'


local data_augmentation_temporal

function math.round(x)
    return math.floor(x + 0.5)
end


function read_data(d, opt, vocab_mapping)

    if torch.uniform() < 1 / 10000 then
        collectgarbage()
    end

    local test_mode = d.test or 0
    local mode = d.mode or torch.random(1, 6)
    local flip = d.flip or 0
    local w_start, w_end
    if mode < 7 then
        w_start = d.w_start or torch.random(1, #d.words - mode + 1)
        w_end = w_start + mode - 1
    end

    local min_frame_v = 1
    local max_frame_v = 75

    local sub = ''
    local frame_v_start = -1
    local frame_v_end = -1

    -- if test mode
    if test_mode == 1 then
        frame_v_start = min_frame_v
        frame_v_end = max_frame_v

        sub = table.concat(d.words, ' ')
    else
        -- How many words to train on
        if mode == 7 then
            frame_v_start = min_frame_v
            frame_v_end = max_frame_v

            sub = table.concat(d.words, ' ')
        else

            -- Generate target
            local words = {}
            for w_i = w_start, w_end do
                table.insert(words, d.words[w_i])
            end
            sub = table.concat(words, ' ')

            frame_v_start = math.max(math.round(75 / 3000 * d.t_start[w_start]), 1)
            frame_v_end = math.min(math.round(75 / 3000 * d.t_end[w_end]), 75)

            -- If too small whole seq
            if frame_v_end - frame_v_start + 1 <= 2 then
                frame_v_start = min_frame_v
                frame_v_end = max_frame_v
                sub = table.concat(d.words, ' ')
            end
        end
    end

    -- Construct output tensor
    local x = {}
    local y = {}

    -- Put subtitle to a ByteTensor
    if opt.mode_sub == 3 then
        for char in sub:gmatch "." do
            y[#y + 1] = vocab_mapping[char]
        end
    else
        for char in sub:gmatch "%S+" do
            y[#y + 1] = vocab_mapping[char]
        end
    end

    -- Data path
    local cur_path = opt.datapath .. '/' .. d.s .. '/' .. d.v .. '/' .. opt.mode_img

    -- Load JPEGS

    -- randomly flip video
    if test_mode == 1 then
        flip = 0
    else
        flip = (flip or torch.random(0, 1))
    end

    local f_count = 1
    for f_frame = frame_v_start, frame_v_end do

        local img = image.load(cur_path .. '/' .. f_frame .. '.jpg')
        if f_frame == frame_v_start then
            x = torch.Tensor(img:size(1), frame_v_end - frame_v_start + 1, img:size(2), img:size(3))
        end

        -- Augment data
        if flip == 1 and test_mode == 0 then
            img = image.hflip(img)
        end
        x[{ {}, f_count, {}, {} }] = img
        f_count = f_count + 1
    end

    assert(x:min() >= 0 and x:max() <= 1, 'image load error')

    -- Temporal Jitter
    if opt.data_augmentation_temporal > 0 and test_mode == 0 then
        data_augmentation_temporal = (data_augmentation_temporal or nn.TemporalJitter(opt.data_augmentation_temporal))

        x = data_augmentation_temporal:forward(x)
    end

    -- Normalise
    if opt.normalise == 1 then
        x[{ { 1 } }]:add(-0.7136):div(0.113855171)
        x[{ { 2 } }]:add(-0.4906):div(0.107828568)
        x[{ { 3 } }]:add(-0.3283):div(0.0917060521)
    end


    return { x, y }
end
